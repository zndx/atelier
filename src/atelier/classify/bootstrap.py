"""Bootstrap convergence helpers: LLM sweep + ML validation + targeted revisit.

Phase helpers and state types used by the classification pipeline's
convergence loop (``pipeline.run_classification_pipeline``).

Three phases:

  Phase 1 — LLM sweep:
    Send ALL columns to the LLM in table-aware batches.  LLMs are strong
    zero-shot classifiers; the vast majority of labels will be correct.

  Phase 2 — ML validation:
    Run the full 6-source DST pipeline with the LLM result included.
    DST conflict K identifies columns where ML evidence disagrees with
    the LLM label.

  Phase 3 — Targeted revisit:
    Re-send only high-K disagreement columns to the LLM with enriched ML
    context (prediction, belief interval, confusable pair).  Iterate on this
    shrinking set until K converges or budget is exhausted.

Ported from signals/src/sigint/bootstrap_agent.py, adapted for atelier's
FSM, HOCON config, and classification pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from atelier.classify.belief import FrameOfDiscernment
from atelier.classify.llm_backend import LLMBackend
from atelier.classify.mass_functions import DiscountConfig
from atelier.classify.sampler import ColumnSample
from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


# ── State types ──────────────────────────────────────────────────


@dataclass
class BootstrapConfig:
    """Bootstrap convergence configuration."""

    max_iterations: int = 5
    k_threshold: float = 0.2
    coverage_target: float = 0.95
    confidence_floor: float = 0.5
    columns_per_call: int = 50
    max_total_llm_calls: int = 5000
    llm_discount: float = 0.10
    frontier_svm_retrain: bool = True
    frontier_svm_min_labels: int = 20
    # Belief-gap convergence (primary convergence criteria)
    gap_threshold: float = 0.15
    clarity_target: float = 0.10
    bel_floor: float = 0.50


def bootstrap_config_from_cfg(cfg) -> BootstrapConfig:
    """Build BootstrapConfig from an AtelierConfig."""
    return BootstrapConfig(
        max_iterations=cfg.classify_bootstrap_max_iterations,
        k_threshold=cfg.classify_bootstrap_k_threshold,
        coverage_target=cfg.classify_bootstrap_coverage_target,
        columns_per_call=cfg.classify_llm_columns_per_call,
        max_total_llm_calls=cfg.classify_bootstrap_max_total_llm_calls,
        llm_discount=cfg.classify_llm_discount,
        frontier_svm_retrain=cfg.classify_bootstrap_frontier_svm_retrain,
        frontier_svm_min_labels=cfg.classify_bootstrap_frontier_svm_min_labels,
        gap_threshold=cfg.classify_bootstrap_gap_threshold,
        clarity_target=cfg.classify_bootstrap_clarity_target,
        bel_floor=cfg.classify_bootstrap_bel_floor,
    )


@dataclass
class IterationMetrics:
    """Metrics captured at each bootstrap iteration."""

    iteration: int
    mean_k: float
    max_k: float
    disagreements: int
    coverage: float
    llm_calls: int
    # MC-aware metrics (populated when MC sampling is active)
    frontier_columns: int = 0
    propagated_columns: int = 0
    escalated_columns: int = 0
    # Belief-gap convergence (primary convergence measure)
    mean_gap: float = 0.0        # mean(Pl - Bel) for predicted categories
    mean_bel: float = 0.0        # mean belief for predicted categories
    frac_unclear: float = 0.0    # fraction of columns needing clarification


@dataclass
class BootstrapState:
    """Mutable per-column tracking across bootstrap phases."""

    iteration: int = 0
    labels: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    label_source: dict[str, str] = field(default_factory=dict)
    ml_prediction: dict[str, str] = field(default_factory=dict)
    ml_confidence: dict[str, float] = field(default_factory=dict)
    ml_conflict: dict[str, float] = field(default_factory=dict)
    ml_uncertainty: dict[str, float] = field(default_factory=dict)
    ml_belief: dict[str, float] = field(default_factory=dict)
    ml_plausibility: dict[str, float] = field(default_factory=dict)
    llm_calls_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    iteration_metrics: list[IterationMetrics] = field(default_factory=list)
    # Agent-driven convergence (populated when classify.agent.enabled=true)
    agent_reasoning: list[str] = field(default_factory=list)
    agent_turns: int = 0
    agent_converged_reason: str | None = None
    # Monte Carlo sampling metadata
    propagated_count: int = 0
    escalated_count: int = 0
    mc_strata_count: int = 0
    mc_sample_fraction: float = 1.0
    # Row-level MC: per-column label history across row-sample iterations
    row_labels_history: dict[str, list[str]] = field(default_factory=dict)
    # Frontier SVM retraining state
    svm_retrain_count: int = 0
    svm_frontier_path: str | None = None


# ── Phase helpers ────────────────────────────────────────────────


def _llm_sweep(
    state: BootstrapState,
    cfg: BootstrapConfig,
    backend: LLMBackend,
    system_prompt: str,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
) -> None:
    """Phase 1: Send all columns to LLM in table-aware batches.

    Raises ``RuntimeError`` if **every** batch call fails — this catches
    configuration errors (wrong region, bad credentials, unsupported API
    parameters) early instead of silently proceeding with zero labels and
    reporting false convergence.
    """
    by_table: dict[str, list[str]] = {}
    for name in column_names:
        table = column_table.get(name, "__flat__")
        by_table.setdefault(table, []).append(name)

    batches_attempted = 0
    batches_failed = 0
    last_error: Exception | None = None

    for table_name, table_cols in by_table.items():
        for i in range(0, len(table_cols), cfg.columns_per_call):
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                break

            chunk = table_cols[i: i + cfg.columns_per_call]
            chunk_samples = [samples[n] for n in chunk]
            tname = table_name if table_name != "__flat__" else None

            batches_attempted += 1
            try:
                response = backend.classify_batch(
                    chunk_samples, system_prompt,
                    table_name=tname,
                )
            except Exception as e:
                batches_failed += 1
                last_error = e
                logger.warning("LLM call failed (%d/%d): %s", batches_failed, batches_attempted, e)
                continue

            state.llm_calls_total += 1
            state.tokens_input += response.input_tokens
            state.tokens_output += response.output_tokens

            for c in response.classifications:
                if c.category_code and c.confidence > 0:
                    state.labels[c.column_name] = c.category_code
                    state.confidence[c.column_name] = c.confidence
                    state.label_source[c.column_name] = "llm"

    # Fail fast if every single batch call failed — this is almost
    # certainly a configuration error, not a transient issue.
    if batches_attempted > 0 and batches_failed == batches_attempted:
        raise RuntimeError(
            f"LLM sweep failed: all {batches_attempted} batch calls failed. "
            f"Last error: {last_error!r}. "
            f"Check LLM backend config (region, credentials, model ID)."
        )


def _run_ml_validation(
    state: BootstrapState,
    cfg: BootstrapConfig,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    has_embeddings: bool,
    discounts: DiscountConfig | None = None,
    propagation_discount: float | None = None,
) -> None:
    """Phase 2: Run ML classification on all columns, compute K.

    When ``propagation_discount`` is set, propagated labels (label_source
    == "propagated") use a higher discount factor, giving less mass to
    LLM evidence and more to Theta. This lets DST conflict detection
    automatically escalate uncertain propagations.
    """
    from atelier.classify.pipeline import _classify_column

    for name in column_names:
        col = samples[name]
        llm_code = state.labels.get(name)
        llm_conf = state.confidence.get(name, 0.0)

        # Use higher discount for propagated labels
        llm_disc = cfg.llm_discount
        if propagation_discount is not None and state.label_source.get(name) == "propagated":
            llm_disc = propagation_discount

        result = _classify_column(
            col, category_set, frame,
            llm_code=llm_code,
            llm_confidence=llm_conf,
            llm_discount=llm_disc,
            use_cosine=has_embeddings,
            discounts=discounts,
        )

        if result["predicted_code"]:
            state.ml_prediction[name] = result["predicted_code"]
            state.ml_confidence[name] = result["confidence"]
            state.ml_conflict[name] = result["conflict"]
            state.ml_uncertainty[name] = result["uncertainty"]
            state.ml_belief[name] = result["belief"]
            state.ml_plausibility[name] = result["plausibility"]


def _identify_disagreements(
    state: BootstrapState,
    column_names: list[str],
    cfg: BootstrapConfig,
) -> list[str]:
    """Find columns where LLM and ML disagree AND K is high."""
    disagreements = []
    for name in column_names:
        llm_code = state.labels.get(name)
        ml_code = state.ml_prediction.get(name)
        k = state.ml_conflict.get(name, 0)

        if llm_code and ml_code and llm_code != ml_code and k > cfg.k_threshold:
            disagreements.append(name)

    disagreements.sort(key=lambda n: -state.ml_conflict.get(n, 0))
    return disagreements


def _llm_revisit(
    state: BootstrapState,
    cfg: BootstrapConfig,
    backend: LLMBackend,
    system_prompt: str,
    disagreements: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
    category_set: HierarchicalCategorySet,
) -> None:
    """Phase 3: Re-classify high-K columns with enriched ML context."""
    revisit_context: dict[str, dict] = {}
    for name in disagreements:
        ml_code = state.ml_prediction.get(name, "")
        ml_cat = category_set.by_code.get(ml_code) or category_set.all_by_code.get(ml_code)
        ml_label = ml_cat.label if ml_cat else ml_code

        llm_code = state.labels.get(name, "")
        llm_cat = category_set.by_code.get(llm_code) or category_set.all_by_code.get(llm_code)
        llm_label = llm_cat.label if llm_cat else llm_code

        confusable = f"{ml_label} / {llm_label}" if ml_label and llm_label else ""

        # Enrich revisit context with concrete pattern/feature evidence
        # so the LLM can reason about deterministic signals.
        col = samples.get(name)
        pattern_signals: dict[str, float] = {}
        value_description = ""
        if col:
            from atelier.classify.features import detect_patterns, _generate_value_description
            pattern_signals = detect_patterns(col.values)
            value_description = _generate_value_description(
                col.values, col.column_type, pattern_signals,
            )

        revisit_context[name] = {
            "ml_prediction": ml_label or ml_code,
            "belief": 0.0,
            "plausibility": 0.0,
            "conflict": state.ml_conflict.get(name, 0),
            "confusable": confusable,
            "pattern_signals": pattern_signals,
            "value_description": value_description,
            "previous": {
                "code": llm_code,
                "confidence": state.confidence.get(name, 0),
            },
        }

    # Group by table for coherent revisit context
    by_table: dict[str, list[str]] = {}
    for name in disagreements:
        table = column_table.get(name, "__flat__")
        by_table.setdefault(table, []).append(name)

    for table_name, table_cols in by_table.items():
        for i in range(0, len(table_cols), cfg.columns_per_call):
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                return

            chunk = table_cols[i: i + cfg.columns_per_call]
            chunk_samples = [samples[n] for n in chunk]
            chunk_context = {n: revisit_context[n] for n in chunk}
            tname = table_name if table_name != "__flat__" else None

            try:
                response = backend.classify_batch(
                    chunk_samples, system_prompt,
                    revisit_context=chunk_context,
                    table_name=tname,
                )
            except Exception as e:
                logger.warning("LLM revisit call failed: %s", e)
                continue

            state.llm_calls_total += 1
            state.tokens_input += response.input_tokens
            state.tokens_output += response.output_tokens

            for c in response.classifications:
                if c.category_code and c.confidence > 0:
                    state.labels[c.column_name] = c.category_code
                    state.confidence[c.column_name] = c.confidence
                    state.label_source[c.column_name] = "llm_revisit"


def _coverage(state: BootstrapState, column_names: list[str]) -> float:
    """Fraction of columns with a label."""
    if not column_names:
        return 1.0
    return sum(1 for n in column_names if n in state.labels) / len(column_names)


def _mean_k(state: BootstrapState, column_names: list[str]) -> float:
    """Mean DST conflict K across all labeled columns."""
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 0.0
    return sum(state.ml_conflict.get(n, 0) for n in labeled) / len(labeled)


def _max_k(state: BootstrapState, column_names: list[str]) -> float:
    """Max DST conflict K across all labeled columns."""
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 0.0
    return max(state.ml_conflict.get(n, 0) for n in labeled)


# ── Belief-gap convergence ──────────────────────────────────────────
#
# K measures source disagreement; [Bel, Pl] measures prediction certainty.
# A column can have K=0.9 but Bel=0.95 — the sources fought hard but the
# winner is clear. The belief-gap (Pl - Bel) is the primary convergence
# measure: when it's small, the prediction is settled regardless of K.


def _mean_gap(state: BootstrapState, column_names: list[str]) -> float:
    """Mean uncertainty gap (Pl - Bel) for predicted categories.

    Low mean gap = predictions are settled. This is the primary
    convergence measure — it directly answers "how certain are we
    about each column's classification?"
    """
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 1.0
    gaps = [
        state.ml_plausibility.get(n, 1.0) - state.ml_belief.get(n, 0.0)
        for n in labeled
    ]
    return sum(gaps) / len(gaps)


def _mean_bel(state: BootstrapState, column_names: list[str]) -> float:
    """Mean belief for predicted categories."""
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 0.0
    return sum(state.ml_belief.get(n, 0.0) for n in labeled) / len(labeled)


def _frac_needing_clarification(
    state: BootstrapState,
    column_names: list[str],
    gap_threshold: float = 0.3,
    bel_floor: float = 0.5,
) -> float:
    """Fraction of columns needing clarification (high gap OR low belief).

    A column needs clarification when its predicted category has either:
    - gap > gap_threshold (prediction not tight enough)
    - bel < bel_floor (not enough evidence for the prediction)
    """
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 1.0
    needing = 0
    for n in labeled:
        gap = state.ml_plausibility.get(n, 1.0) - state.ml_belief.get(n, 0.0)
        bel = state.ml_belief.get(n, 0.0)
        if gap > gap_threshold or bel < bel_floor:
            needing += 1
    return needing / len(labeled)


def _identify_uncertain_columns(
    state: BootstrapState,
    column_names: list[str],
    cfg: BootstrapConfig,
) -> list[str]:
    """Find columns where the prediction is uncertain.

    Targets columns for LLM revisit based on belief-gap metrics rather
    than K-based disagreement. A column is uncertain when:
    - Pl - Bel > 0.3 (wide belief interval), OR
    - Bel < bel_floor (insufficient supporting evidence)

    Sorted by gap descending (most uncertain first).
    """
    uncertain = []
    for name in column_names:
        if name not in state.labels:
            continue
        gap = state.ml_plausibility.get(name, 1.0) - state.ml_belief.get(name, 0.0)
        bel = state.ml_belief.get(name, 0.0)
        if gap > 0.3 or bel < cfg.bel_floor:
            uncertain.append(name)
    uncertain.sort(
        key=lambda n: -(state.ml_plausibility.get(n, 1.0) - state.ml_belief.get(n, 0.0))
    )
    return uncertain


def record_iteration_metrics(
    state: BootstrapState,
    column_names: list[str],
    disagreement_count: int,
) -> IterationMetrics:
    """Capture metrics for the current iteration."""
    metrics = IterationMetrics(
        iteration=state.iteration,
        mean_k=round(_mean_k(state, column_names), 4),
        max_k=round(_max_k(state, column_names), 4),
        disagreements=disagreement_count,
        coverage=round(_coverage(state, column_names), 4),
        llm_calls=state.llm_calls_total,
        mean_gap=round(_mean_gap(state, column_names), 4),
        mean_bel=round(_mean_bel(state, column_names), 4),
        frac_unclear=round(_frac_needing_clarification(
            state, column_names,
        ), 4),
    )
    state.iteration_metrics.append(metrics)
    logger.info(
        "Iteration %d: mean_K=%.4f mean_gap=%.4f mean_bel=%.4f "
        "unclear=%.1f%% disagreements=%d coverage=%.1f%%",
        metrics.iteration, metrics.mean_k, metrics.mean_gap, metrics.mean_bel,
        metrics.frac_unclear * 100, metrics.disagreements, metrics.coverage * 100,
    )
    return metrics


def k_convergence_rate(state: BootstrapState) -> float:
    """Slope of mean K over iterations. Negative = improving.

    Computed as simple linear slope of mean_k values. Returns 0.0
    if fewer than 2 iterations recorded.
    """
    metrics = state.iteration_metrics
    if len(metrics) < 2:
        return 0.0
    return (metrics[-1].mean_k - metrics[0].mean_k) / (len(metrics) - 1)


def gap_convergence_rate(state: BootstrapState) -> float:
    """Slope of mean uncertainty gap over iterations. Negative = improving."""
    metrics = state.iteration_metrics
    if len(metrics) < 2:
        return 0.0
    return (metrics[-1].mean_gap - metrics[0].mean_gap) / (len(metrics) - 1)


def should_stop_early(state: BootstrapState) -> bool:
    """True when uncertainty gap is no longer decreasing.

    Uses the proof-of-progress paradigm: as long as the mean gap is
    shrinking, the agent is making verifiable progress toward settled
    predictions. When it plateaus for 2 consecutive iterations, stop.

    Falls back to K-based plateau detection when gap data is unavailable
    (e.g., iteration 0 before ML validation populates belief/plausibility).
    """
    metrics = state.iteration_metrics
    if len(metrics) < 3:
        return False

    # Prefer gap-based plateau detection
    if metrics[-1].mean_gap > 0:
        delta_1 = metrics[-1].mean_gap - metrics[-2].mean_gap
        delta_2 = metrics[-2].mean_gap - metrics[-3].mean_gap
        return delta_1 >= -1e-6 and delta_2 >= -1e-6

    # Fallback: K-based plateau (pre-ML-validation iterations)
    delta_1 = metrics[-1].mean_k - metrics[-2].mean_k
    delta_2 = metrics[-2].mean_k - metrics[-3].mean_k
    return delta_1 >= -1e-6 and delta_2 >= -1e-6


def row_stability(state: BootstrapState, name: str) -> float:
    """Fraction of iterations that produced the most common label. 1.0 = stable.

    Used by the row-level MC adaptive escalation: columns where different
    row subsets produce different classifications have row_stability < 1.0,
    indicating the column type depends on which values are observed.
    """
    history = state.row_labels_history.get(name, [])
    if len(history) < 2:
        return 1.0
    from collections import Counter
    most_common_count = Counter(history).most_common(1)[0][1]
    return most_common_count / len(history)
