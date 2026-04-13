"""Bootstrap convergence loop: LLM sweep + ML validation + targeted revisit.

Three-phase architecture:

  Phase 1 — LLM sweep:
    Send ALL columns to the LLM in table-aware batches.  LLMs are strong
    zero-shot classifiers; the vast majority of labels will be correct.

  Phase 2 — ML validation:
    Run the existing 3-source ML pipeline (cosine + pattern + name_match)
    with the LLM result as a 4th evidence source.  DST conflict K identifies
    columns where ML evidence disagrees with the LLM label.

  Phase 3 — Targeted revisit:
    Re-send only high-K disagreement columns to the LLM with enriched ML
    context (prediction, belief interval, confusable pair).  Iterate on this
    shrinking set until K converges or budget is exhausted.

Ported from signals/src/sigint/bootstrap_agent.py, adapted for atelier's
FSM, HOCON config, and classification pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from atelier.classify.belief import (
    FrameOfDiscernment,
    HierarchicalClassification,
)
from atelier.classify.features import extract_features
from atelier.classify.fsm import AgentFSM, FSMState
from atelier.classify.llm_backend import (
    ColumnClassification,
    LLMBackend,
    build_batch_user_prompt,
    build_category_table,
    build_system_prompt,
    create_backend_from_cfg,
)
from atelier.classify.mass_functions import (
    catboost_to_mass,
    cosine_to_mass,
    llm_to_mass,
    name_match_to_mass,
    pattern_to_mass,
    svm_to_mass,
)
from atelier.classify.pipeline import (
    _evaluate_results,
    _is_vacuous,
    _load_vocabulary,
    _mass_summary,
    _write_parquet,
)
from atelier.classify.sampler import (
    ColumnSample,
    TableSample,
    discover_tables,
    load_all_mock_samples,
    load_annotations_from_hive,
    sample_table_metadata,
)
from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


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


def bootstrap_config_from_cfg(cfg) -> BootstrapConfig:
    """Build BootstrapConfig from an AtelierConfig."""
    return BootstrapConfig(
        max_iterations=cfg.classify_bootstrap_max_iterations,
        k_threshold=cfg.classify_bootstrap_k_threshold,
        coverage_target=cfg.classify_bootstrap_coverage_target,
        columns_per_call=cfg.classify_llm_columns_per_call,
        max_total_llm_calls=cfg.classify_bootstrap_max_total_llm_calls,
        llm_discount=cfg.classify_llm_discount,
    )


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
    llm_calls_total: int = 0
    tokens_input: int = 0
    tokens_output: int = 0


@dataclass(frozen=True)
class BootstrapResult:
    """Final output of the bootstrap convergence loop."""

    ground_truth: dict[str, str]
    confidence_map: dict[str, float]
    source_map: dict[str, str]
    converged: bool
    final_coverage: float
    final_mean_k: float
    iterations: int
    llm_calls: int
    tokens_input: int
    tokens_output: int


# ── Pipeline entry point ─────────────────────────────────────────


def run_bootstrap_pipeline(
    cfg,
    fsm: AgentFSM,
    *,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int = 50,
    tables_limit: int = 100,
    use_mock: bool = False,
    llm_backend: LLMBackend | None = None,
) -> dict[str, Any]:
    """Run the bootstrap convergence pipeline.

    This wraps the existing ML pipeline with an LLM-driven convergence loop.
    The LLM provides a 4th evidence source; DST conflict K drives iteration.

    Args:
        cfg: AtelierConfig.
        fsm: AgentFSM for state tracking.
        llm_backend: LLM backend (injected for testing; created from config if None).
        use_mock: Force mock data for devenv/CI.
    """
    run = fsm.start_run(config={
        "connection_name": connection_name,
        "database": database,
        "sample_size": sample_size,
        "tables_limit": tables_limit,
        "use_mock": use_mock,
        "pipeline": "bootstrap",
    })
    run_id = run.id

    build_dir = _PROJECT_ROOT / "build"
    results_dir = build_dir / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ── LOADING_VOCAB ────────────────────────────────────────
        fsm.advance(run_id, FSMState.LOADING_VOCAB, progress={"step": "loading_vocab"})
        category_set = _load_vocabulary(cfg, build_dir, connection_name, use_mock)
        logger.info("Loaded %d leaf categories", len(category_set.categories))

        if not isinstance(category_set, HierarchicalCategorySet):
            raise RuntimeError("Expected HierarchicalCategorySet")

        frame = FrameOfDiscernment(category_set)
        fsm.advance(run_id, FSMState.DISCOVERING, progress={
            "categories_loaded": len(category_set.categories),
        })

        # ── DISCOVERING ──────────────────────────────────────────
        if use_mock:
            table_names = [t.name for t in load_all_mock_samples()]
        else:
            table_names = discover_tables(
                cfg, connection_name, database, limit=tables_limit
            )
        logger.info("Discovered %d tables", len(table_names))

        fsm.advance(run_id, FSMState.SAMPLING, progress={
            "tables_discovered": len(table_names),
        })

        # ── SAMPLING ─────────────────────────────────────────────
        all_samples: list[TableSample] = []
        if use_mock:
            all_samples = load_all_mock_samples()
        else:
            for tname in table_names:
                try:
                    ts = sample_table_metadata(
                        cfg, tname, connection_name, database, sample_size
                    )
                    all_samples.append(ts)
                except Exception as exc:
                    logger.warning("Failed to sample %s: %s", tname, exc)

        # Flatten to column list with table mapping
        all_columns: list[ColumnSample] = []
        column_table: dict[str, str] = {}
        for ts in all_samples:
            for col in ts.columns:
                all_columns.append(col)
                column_table[col.name] = ts.name

        total_columns = len(all_columns)
        logger.info("Sampled %d columns across %d tables", total_columns, len(all_samples))

        # Build samples dict for lookup
        samples_by_name: dict[str, ColumnSample] = {c.name: c for c in all_columns}
        column_names = list(samples_by_name.keys())

        # ── Create LLM backend ───────────────────────────────────
        if llm_backend is None:
            llm_backend = create_backend_from_cfg(cfg)

        # Build bootstrap config and system prompt
        boot_cfg = bootstrap_config_from_cfg(cfg)
        category_table = build_category_table(category_set)
        system_prompt = build_system_prompt(category_table)

        # Try sentence-transformers for cosine
        has_embeddings = False
        try:
            from atelier.classify.embedding import classify_cosine
            has_embeddings = True
        except ImportError:
            logger.warning("sentence-transformers not available; using name+pattern only")

        # ── LLM SWEEP ────────────────────────────────────────────
        fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
            "columns_total": total_columns,
            "phase": "llm_sweep",
        })

        state = BootstrapState()

        _llm_sweep(
            state, boot_cfg, llm_backend, system_prompt,
            column_names, samples_by_name, column_table,
        )

        coverage = _coverage(state, column_names)
        logger.info(
            "LLM sweep: labeled %d/%d (coverage=%.1f%%, calls=%d)",
            len(state.labels), total_columns,
            coverage * 100, state.llm_calls_total,
        )

        # ── VALIDATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.VALIDATING, progress={
            "phase": "ml_validation",
            "llm_labeled": len(state.labels),
            "coverage": round(coverage, 4),
        })

        _run_ml_validation(
            state, boot_cfg, column_names, samples_by_name,
            category_set, frame, has_embeddings,
        )

        disagreements = _identify_disagreements(state, column_names, boot_cfg)
        mean_k = _mean_k(state, column_names)

        logger.info(
            "ML validation: mean K=%.3f, disagreements=%d",
            mean_k, len(disagreements),
        )

        # ── TARGETED REVISIT LOOP ────────────────────────────────
        for iteration in range(1, boot_cfg.max_iterations + 1):
            if not disagreements:
                logger.info("No disagreements — converged")
                break

            if state.llm_calls_total >= boot_cfg.max_total_llm_calls:
                logger.info("Budget exhausted (%d calls)", state.llm_calls_total)
                break

            if mean_k < boot_cfg.k_threshold:
                logger.info("Mean K=%.3f < threshold — converged", mean_k)
                break

            state.iteration = iteration

            # Back to LLM_SWEEP for revisit
            fsm.advance(run_id, FSMState.LLM_SWEEP, progress={
                "phase": "revisit",
                "iteration": iteration,
                "disagreements": len(disagreements),
                "mean_k": round(mean_k, 4),
            })

            _llm_revisit(
                state, boot_cfg, llm_backend, system_prompt,
                disagreements, samples_by_name, column_table, category_set,
            )

            # Back to VALIDATING
            fsm.advance(run_id, FSMState.VALIDATING, progress={
                "phase": "revalidation",
                "iteration": iteration,
                "llm_calls": state.llm_calls_total,
            })

            _run_ml_validation(
                state, boot_cfg, column_names, samples_by_name,
                category_set, frame, has_embeddings,
            )

            disagreements = _identify_disagreements(state, column_names, boot_cfg)
            mean_k = _mean_k(state, column_names)
            coverage = _coverage(state, column_names)

            logger.info(
                "Revisit %d: mean K=%.3f, disagreements=%d, coverage=%.1f%%, calls=%d",
                iteration, mean_k, len(disagreements),
                coverage * 100, state.llm_calls_total,
            )

        # ── FINAL CLASSIFICATION PASS ────────────────────────────
        # Run full pipeline with LLM evidence included
        coverage = _coverage(state, column_names)
        mean_k = _mean_k(state, column_names)
        converged = coverage >= boot_cfg.coverage_target and mean_k < boot_cfg.k_threshold

        fsm.advance(run_id, FSMState.CLASSIFYING, progress={
            "phase": "final_classification",
            "converged": converged,
            "mean_k": round(mean_k, 4),
            "coverage": round(coverage, 4),
        })

        # Build final classifications with LLM evidence
        classifications: list[dict[str, Any]] = []
        for col in all_columns:
            llm_code = state.labels.get(col.name)
            llm_conf = state.confidence.get(col.name, 0.0)
            result = _classify_column_with_llm(
                col, category_set, frame,
                llm_code=llm_code,
                llm_confidence=llm_conf,
                llm_discount=boot_cfg.llm_discount,
                use_cosine=has_embeddings,
            )
            classifications.append(result)

        fsm.advance(run_id, FSMState.FUSING, progress={
            "columns_classified": len(classifications),
        })

        # ── EVALUATING ───────────────────────────────────────────
        fsm.advance(run_id, FSMState.EVALUATING, progress={
            "columns_fused": len(classifications),
        })

        summary = _evaluate_results(classifications)
        summary["converged"] = converged
        summary["bootstrap_iterations"] = state.iteration
        summary["llm_calls"] = state.llm_calls_total
        summary["tokens_input"] = state.tokens_input
        summary["tokens_output"] = state.tokens_output
        summary["mean_k"] = round(mean_k, 4)
        summary["bootstrap_coverage"] = round(coverage, 4)

        # Write results
        results_path = results_dir / "classifications.json"
        results_path.write_text(json.dumps(classifications, indent=2, default=str) + "\n")

        parquet_path = _write_parquet(classifications, results_dir / "atelier_embeddings.parquet")

        fsm.advance(run_id, FSMState.CONVERGED, progress={
            **summary,
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
        }, result_path=str(parquet_path) if parquet_path else str(results_path))

        return {
            "run_id": run_id,
            "state": "CONVERGED",
            "classifications": len(classifications),
            "result_path": str(results_path),
            "parquet_path": str(parquet_path) if parquet_path else None,
            **summary,
        }

    except Exception as exc:
        logger.exception("Bootstrap pipeline failed: %s", exc)
        try:
            fsm.advance(run_id, FSMState.ERROR, error=str(exc))
        except ValueError:
            pass
        return {
            "run_id": run_id,
            "state": "ERROR",
            "error": str(exc),
        }


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
    """Phase 1: Send all columns to LLM in table-aware batches."""
    by_table: dict[str, list[str]] = {}
    for name in column_names:
        table = column_table.get(name, "__flat__")
        by_table.setdefault(table, []).append(name)

    for table_name, table_cols in by_table.items():
        for i in range(0, len(table_cols), cfg.columns_per_call):
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                return

            chunk = table_cols[i: i + cfg.columns_per_call]
            chunk_samples = [samples[n] for n in chunk]
            tname = table_name if table_name != "__flat__" else None

            try:
                response = backend.classify_batch(
                    chunk_samples, system_prompt,
                    table_name=tname,
                )
            except Exception as e:
                logger.warning("LLM call failed: %s", e)
                continue

            state.llm_calls_total += 1
            state.tokens_input += response.input_tokens
            state.tokens_output += response.output_tokens

            for c in response.classifications:
                if c.category_code and c.confidence > 0:
                    state.labels[c.column_name] = c.category_code
                    state.confidence[c.column_name] = c.confidence
                    state.label_source[c.column_name] = "llm"


def _run_ml_validation(
    state: BootstrapState,
    cfg: BootstrapConfig,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    has_embeddings: bool,
) -> None:
    """Phase 2: Run ML classification on all columns, compute K."""
    for name in column_names:
        col = samples[name]
        llm_code = state.labels.get(name)
        llm_conf = state.confidence.get(name, 0.0)

        result = _classify_column_with_llm(
            col, category_set, frame,
            llm_code=llm_code,
            llm_confidence=llm_conf,
            llm_discount=cfg.llm_discount,
            use_cosine=has_embeddings,
        )

        if result["predicted_code"]:
            state.ml_prediction[name] = result["predicted_code"]
            state.ml_confidence[name] = result["confidence"]
            state.ml_conflict[name] = result["conflict"]
            state.ml_uncertainty[name] = result["uncertainty"]


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

        revisit_context[name] = {
            "ml_prediction": ml_label or ml_code,
            "belief": 0.0,
            "plausibility": 0.0,
            "conflict": state.ml_conflict.get(name, 0),
            "confusable": confusable,
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


# ── Column classification with LLM evidence ─────────────────────


def _classify_column_with_llm(
    col: ColumnSample,
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    *,
    llm_code: str | None = None,
    llm_confidence: float = 0.0,
    llm_alternatives: list[dict] | None = None,
    llm_discount: float = 0.10,
    use_cosine: bool = True,
) -> dict[str, Any]:
    """Classify a column using ML sources + optional LLM evidence.

    Extends pipeline._classify_column() by adding llm_to_mass()
    to the source_masses dict before fusion.
    """
    features = extract_features(
        column_name=col.name,
        column_type=col.column_type,
        values=col.values,
        siblings=col.siblings,
        source_table=col.table_name,
        total_count=col.total_count,
        null_count=col.null_count,
    )

    source_masses: dict[str, Any] = {}

    # 1. Name matching
    name_mass = name_match_to_mass(col.name, frame, category_set)
    if not _is_vacuous(name_mass):
        source_masses["name_match"] = name_mass

    # 2. Pattern detection
    pattern_mass = pattern_to_mass(features.pattern_signals, frame)
    if not _is_vacuous(pattern_mass):
        source_masses["pattern"] = pattern_mass

    # 3. Cosine similarity
    if use_cosine:
        try:
            from atelier.classify.embedding import classify_cosine as _cosine
            similarities = _cosine(features, category_set)
            cosine_mass = cosine_to_mass(similarities, frame)
            source_masses["cosine"] = cosine_mass
        except Exception:
            pass

    # 4. LLM evidence
    if llm_code:
        llm_mass = llm_to_mass(
            llm_code, llm_confidence,
            llm_alternatives or [],
            frame, discount=llm_discount,
        )
        if not _is_vacuous(llm_mass):
            source_masses["llm"] = llm_mass

    # 5. CatBoost (if model available)
    try:
        from atelier.classify.ml_inference import predict_catboost
        cb_result = predict_catboost(features, category_set)
        if cb_result:
            proba, variance = cb_result
            cb_mass = catboost_to_mass(proba, frame, variance)
            if not _is_vacuous(cb_mass):
                source_masses["catboost"] = cb_mass
    except Exception:
        pass

    # 6. SVM (if model available)
    try:
        from atelier.classify.ml_inference import predict_svm
        svm_proba = predict_svm(features)
        if svm_proba:
            svm_mass = svm_to_mass(svm_proba, frame)
            if not _is_vacuous(svm_mass):
                source_masses["svm"] = svm_mass
    except Exception:
        pass

    # Fuse all evidence
    if not source_masses:
        return {
            "table_name": col.table_name,
            "column_name": col.name,
            "column_type": col.column_type,
            "predicted_code": None,
            "predicted_label": "",
            "confidence": 0.0,
            "belief": 0.0,
            "plausibility": 1.0,
            "uncertainty": 1.0,
            "conflict": 0.0,
            "needs_clarification": True,
            "evidence": "",
            "evidence_sources": {},
            "pattern_signals": features.pattern_signals,
            "ground_truth": col.ground_truth,
            "is_correct": None,
        }

    hc = HierarchicalClassification.from_combined_evidence(
        source_masses=source_masses,
        frame=frame,
        category_set=category_set,
    )

    best_code = hc.category.code
    bel, pl = hc.interval_at(best_code)

    return {
        "table_name": col.table_name,
        "column_name": col.name,
        "column_type": col.column_type,
        "predicted_code": best_code,
        "predicted_label": hc.category.label,
        "confidence": hc.confidence,
        "belief": round(bel, 4),
        "plausibility": round(pl, 4),
        "uncertainty": round(pl - bel, 4),
        "conflict": hc.conflict,
        "needs_clarification": hc.needs_clarification,
        "evidence": hc.evidence,
        "evidence_sources": {name: _mass_summary(ba) for name, ba in source_masses.items()},
        "pattern_signals": features.pattern_signals,
        "ground_truth": col.ground_truth,
        "is_correct": (
            col.ground_truth == best_code
            if col.ground_truth and best_code
            else None
        ),
    }
