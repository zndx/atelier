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
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from atelier.classify.belief import FrameOfDiscernment
from atelier.classify.llm_backend import LLMBackend
from atelier.classify.mass_functions import DiscountConfig
from atelier.classify.sampler import ColumnSample
from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


# ── Error classification for halving retry ───────────────────────
#
# Recoverable failures (transient or response-shape issues) halve the
# batch and recurse — at min_columns_per_call=1 this degenerates into
# per-column best-effort with each failure recorded, so overwatch sees
# the specific column that blocked rather than a silent 25-column drop.
#
# Fatal failures (auth, invariant violations) abort the sweep with a
# clear error — halving can't fix a 401 and would just burn budget.

_FATAL_CLASS_NAMES: frozenset[str] = frozenset({
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",         # model ID wrong — halving won't help
    "BadRequestError",       # invariant / schema violation
    "InvalidRequestError",
    "ClientError",           # boto core-level auth/permission wrapper
    "UnrecognizedClientException",
    "AccessDeniedException",
})

_FATAL_MSG_SUBSTRINGS: tuple[str, ...] = (
    "invalid api key",
    "invalid_api_key",
    "unauthorized",
    "forbidden",
    "access denied",
    "accessdenied",
    "authentication",
    "credential",
    "not authorized",
    "model not found",
    "not a valid model",
    "resource not found",
)


def _classify_error(exc: Exception) -> str:
    """Return ``"fatal"`` or ``"recoverable"`` for an LLM call exception.

    Fatal errors abort the sweep immediately — halving a batch cannot
    repair auth problems or a wrong model ID, and silently retrying
    would waste compute budget while producing a false "partial
    coverage" result.  Everything else (timeouts, 5xx, JSON parse
    issues, rate limits, connection resets) halves and retries down
    to the configured ``min_columns_per_call`` (default 1).
    """
    cls_name = type(exc).__name__
    if cls_name in _FATAL_CLASS_NAMES:
        return "fatal"
    msg = str(exc).lower()
    for marker in _FATAL_MSG_SUBSTRINGS:
        if marker in msg:
            return "fatal"
    return "recoverable"


class FatalLLMError(RuntimeError):
    """Raised when the LLM backend reports an unrecoverable error.

    Halving cannot fix auth, permission, or invariant violations, so
    the sweep aborts rather than continuing with partial labels that
    would be indistinguishable from clean output to downstream stages.
    """


class SweepDeadlineError(FatalLLMError):
    """Raised when the wall-clock sweep deadline is exceeded.

    Inherits from ``FatalLLMError`` so existing ``except FatalLLMError``
    handlers in ``_llm_sweep`` and callers route it through the same
    abort-with-ERROR path — the subclass exists for log-readability
    (operators can distinguish "hit the 30-min wall" from "auth
    rejected the key") and for any watchdog that wants to branch on
    type rather than parse the error string.
    """


# ── State types ──────────────────────────────────────────────────


@dataclass
class BootstrapConfig:
    """Bootstrap convergence configuration."""

    max_iterations: int = 5
    k_threshold: float = 0.2
    coverage_target: float = 1.0
    confidence_floor: float = 0.5
    columns_per_call: int = 50
    # Floor for exhaustive-halving retry.  A failing batch is repeatedly
    # halved and recursed; 1 means "fall back to per-column attempts" so
    # no column is silently dropped.  Bump only when the backend cannot
    # tolerate single-column calls.
    min_columns_per_call: int = 1
    # Cap on *successful* LLM responses (preserved semantics — existing
    # users tune this to limit the cost of a healthy sweep).  A failing
    # endpoint never increments this so it is not a retry-storm brake;
    # see ``max_total_llm_attempts`` for that role.
    max_total_llm_calls: int = 5000
    # Cap on every entry into ``_classify_batch_with_retry`` — success,
    # failure, halved retry, truncated.  Defaults to 2× calls so a
    # healthy sweep (where attempts ≈ calls) never trips it; on a
    # blackholed endpoint where attempts pile up without calls, this is
    # the gate that stops the retry storm.
    max_total_llm_attempts: int = 10000
    llm_discount: float = 0.15
    frontier_svm_retrain: bool = True
    frontier_svm_min_labels: int = 20
    # Belief-gap convergence (primary convergence criteria)
    gap_threshold: float = 0.15
    clarity_target: float = 0.10
    bel_floor: float = 0.50
    # Minimum independent-tier consensus mass required to fire a revisit
    # on the basis of indep-tier vs LLM disagreement.  Below this floor
    # the cosine/pattern/name_match signal is too diffuse to act on
    # alone (the existing high-K branch still fires when relevant).
    indep_revisit_mass_threshold: float = 0.45
    # Wall-clock deadline for the LLM sweep.  ``0`` (default) disables
    # the brake — the attempts cap and consecutive-failure breaker cover
    # the dead-endpoint case, and a healthy-but-slow sweep (large vocab
    # + Bedrock latency) should not be guillotined by a fixed clock.
    # Set a positive value in HOCON / Settings to re-enable as a hard
    # ceiling; a ``SweepDeadlineError`` is raised on overrun.
    sweep_deadline_s: float = 0.0
    # Stop halving after this many consecutive recoverable failures in
    # a row — at that point the endpoint is effectively dead and each
    # further retry just compounds the outage.  Counts failures at any
    # depth (a halved_on_error at depth 3 increments; a success resets).
    max_consecutive_halve_failures: int = 8


def bootstrap_config_from_cfg(cfg) -> BootstrapConfig:
    """Build BootstrapConfig from an AtelierConfig."""
    max_calls = cfg.classify_bootstrap_max_total_llm_calls
    return BootstrapConfig(
        max_iterations=cfg.classify_bootstrap_max_iterations,
        k_threshold=cfg.classify_bootstrap_k_threshold,
        coverage_target=cfg.classify_bootstrap_coverage_target,
        columns_per_call=cfg.classify_llm_columns_per_call,
        min_columns_per_call=getattr(cfg, "classify_llm_min_columns_per_call", 1),
        max_total_llm_calls=max_calls,
        max_total_llm_attempts=getattr(
            cfg, "classify_bootstrap_max_total_llm_attempts", 2 * max_calls,
        ),
        sweep_deadline_s=getattr(
            cfg, "classify_bootstrap_sweep_deadline_s", 0.0,
        ),
        max_consecutive_halve_failures=getattr(
            cfg, "classify_bootstrap_max_consecutive_halve_failures", 8,
        ),
        llm_discount=cfg.classify_llm_discount,
        frontier_svm_retrain=cfg.classify_bootstrap_frontier_svm_retrain,
        frontier_svm_min_labels=cfg.classify_bootstrap_frontier_svm_min_labels,
        gap_threshold=cfg.classify_bootstrap_gap_threshold,
        clarity_target=cfg.classify_bootstrap_clarity_target,
        bel_floor=cfg.classify_bootstrap_bel_floor,
        indep_revisit_mass_threshold=getattr(
            cfg, "classify_bootstrap_indep_revisit_mass_threshold", 0.45,
        ),
    )


@dataclass
class IterationMetrics:
    """Metrics captured at each bootstrap iteration.

    The bootstrap loop is iterative refinement on a belief-assignment
    vector B over columns: B_{n+1} = T(B_n), where T composes the
    LLM sweep, ML validation, DST fusion, and targeted revisit on
    disagreement.  Per Saad 2003 §4.1 (*Iterative Methods for Sparse
    Linear Systems*), an iterative method's primary diagnostic is
    its **residual norm** ``‖r(B)‖`` — a scalar measure of distance
    from the fixed point — and the **contraction factor** ``ρ_n =
    ‖r_{n+1}‖ / ‖r_n‖``, which distinguishes contractive (ρ < 1,
    converging) from stalled (ρ → 1) from diverging (ρ > 1) regimes.

    ``residual_norm`` and ``contraction_rate`` here implement that
    diagnostic for Atelier's pipeline.  See
    ``docs/src/architecture/dst-evidence-independence.md``.
    """

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
    # Numerical-methods convergence diagnostics.  ``residual_norm`` is
    # a unified scalar combining gap, K, frac_unclear, and indep-tier
    # disagreement (each normalized so 1.0 means "at threshold" and 0
    # means "fully converged").  ``contraction_rate`` is ρ_n =
    # residual_norm / prior_residual_norm; <1 means converging, →1
    # stalled, >1 diverging (Saad 2003 §4.1).
    residual_norm: float = 0.0
    contraction_rate: float = 0.0
    indep_tier_disagreement_frac: float = 0.0


@dataclass
class ColumnResidualSnapshot:
    """Per-column residual at one iteration of the bootstrap loop.

    Captured by ``record_iteration_metrics`` after each iteration's
    ML validation completes, so the snapshot reflects the post-
    iteration state.  Together with the corpus-wide
    ``IterationMetrics``, snapshots form the column-major view that
    operators / overwatch / future acceleration schemes consume.

    The corpus-wide ``IterationMetrics`` is time-major (one row per
    iteration with aggregates across columns).  ``column_history``
    on ``BootstrapState`` is column-major (one list per column with
    snapshots across iterations).  Both views are needed: corpus
    aggregates drive the headline convergence loop; per-column
    trajectories let operators see which specific columns are
    converging vs stalling — and let future acceleration schemes
    (bandit revisit ordering, Aitken Δ² early-stop, per-column
    Anderson on oscillating populations) operate on real per-column
    data.
    """

    iteration: int
    gap: float                  # Pl − Bel for the predicted code
    bel: float
    K: float                    # ml_conflict (fused conflict)
    indep_top1_code: str | None
    indep_top1_mass: float
    label: str | None           # state.labels[name] at snapshot time
    label_source: str | None    # "llm" | "llm_revisit" | "propagated"
    revisited: bool             # True iff this iteration's _llm_revisit touched it


@dataclass
class BatchAttempt:
    """Audit record for a single LLM batch call (success, truncated, or failed).

    Populated by ``_classify_batch_with_retry``; exposed to overwatch
    and to the Status page so the reason every column was (or wasn't)
    classified is inspectable rather than hidden behind a warning log.
    """

    batch_index: int           # zero-based index within the sweep
    col_count: int             # number of columns attempted
    depth: int                 # recursion depth (0 = initial batch)
    status: str                # "success" | "truncated" | "failed" | "fatal"
    retry_of: int | None = None  # parent batch_index when halved (else None)
    error_class: str | None = None
    error_message: str | None = None
    columns: list[str] = field(default_factory=list)
    ts: float = 0.0


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
    # Independent-tier consensus (cosine + pattern + name_match) per
    # column.  Populated alongside ml_prediction in _run_ml_validation.
    # The revisit gate compares these against the LLM vote without the
    # tautological inclusion of LLM-derivative ML sources.
    independent_top1: dict[str, str] = field(default_factory=dict)
    independent_top1_mass: dict[str, float] = field(default_factory=dict)
    independent_top1_conflict: dict[str, float] = field(default_factory=dict)
    llm_calls_total: int = 0
    # Every entry into ``_classify_batch_with_retry`` increments this —
    # success, halve, or failed-at-min.  ``llm_calls_total`` only moves on
    # a successful backend response, so the old budget gate couldn't cap
    # a retry storm.  Attempts is what we actually want to bound.
    llm_attempts_total: int = 0
    # Wall-clock start of the current LLM sweep (seconds since epoch).
    # Zeroed between sweeps; ``_classify_batch_with_retry`` reads this to
    # enforce ``BootstrapConfig.sweep_deadline_s``.
    sweep_started_at: float = 0.0
    # Running count of consecutive recoverable failures.  Reset on any
    # success; LS-4 circuit breaker raises FatalLLMError when it reaches
    # ``cfg.max_consecutive_halve_failures``.
    consecutive_halve_failures: int = 0
    tokens_input: int = 0
    tokens_output: int = 0
    iteration_metrics: list[IterationMetrics] = field(default_factory=list)
    # Halving-retry audit: every LLM batch attempt, success or failure.
    # Nautilus (the mid-run watcher) observes the length + tail of this
    # list to decide whether to intervene.
    batch_audit: list[BatchAttempt] = field(default_factory=list)
    failed_columns: list[str] = field(default_factory=list)
    # Cooperative cancellation.  Nautilus sets ``cancelled=True`` when it
    # decides the run isn't worth continuing (stall or accumulated
    # failures); phase helpers poll the flag between batches and exit
    # cleanly rather than being SIGKILLed mid-call.
    cancelled: bool = False
    cancellation_reason: str = ""
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
    # LLM truncation tracking
    truncation_count: int = 0
    effective_batch_size: int = 50
    # Frontier SVM retraining state
    svm_retrain_count: int = 0
    svm_frontier_path: str | None = None
    # Per-column residual trajectory.  Appended in
    # ``record_iteration_metrics`` after each iteration's ML
    # validation; column-major view of the convergence behaviour.
    # See ``ColumnResidualSnapshot`` and
    # docs/src/architecture/dst-evidence-independence.md.
    column_history: dict[str, list[ColumnResidualSnapshot]] = field(default_factory=dict)


# ── Phase helpers ────────────────────────────────────────────────


def _estimate_safe_batch_size(category_count: int, max_output_tokens: int = 65536) -> int:
    """Estimate columns-per-call that avoids LLM response truncation.

    The LLM must produce a JSON classification object per column (~200
    tokens each including code, confidence, evidence, alternatives).
    Reserve 20% of max_output_tokens for overhead/reasoning.

    For large vocabularies (>200 categories), also reduce batch size
    to keep the system prompt from consuming too much input context.
    """
    # Output budget: each column classification is ~200 output tokens
    output_budget = int(max_output_tokens * 0.8)
    tokens_per_classification = 200
    output_limit = max(5, output_budget // tokens_per_classification)

    # Input budget: large vocab tables consume more input context
    # >200 categories starts eating into available batch capacity
    if category_count > 200:
        input_limit = max(5, 50 - (category_count - 200) // 10)
    else:
        input_limit = 50

    return max(5, min(50, output_limit, input_limit))


def _classify_batch_with_retry(
    backend: LLMBackend,
    chunk_samples: list[ColumnSample],
    system_prompt: str,
    state: BootstrapState,
    *,
    revisit_context: dict[str, dict] | None = None,
    table_name: str | None = None,
    min_batch: int = 1,
    _depth: int = 0,
    _batch_index: int | None = None,
    _parent_index: int | None = None,
    heartbeat: Callable[[str], None] | None = None,
    cfg: BootstrapConfig | None = None,
) -> list:
    """Classify a batch; on truncation OR recoverable failure, halve and recurse.

    Halving terminates at ``min_batch`` (default 1 = per-column fallback),
    at which point a failed sub-batch records a ``failed`` audit entry
    and returns no classifications rather than silently dropping the
    column.  Fatal failures (auth, permission, invariant) raise
    ``FatalLLMError`` immediately so the sweep aborts with a clear
    diagnostic instead of burning retry budget.

    Every call — success, truncation, halved retry, or terminal failure
    — appends a ``BatchAttempt`` to ``state.batch_audit`` so the mid-run
    nautilus watcher (Pillar 2) and the post-mortem overwatch (Pillar 3)
    can reconstruct exactly which columns were attempted and what the
    outcome was.

    Three hang-avoidance mechanisms engage here:

    * ``heartbeat`` fires before each backend call and after each
      ``_record``, so FSM.updated_at advances even during a deep halving
      tree — a multi-minute sub-batch looks distinguishable from a dead
      thread.
    * ``cfg.sweep_deadline_s`` bounds total sweep wall-clock (default
      30 min); a blackholed endpoint raises FatalLLMError instead of
      recursing indefinitely.
    * ``cfg.max_consecutive_halve_failures`` trips when the endpoint is
      effectively dead — converts a ~15min-per-batch retry storm into a
      ~2min fail-fast.
    """
    n = len(chunk_samples)
    batch_index = len(state.batch_audit) if _batch_index is None else _batch_index
    col_names = [s.name for s in chunk_samples]

    def _beat(phase: str) -> None:
        if heartbeat is None:
            return
        try:
            heartbeat(phase)
        except Exception:
            pass  # heartbeat is observability; never let it fail the sweep

    def _record(status: str, error: Exception | None = None) -> None:
        state.batch_audit.append(BatchAttempt(
            batch_index=batch_index,
            col_count=n,
            depth=_depth,
            status=status,
            retry_of=_parent_index,
            error_class=type(error).__name__ if error else None,
            error_message=str(error)[:500] if error else None,
            columns=col_names,
            ts=time.time(),
        ))

    # LS-7 — cancel short-circuit before spending more budget.  The outer
    # loop checks this too; the inner check stops a halving tree from
    # burning through every sub-branch after operator cancel.
    if state.cancelled:
        return []

    # LS-3 — wall-clock deadline.  Bounded so a blackholed endpoint with
    # a 15s connect-timeout can't recurse through ~1 hour of retries.
    # ``sweep_deadline_s <= 0`` disables the wall-clock brake; the
    # attempts cap (LS-2) and consecutive-failure breaker (LS-4) remain
    # as the fast brakes on a truly dead endpoint.  Disabled is the
    # default so a slow-but-healthy Bedrock sweep isn't guillotined.
    if cfg is not None and cfg.sweep_deadline_s > 0 and state.sweep_started_at > 0:
        elapsed = time.time() - state.sweep_started_at
        if elapsed > cfg.sweep_deadline_s:
            _record("deadline_exceeded")
            _beat("sweep_deadline_exceeded")
            raise SweepDeadlineError(
                f"Sweep wall-clock deadline exceeded "
                f"({elapsed:.0f}s > {cfg.sweep_deadline_s:.0f}s) — "
                f"attempts={state.llm_attempts_total}, "
                f"successful_calls={state.llm_calls_total}, "
                f"labels={len(state.labels)}"
            )

    # Attempts ceiling — independent of success count.  A failing endpoint
    # never increments llm_calls_total, so the existing max_total_llm_calls
    # cap has no brake on a retry storm; this is the gate that catches it.
    if cfg is not None and state.llm_attempts_total >= cfg.max_total_llm_attempts:
        _record("attempt_cap_exceeded")
        _beat("sweep_attempt_cap_exceeded")
        raise FatalLLMError(
            f"LLM sweep exceeded max_total_llm_attempts "
            f"({cfg.max_total_llm_attempts}) — endpoint likely unhealthy "
            f"(successful_calls={state.llm_calls_total})"
        )

    # LS-4 — consecutive recoverable failures; endpoint appears dead.
    # Tighter than the attempts cap on a blackholed endpoint: fires in
    # ~2 min (8 × 15s connect-timeout) vs ~40 min for the attempts cap,
    # so it's the primary fast-fail and the attempts cap is the backstop.
    if (
        cfg is not None
        and state.consecutive_halve_failures >= cfg.max_consecutive_halve_failures
    ):
        _record("fatal")
        _beat("sweep_consecutive_failure_breaker")
        raise FatalLLMError(
            f"{state.consecutive_halve_failures} consecutive recoverable "
            f"LLM failures — endpoint appears dead; aborting sweep."
        )

    # LS-2 — count every attempt.  Gates in _llm_sweep / _llm_revisit
    # use this counter so a retry storm actually hits the budget ceiling
    # (llm_calls_total only increments on successful responses).
    state.llm_attempts_total += 1

    # LS-1 — heartbeat before the (possibly hanging) backend call so
    # FSM.updated_at moves during a multi-minute halving tree.  Without
    # this, a sub-batch burning through its connect-timeout budget looks
    # identical to a silently-dead thread.
    _beat(f"sweep_call_start_d{_depth}")

    try:
        response = backend.classify_batch(
            chunk_samples, system_prompt,
            revisit_context=revisit_context,
            table_name=table_name,
        )
    except Exception as exc:
        kind = _classify_error(exc)
        if kind == "fatal":
            _record("fatal", exc)
            _beat("sweep_fatal")
            raise FatalLLMError(
                f"LLM backend reported unrecoverable error ({type(exc).__name__}): {exc}"
            ) from exc

        # Recoverable: halve if possible, else record per-column failure.
        if n <= min_batch:
            _record("failed", exc)
            state.consecutive_halve_failures += 1
            _beat("sweep_failed_at_min")
            for name in col_names:
                if name not in state.failed_columns:
                    state.failed_columns.append(name)
            logger.warning(
                "LLM call failed at minimum batch size (%d col%s): %s",
                n, "s" if n != 1 else "", exc,
            )
            return []

        _record("halved_on_error", exc)
        state.consecutive_halve_failures += 1
        _beat(f"sweep_halve_d{_depth}")
        mid = n // 2
        logger.warning(
            "Recoverable LLM error on batch of %d — halving to 2x%d (%s)",
            n, mid, type(exc).__name__,
        )
        results: list = []
        for sub in (chunk_samples[:mid], chunk_samples[mid:]):
            # LS-7 — stop halving immediately on cancel.
            if state.cancelled:
                break
            sub_context = None
            if revisit_context:
                sub_names = [s.name for s in sub]
                sub_context = {n2: revisit_context[n2] for n2 in sub_names if n2 in revisit_context}
            results.extend(_classify_batch_with_retry(
                backend, sub, system_prompt, state,
                revisit_context=sub_context,
                table_name=table_name,
                min_batch=min_batch,
                _depth=_depth + 1,
                _parent_index=batch_index,
                heartbeat=heartbeat,
                cfg=cfg,
            ))
        return results

    # Call returned successfully — reset the consecutive-failure streak.
    state.llm_calls_total += 1
    state.consecutive_halve_failures = 0
    state.tokens_input += response.input_tokens
    state.tokens_output += response.output_tokens

    if not response.truncated or n <= min_batch:
        if response.truncated:
            state.truncation_count += 1
            _record("truncated_at_min")
            _beat("sweep_truncated_at_min")
            logger.warning(
                "Truncated response at minimum batch size (%d columns)", n,
            )
        else:
            _record("success")
            _beat(f"sweep_success_d{_depth}")
        return list(response.classifications)

    # Truncation detected — halve and retry.
    state.truncation_count += 1
    _record("truncated")
    _beat(f"sweep_halve_truncation_d{_depth}")
    mid = n // 2
    logger.warning(
        "Truncated response for %d columns — retrying as 2x%d", n, mid,
    )
    results = []
    for sub in (chunk_samples[:mid], chunk_samples[mid:]):
        if state.cancelled:
            break
        sub_context = None
        if revisit_context:
            sub_names = [s.name for s in sub]
            sub_context = {n2: revisit_context[n2] for n2 in sub_names if n2 in revisit_context}
        results.extend(_classify_batch_with_retry(
            backend, sub, system_prompt, state,
            revisit_context=sub_context,
            table_name=table_name,
            min_batch=min_batch,
            _depth=_depth + 1,
            _parent_index=batch_index,
            heartbeat=heartbeat,
            cfg=cfg,
        ))
    return results


def _llm_sweep(
    state: BootstrapState,
    cfg: BootstrapConfig,
    backend: LLMBackend,
    system_prompt: str,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
    category_count: int = 0,
    progress_callback=None,
) -> None:
    """Phase 1: Send all columns to LLM in table-aware batches.

    Raises ``RuntimeError`` if **every** batch call fails — this catches
    configuration errors (wrong region, bad credentials, unsupported API
    parameters) early instead of silently proceeding with zero labels and
    reporting false convergence.

    ``progress_callback`` is invoked after every top-level batch (and
    the coverage-gap retry) with a ``dict`` the caller can forward to
    the FSM / UI.  Wiring this is how we stop a silent thread death
    inside the sweep from leaving the FSM frozen at
    ``progress.step = "loading_vocab"`` with no heartbeat — the
    updated_at stamp moves every batch, so a stalled run is
    immediately visible to operators and any watchdog.
    """
    def _heartbeat(phase: str) -> None:
        if progress_callback is None:
            return
        try:
            progress_callback({
                "phase": phase,
                "columns_total": len(column_names),
                "columns_labeled": len(state.labels),
                "llm_calls_total": state.llm_calls_total,
                "truncation_count": state.truncation_count,
                "failed_columns": len(state.failed_columns),
                "batches_attempted": len(state.batch_audit),
            })
        except Exception as exc:
            logger.debug("progress_callback failed (non-fatal): %s", exc)
    # Adaptive batch sizing: reduce batch for large vocabularies
    # AND for backends whose effective output-token ceiling is lower
    # than the configured max_tokens (Bedrock, which silently clamps to
    # the model's native limit).
    effective_fn = getattr(backend, "effective_max_tokens", None)
    backend_ceiling: int = 65536
    if callable(effective_fn):
        try:
            backend_ceiling = int(effective_fn())
        except Exception:
            backend_ceiling = 65536
    safe_batch = (
        _estimate_safe_batch_size(category_count, max_output_tokens=backend_ceiling)
        if category_count else cfg.columns_per_call
    )
    effective_batch = min(cfg.columns_per_call, safe_batch)
    state.effective_batch_size = effective_batch
    if effective_batch < cfg.columns_per_call:
        logger.info(
            "Adaptive batch: %d columns (reduced from %d for %d "
            "categories, backend ceiling=%d)",
            effective_batch, cfg.columns_per_call, category_count,
            backend_ceiling,
        )

    by_table: dict[str, list[str]] = {}
    for name in column_names:
        table = column_table.get(name, "__flat__")
        by_table.setdefault(table, []).append(name)

    batches_attempted = 0
    audit_start = len(state.batch_audit)
    fatal_error: FatalLLMError | None = None

    # Anchor the wall-clock deadline (LS-3).  Only overwrite if unset so
    # downstream phases reuse the same baseline rather than resetting
    # the budget every time they call into _classify_batch_with_retry.
    if state.sweep_started_at == 0.0:
        state.sweep_started_at = time.time()

    # Initial heartbeat — stamps FSM updated_at the moment the sweep
    # starts so "no progress" watchdogs have a baseline.
    _heartbeat("sweep_started")

    for table_name, table_cols in by_table.items():
        for i in range(0, len(table_cols), effective_batch):
            # Two budget gates: success-count (preserves the original
            # max_total_llm_calls semantics — tune to cap cost on a
            # healthy sweep) and attempts (LS-2; catches the retry storm
            # that the calls-count gate can't see because a failing
            # endpoint never increments it).
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                break
            if state.llm_attempts_total >= cfg.max_total_llm_attempts:
                break
            # Cooperative cancellation — nautilus (or an operator via the
            # gateway) sets this flag between batches; exit cleanly here
            # rather than in the middle of a multi-minute LLM call.
            if state.cancelled:
                logger.warning(
                    "LLM sweep aborting — cancellation requested: %s",
                    state.cancellation_reason or "(no reason given)",
                )
                _heartbeat("sweep_cancelled")
                return

            chunk = table_cols[i: i + effective_batch]
            chunk_samples = [samples[n] for n in chunk]
            tname = table_name if table_name != "__flat__" else None

            batches_attempted += 1
            try:
                classifications = _classify_batch_with_retry(
                    backend, chunk_samples, system_prompt, state,
                    table_name=tname,
                    min_batch=cfg.min_columns_per_call,
                    heartbeat=_heartbeat,
                    cfg=cfg,
                )
            except FatalLLMError as exc:
                # Halving can't fix auth/permission/invariant failures —
                # nor the LS-3 wall-clock deadline or LS-4 consecutive-
                # failure breaker, which also surface as FatalLLMError.
                # Abort immediately so the FSM lands on ERROR with a
                # clear diagnostic rather than continuing with partial
                # coverage that looks like success.
                fatal_error = exc
                break

            for c in classifications:
                if c.category_code and c.confidence > 0:
                    state.labels[c.column_name] = c.category_code
                    state.confidence[c.column_name] = c.confidence
                    state.label_source[c.column_name] = "llm"

            # Per-batch heartbeat — advances FSM.updated_at so operators
            # and watchdogs can tell a running sweep from a stalled one
            # even without per-sub-batch granularity in the audit log.
            _heartbeat("sweep_batch_completed")

        if fatal_error is not None:
            break

    if fatal_error is not None:
        raise fatal_error

    # Fail fast if every top-level batch attempt bottomed out without
    # producing any labels — almost certainly a configuration problem
    # rather than a pattern the halving retry can unstick.
    new_audit = state.batch_audit[audit_start:]
    top_level_attempts = [a for a in new_audit if a.depth == 0]
    if top_level_attempts and all(
        a.status in {"failed", "fatal"} for a in top_level_attempts
    ):
        last = top_level_attempts[-1]
        raise RuntimeError(
            f"LLM sweep produced no labels after {len(top_level_attempts)} "
            f"top-level batches (every attempt halved down to failure). "
            f"Last error: {last.error_class}: {last.error_message!r}. "
            f"Check LLM backend config (region, credentials, model ID)."
        )

    # Coverage-gap retry — targeted sweep over any columns that came
    # out with no label.  Two root causes we've seen:
    #   1. Bedrock silently capping maxTokens so a single batch
    #      returned fewer classifications than columns requested
    #      (finish_reason == end_turn despite truncation).
    #   2. A parse recovered N items for a batch of N+k columns;
    #      halving didn't trigger because the response looked clean.
    # The halving retry now engages on response.partial, so this
    # coverage sweep is belt-and-suspenders: it guarantees every column
    # has been *offered* to the LLM before the pipeline declares a
    # labelled corpus.  Retried in per-column batches (min_batch=1) so
    # even the pathological "LLM keeps dropping column X" case at least
    # produces a failed_columns audit entry with the column name.
    missing_after_sweep = [n for n in column_names if n not in state.labels]
    if (
        missing_after_sweep
        and state.llm_calls_total < cfg.max_total_llm_calls
        and state.llm_attempts_total < cfg.max_total_llm_attempts
    ):
        logger.warning(
            "LLM sweep left %d/%d columns without labels — running "
            "targeted coverage retry at min_batch=%d",
            len(missing_after_sweep), len(column_names),
            cfg.min_columns_per_call,
        )
        # Preserve table grouping for retry so prompts still carry
        # sibling context from the actual neighbor set.
        by_table_missing: dict[str, list[str]] = {}
        for name in missing_after_sweep:
            t = column_table.get(name, "__flat__")
            by_table_missing.setdefault(t, []).append(name)
        for table_name, names in by_table_missing.items():
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                break
            if state.llm_attempts_total >= cfg.max_total_llm_attempts:
                break
            if state.cancelled:
                break
            # Use the effective batch so retries respect backend ceiling;
            # if that's still too big, halving will drive toward min.
            retry_batch = max(cfg.min_columns_per_call, effective_batch // 2)
            for i in range(0, len(names), retry_batch):
                if state.llm_calls_total >= cfg.max_total_llm_calls:
                    break
                if state.llm_attempts_total >= cfg.max_total_llm_attempts:
                    break
                chunk = names[i: i + retry_batch]
                chunk_samples = [samples[n] for n in chunk]
                tname = table_name if table_name != "__flat__" else None
                try:
                    classifications = _classify_batch_with_retry(
                        backend, chunk_samples, system_prompt, state,
                        table_name=tname,
                        min_batch=cfg.min_columns_per_call,
                        heartbeat=_heartbeat,
                        cfg=cfg,
                    )
                except FatalLLMError:
                    raise
                for c in classifications:
                    if c.category_code and c.confidence > 0:
                        state.labels[c.column_name] = c.category_code
                        state.confidence[c.column_name] = c.confidence
                        state.label_source[c.column_name] = "llm"

        final_missing = [n for n in column_names if n not in state.labels]
        if final_missing:
            logger.warning(
                "LLM coverage retry still left %d columns unlabeled: %s",
                len(final_missing),
                ", ".join(final_missing[:10]) + ("…" if len(final_missing) > 10 else ""),
            )
            for name in final_missing:
                if name not in state.failed_columns:
                    state.failed_columns.append(name)


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

        indep_code = result.get("independent_top1_code")
        if indep_code:
            state.independent_top1[name] = indep_code
            state.independent_top1_mass[name] = float(
                result.get("independent_top1_mass", 0.0) or 0.0,
            )
            state.independent_top1_conflict[name] = float(
                result.get("independent_top1_conflict", 0.0) or 0.0,
            )


def _identify_disagreements(
    state: BootstrapState,
    column_names: list[str],
    cfg: BootstrapConfig,
) -> list[str]:
    """Find columns that warrant an LLM revisit.

    A revisit fires when *either*:

    1. The independent-tier consensus (cosine + pattern + name_match)
       has a top-1 prediction at meaningful mass (≥
       ``indep_revisit_mass_threshold``) that disagrees with the LLM
       vote.  This branch matters because the LLM-derivative ML
       sources (CatBoost-fit-to-LLM, frontier SVM) cannot, by
       construction, contradict the LLM — see Shafer 1976 §11.3 on
       reliability discounting and Denoeux 2008 on non-distinct
       evidence.  The fully-fused ``ml_prediction`` therefore tracks
       the LLM whenever the LLM is loud, masking genuine
       independent-source disagreement.  Comparing LLM against the
       ``{cosine, pattern, name_match}`` consensus restores a real
       cross-source disagreement test.

    2. K is high *and* the fused prediction differs from LLM.  This
       is the legacy gate, retained as a safety net for cases where
       the independent consensus is diffuse but conflict is
       diagnostically high (typical when the LLM is over-confident
       on a code the ML cluster scattered against).

    Indep-tier disagreement is prioritized in the returned ordering
    because it carries the strongest soundness signal.
    """
    disagreements: list[tuple[str, int, float]] = []
    for name in column_names:
        llm_code = state.labels.get(name)
        ml_code = state.ml_prediction.get(name)
        k = state.ml_conflict.get(name, 0.0)
        indep_code = state.independent_top1.get(name)
        indep_mass = state.independent_top1_mass.get(name, 0.0)

        fire_indep = bool(
            llm_code
            and indep_code
            and indep_code != llm_code
            and indep_mass >= cfg.indep_revisit_mass_threshold
        )
        fire_high_k = bool(
            llm_code
            and ml_code
            and llm_code != ml_code
            and k > cfg.k_threshold
        )

        if fire_indep:
            # Tier 0: indep-tier disagreement at meaningful mass.
            disagreements.append((name, 0, -indep_mass))
        elif fire_high_k:
            # Tier 1: high-K legacy safety net.
            disagreements.append((name, 1, -k))

    disagreements.sort(key=lambda entry: (entry[1], entry[2]))
    return [name for name, _, _ in disagreements]


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

        # Independent-tier consensus and label, when present, surface
        # the LLM-independent counter-evidence the gate fired on so the
        # revisit prompt can reason against it explicitly.
        indep_code = state.independent_top1.get(name, "")
        indep_label = ""
        if indep_code:
            indep_cat = (
                category_set.by_code.get(indep_code)
                or category_set.all_by_code.get(indep_code)
            )
            indep_label = indep_cat.label if indep_cat else indep_code

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
            "independent_consensus": {
                "code": indep_code,
                "label": indep_label,
                "mass": round(state.independent_top1_mass.get(name, 0.0), 4),
            },
        }

    # Group by table for coherent revisit context
    by_table: dict[str, list[str]] = {}
    for name in disagreements:
        table = column_table.get(name, "__flat__")
        by_table.setdefault(table, []).append(name)

    effective_batch = state.effective_batch_size or cfg.columns_per_call

    for table_name, table_cols in by_table.items():
        for i in range(0, len(table_cols), effective_batch):
            if state.llm_calls_total >= cfg.max_total_llm_calls:
                return
            if state.llm_attempts_total >= cfg.max_total_llm_attempts:
                return
            if state.cancelled:
                return

            chunk = table_cols[i: i + effective_batch]
            chunk_samples = [samples[n] for n in chunk]
            chunk_context = {n: revisit_context[n] for n in chunk}
            tname = table_name if table_name != "__flat__" else None

            try:
                classifications = _classify_batch_with_retry(
                    backend, chunk_samples, system_prompt, state,
                    revisit_context=chunk_context,
                    table_name=tname,
                    min_batch=cfg.min_columns_per_call,
                    cfg=cfg,
                )
            except FatalLLMError as exc:
                logger.error(
                    "Fatal LLM error during revisit — aborting revisit phase: %s", exc,
                )
                raise

            for c in classifications:
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
    cfg: BootstrapConfig | None = None,
    revisited_this_iter: set[str] | None = None,
) -> IterationMetrics:
    """Capture metrics for the current iteration.

    Appends one ``ColumnResidualSnapshot`` per labeled column to
    ``state.column_history`` so the column-major trajectory view
    stays in sync with the time-major ``IterationMetrics`` aggregate.
    Columns absent from ``state.labels`` are skipped (symmetric with
    ``_mean_gap`` / ``_frac_needing_clarification``).

    Args:
        revisited_this_iter: Names of columns whose label was
            re-evaluated by ``_llm_revisit`` during the iteration.
            Surfaced on each snapshot so downstream analysis (per-
            column ρ, bandit ordering, Aitken early-stop) can
            distinguish "settled spontaneously" from "settled after
            revisit".  Optional; defaults to no columns flagged.
    """
    # Compute the unified residual + cross-source disagreement
    # fraction.  When cfg is provided (the production path), these
    # become first-class diagnostics tracking convergence per Saad
    # 2003 §4.1.  Falls back gracefully when cfg is omitted (legacy
    # callers / tests).
    r_norm = 0.0
    indep_frac = 0.0
    if cfg is not None:
        r_norm = round(residual_norm(state, column_names, cfg), 4)
        indep_frac = round(
            _indep_tier_disagreement_frac(state, column_names, cfg), 4,
        )

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
        residual_norm=r_norm,
        indep_tier_disagreement_frac=indep_frac,
    )
    state.iteration_metrics.append(metrics)
    # Contraction rate needs the just-appended row, so compute after.
    metrics.contraction_rate = round(contraction_rate(state), 4)

    # Per-column trajectory append.  One snapshot per labeled column
    # per iteration; column-major view that complements the time-
    # major IterationMetrics.  Foundation for per-column ρ, bandit
    # ordering, Aitken early-stop (Phase B) and possible Anderson on
    # oscillating populations (Phase C).
    revisited_set = revisited_this_iter or set()
    for name in column_names:
        if name not in state.labels:
            continue
        snap = ColumnResidualSnapshot(
            iteration=state.iteration,
            gap=round(
                state.ml_plausibility.get(name, 1.0)
                - state.ml_belief.get(name, 0.0), 4,
            ),
            bel=round(state.ml_belief.get(name, 0.0), 4),
            K=round(state.ml_conflict.get(name, 0.0), 4),
            indep_top1_code=state.independent_top1.get(name),
            indep_top1_mass=round(
                state.independent_top1_mass.get(name, 0.0), 4,
            ),
            label=state.labels.get(name),
            label_source=state.label_source.get(name),
            revisited=(name in revisited_set),
        )
        state.column_history.setdefault(name, []).append(snap)

    logger.info(
        "Iteration %d: mean_K=%.4f mean_gap=%.4f mean_bel=%.4f "
        "unclear=%.1f%% disagreements=%d coverage=%.1f%% "
        "‖r‖=%.3f ρ=%.3f indep_disagree=%.1f%%",
        metrics.iteration, metrics.mean_k, metrics.mean_gap, metrics.mean_bel,
        metrics.frac_unclear * 100, metrics.disagreements, metrics.coverage * 100,
        metrics.residual_norm, metrics.contraction_rate,
        metrics.indep_tier_disagreement_frac * 100,
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


def _indep_tier_disagreement_frac(
    state: BootstrapState,
    column_names: list[str],
    cfg: BootstrapConfig,
) -> float:
    """Fraction of labeled columns where indep-tier consensus
    disagrees with the LLM at meaningful mass.

    A column contributes to this residual when:
      - LLM and indep-tier top-1 disagree, AND
      - indep-tier top-1 mass ≥ ``cfg.indep_revisit_mass_threshold``.

    This is the cross-source soundness component of the unified
    residual — it measures the fraction of columns where the
    publicly-grounded sources (cosine, pattern, name_match) are
    at odds with the LLM-derivative cluster.
    """
    labeled = [n for n in column_names if n in state.labels]
    if not labeled:
        return 0.0
    threshold = cfg.indep_revisit_mass_threshold
    bad = 0
    for name in labeled:
        llm_code = state.labels.get(name)
        indep_code = state.independent_top1.get(name)
        indep_mass = state.independent_top1_mass.get(name, 0.0)
        if (
            llm_code and indep_code
            and indep_code != llm_code
            and indep_mass >= threshold
        ):
            bad += 1
    return bad / len(labeled)


def residual_norm(
    state: BootstrapState,
    column_names: list[str],
    cfg: BootstrapConfig,
) -> float:
    """Unified residual norm ``‖r(B)‖`` for the bootstrap iteration.

    The bootstrap loop is iterative refinement on a belief-assignment
    vector B; the residual is a scalar measure of distance from
    convergence (Saad 2003 §4.1).  This implementation combines four
    normalized components, each scaled so 1.0 means "at convergence
    threshold" and 0 means "fully converged":

      r_gap = mean(Pl - Bel) / gap_threshold
      r_unclear = frac_unclear / clarity_target
      r_K = mean(K) / k_threshold
      r_indep = frac(indep-tier disagreement at meaningful mass)

    Combined via L2 norm (root-mean-square) so any single component
    going large dominates while many small components don't add up
    spuriously:  ``r = sqrt((r_gap^2 + r_unclear^2 + r_K^2 + r_indep^2) / 4)``

    A residual_norm of 1.0 means "at convergence threshold across the
    board"; values <1 are converged, >1 are not.  Tracking the
    residual across iterations exposes the contraction factor ρ_n =
    r_n / r_{n-1} — see ``contraction_rate``.
    """
    if not column_names:
        return 0.0

    # Each component normalized to its convergence threshold.
    gap = _mean_gap(state, column_names)
    r_gap = gap / cfg.gap_threshold if cfg.gap_threshold > 0 else gap

    frac_unc = _frac_needing_clarification(
        state, column_names,
        gap_threshold=cfg.gap_threshold, bel_floor=cfg.bel_floor,
    )
    r_unclear = (
        frac_unc / cfg.clarity_target if cfg.clarity_target > 0 else frac_unc
    )

    mean_k = _mean_k(state, column_names)
    r_k = mean_k / cfg.k_threshold if cfg.k_threshold > 0 else mean_k

    r_indep = _indep_tier_disagreement_frac(state, column_names, cfg)

    components = [r_gap, r_unclear, r_k, r_indep]
    sq = sum(c * c for c in components) / len(components)
    return math.sqrt(sq)


def contraction_rate(state: BootstrapState) -> float:
    """Per-iteration contraction factor ρ_n = ‖r_{n+1}‖ / ‖r_n‖.

    Following Saad 2003 §4.1, the contraction factor is the
    headline diagnostic for any iterative method:

      * ρ < 1: converging — successive iterations reduce residual.
      * ρ → 1: stalled — iterations not making progress.
      * ρ > 1: diverging — iterations growing residual.

    For a linear contraction, ρ is constant and equals the
    asymptotic convergence rate.  For nonlinear iterations, ρ is
    typically estimated empirically per-step (this function).

    Returns 0.0 if fewer than two iterations have been recorded
    or the prior residual is zero (avoid division-by-zero).
    """
    metrics = state.iteration_metrics
    if len(metrics) < 2:
        return 0.0
    prev = metrics[-2].residual_norm
    curr = metrics[-1].residual_norm
    if prev <= 1e-12:
        return 0.0
    return curr / prev


def column_contraction(state: BootstrapState, name: str) -> float | None:
    """Per-column contraction factor ρ_col over the column's trajectory.

    Returns ``residual_now / residual_prev`` using the column's gap as
    residual (falls back to K when the prior gap is zero).  ``None``
    when fewer than two snapshots exist for the column.

    Mirrors :func:`contraction_rate` at the column level: ρ_col < 1
    means the column is converging, ρ_col → 1 stalled, ρ_col > 1
    diverging.  Per-column ρ is what bandit-style revisit ordering
    and Aitken Δ² early-stop (Phase B) consume — Phase A delivers
    the helper, but does not yet act on it.
    """
    hist = state.column_history.get(name, [])
    if len(hist) < 2:
        return None
    prev, curr = hist[-2], hist[-1]
    if prev.gap > 1e-9:
        return curr.gap / prev.gap
    if prev.K > 1e-9:
        return curr.K / prev.K
    return None


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
