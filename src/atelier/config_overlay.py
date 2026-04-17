"""In-memory runtime config overlay for the settings page.

Settings applied here survive for the current gateway process.
For persistence across restarts, operators update ``config/base.conf``
or set environment variables (which feed HOCON ``${?VAR}`` substitution).

This is honest UX — the settings page explicitly labels these as
session-level, with a disclaimer that they reset on restart.

Validation is enforced here (ranges, allowed values) so that the
gateway PATCH endpoint and the pipeline integration share a single
source of truth.

Each metadata entry includes:
- ``group``: one of "convergence", "evidence", "sampling", "training",
  "llm_system" — assigns the control to a super-tab in the UI.
- ``default_focus``: when True, this control appears in the "Focus"
  section of the Settings page before any pipeline run has produced
  overwatch-driven focus recommendations. Used as the empty-state
  starter set (see plan phase 2).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

log = logging.getLogger(__name__)


# ── Parameter metadata ─────────────────────────────────────────────
#
# Each entry describes a tunable parameter: its semantic description,
# valid range or allowed values, and a caption template for the UI.
# The settings page fetches this via GET /api/settings.

SETTINGS_METADATA: dict[str, dict[str, Any]] = {
    # ── Convergence ───────────────────────────────────────────────
    "classify_bootstrap_max_iterations": {
        "hocon_path": "classify.bootstrap.max_iterations",
        "label": "Max Iterations",
        "description": "Upper bound on bootstrap revisit iterations",
        "group": "convergence",
        "type": "int",
        "min": 1,
        "max": 15,
        "step": 1,
        "default": 5,
        "caption_template": "Stop the bootstrap loop after {value} revisit iterations even if not converged.",
    },
    "classify_bootstrap_k_threshold": {
        "hocon_path": "classify.bootstrap.k_threshold",
        "label": "Conflict Threshold (K)",
        "description": "Diagnostic K gate — high K signals fundamental source disagreement",
        "group": "convergence",
        "type": "float",
        "min": 0.10,
        "max": 0.50,
        "step": 0.01,
        "default": 0.20,
        "caption_template": "Diagnostic: K > {value} is high-conflict territory — inspect source-level disagreement.",
    },
    "classify_bootstrap_coverage_target": {
        "hocon_path": "classify.bootstrap.coverage_target",
        "label": "Coverage Target",
        "description": "Fraction of columns that must have a predicted_code before the loop can converge",
        "group": "convergence",
        "type": "float",
        "min": 0.80,
        "max": 1.00,
        "step": 0.01,
        "default": 1.00,
        "default_focus": True,
        "caption_template": "Converge only when ≥ {value_pct}% of columns have a prediction — 100% blocks convergence on any uncovered column.",
    },
    "classify_bootstrap_max_total_llm_calls": {
        "hocon_path": "classify.bootstrap.max_total_llm_calls",
        "label": "Max Total LLM Calls",
        "description": "Hard ceiling on LLM batch invocations per run",
        "group": "convergence",
        "type": "int",
        "min": 500,
        "max": 50000,
        "step": 100,
        "default": 5000,
        "caption_template": "Hard stop after {value} LLM batch calls — budget guard for large corpora.",
    },
    "classify_bootstrap_clarity_target": {
        "hocon_path": "classify.bootstrap.clarity_target",
        "label": "Clarity Target",
        "description": "Fraction of columns whose top-1 confidence exceeds the clarity floor",
        "group": "convergence",
        "type": "float",
        "min": 0.05,
        "max": 0.25,
        "step": 0.01,
        "default": 0.10,
        "default_focus": True,
        "caption_template": "Revisit any column whose clarity < {value_pct}% — raises LLM re-sweep aggressiveness.",
    },
    "classify_bootstrap_gap_threshold": {
        "hocon_path": "classify.bootstrap.gap_threshold",
        "label": "Gap Threshold",
        "description": "Belief-gap convergence target: mean(Pl − Bel)",
        "group": "convergence",
        "type": "float",
        "min": 0.08,
        "max": 0.25,
        "step": 0.01,
        "default": 0.15,
        "caption_template": "Converge when mean(Pl − Bel) < {value}.",
    },
    "classify_bootstrap_bel_floor": {
        "hocon_path": "classify.bootstrap.bel_floor",
        "label": "Belief Floor",
        "description": "Minimum belief for a prediction to be 'settled'",
        "group": "convergence",
        "type": "float",
        "min": 0.40,
        "max": 0.70,
        "step": 0.01,
        "default": 0.50,
        "caption_template": "A prediction is 'settled' when Bel ≥ {value}.",
    },
    "classify_bootstrap_frontier_svm_retrain": {
        "hocon_path": "classify.bootstrap.frontier_svm_retrain",
        "label": "Frontier SVM Retrain",
        "description": "Hot-swap the frontier SVM mid-loop as LLM labels accumulate",
        "group": "convergence",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Retrain the frontier SVM on accumulated LLM labels each iteration — adaptive, slower.",
            False: "Freeze the frontier SVM after first fit — faster but stale for high-conflict corpora.",
        },
    },
    # ── Evidence & Fusion ─────────────────────────────────────────
    "classify_fusion_strategy": {
        "hocon_path": "classify.fusion_strategy",
        "label": "Fusion Strategy",
        "description": "DST combination rule for evidence fusion",
        "group": "evidence",
        "type": "choice",
        "choices": ["dempster", "yager"],
        "default": "dempster",
        "captions": {
            "dempster": "Dempster: [m1 ⊕ m2] ÷ (1−K) — normalizes conflict",
            "yager": "Yager: [m1 ⊕ m2], conflict → Θ — preserves ignorance",
        },
    },
    "classify_discount_cosine": {
        "hocon_path": "classify.discounts.cosine",
        "label": "Cosine Discount",
        "description": "Mass allocated to ignorance from embedding similarity",
        "group": "evidence",
        "type": "float",
        "min": 0.20,
        "max": 0.45,
        "step": 0.01,
        "default": 0.30,
        "caption_template": "{value_pct}% of cosine mass allocated to ignorance.",
    },
    "classify_discount_svm": {
        "hocon_path": "classify.discounts.svm",
        "label": "SVM Discount",
        "description": "Mass allocated to ignorance from frontier SVM evidence",
        "group": "evidence",
        "type": "float",
        "min": 0.10,
        "max": 0.40,
        "step": 0.01,
        "default": 0.20,
        "caption_template": "{value_pct}% of SVM mass allocated to ignorance.",
    },
    "classify_discount_pattern_theta": {
        "hocon_path": "classify.discounts.pattern_theta",
        "label": "Pattern Discount",
        "description": "Mass allocated to ignorance from regex pattern matches",
        "group": "evidence",
        "type": "float",
        "min": 0.15,
        "max": 0.40,
        "step": 0.01,
        "default": 0.25,
        "caption_template": "{value_pct}% of pattern-match mass allocated to ignorance.",
    },
    "classify_discount_name_match_exact": {
        "hocon_path": "classify.discounts.name_match_exact",
        "label": "Name-Match Exact Discount",
        "description": "Ignorance mass when column name exactly matches a term",
        "group": "evidence",
        "type": "float",
        "min": 0.50,
        "max": 0.80,
        "step": 0.01,
        "default": 0.70,
        "default_focus": True,
        "caption_template": "Exact name-match keeps {value_pct}% as ignorance — lower = stronger exact-match signal.",
    },
    "classify_discount_name_match_code": {
        "hocon_path": "classify.discounts.name_match_code",
        "label": "Name-Match Code Discount",
        "description": "Ignorance mass when column name matches a term mnemonic / code",
        "group": "evidence",
        "type": "float",
        "min": 0.30,
        "max": 0.65,
        "step": 0.01,
        "default": 0.50,
        "caption_template": "Code-level name-match keeps {value_pct}% as ignorance.",
    },
    "classify_discount_name_match_alias": {
        "hocon_path": "classify.discounts.name_match_alias",
        "label": "Name-Match Alias Discount",
        "description": "Ignorance mass when column name matches a declared alias",
        "group": "evidence",
        "type": "float",
        "min": 0.30,
        "max": 0.65,
        "step": 0.01,
        "default": 0.50,
        "caption_template": "Alias-level name-match keeps {value_pct}% as ignorance.",
    },
    "classify_discount_name_match_overlap": {
        "hocon_path": "classify.discounts.name_match_overlap",
        "label": "Name-Match Overlap Discount",
        "description": "Ignorance mass when column name partially overlaps a term",
        "group": "evidence",
        "type": "float",
        "min": 0.10,
        "max": 0.50,
        "step": 0.01,
        "default": 0.30,
        "caption_template": "Partial-overlap name-match keeps {value_pct}% as ignorance.",
    },
    "classify_discount_catboost_base": {
        "hocon_path": "classify.discounts.catboost_base",
        "label": "CatBoost Base Discount",
        "description": "Baseline CatBoost ignorance mass before variance inflation",
        "group": "evidence",
        "type": "float",
        "min": 0.05,
        "max": 0.25,
        "step": 0.01,
        "default": 0.10,
        "caption_template": "CatBoost baseline discount floor = {value_pct}%.",
    },
    "classify_discount_catboost_variance_scale": {
        "hocon_path": "classify.discounts.catboost_variance_scale",
        "label": "CatBoost Variance Scale",
        "description": "Multiplier applied to CatBoost prediction variance when inflating discount",
        "group": "evidence",
        "type": "float",
        "min": 0.5,
        "max": 3.0,
        "step": 0.1,
        "default": 1.6,
        "caption_template": "Inflate CatBoost discount by variance × {value} — higher = more conservative on uncertain rows.",
    },
    "classify_discount_catboost_max": {
        "hocon_path": "classify.discounts.catboost_max",
        "label": "CatBoost Max Discount",
        "description": "Upper bound on CatBoost ignorance mass",
        "group": "evidence",
        "type": "float",
        "min": 0.30,
        "max": 0.70,
        "step": 0.01,
        "default": 0.50,
        "caption_template": "CatBoost discount capped at {value_pct}% — prevents runaway variance ignorance.",
    },
    "classify_discount_catboost_fallback": {
        "hocon_path": "classify.discounts.catboost_fallback",
        "label": "CatBoost Fallback Discount",
        "description": "Discount used when CatBoost variance signal is unavailable",
        "group": "evidence",
        "type": "float",
        "min": 0.05,
        "max": 0.30,
        "step": 0.01,
        "default": 0.15,
        "caption_template": "Fallback CatBoost discount = {value_pct}% when variance signal is missing.",
    },
    "classify_discount_confusable_ratio_threshold": {
        "hocon_path": "classify.discounts.confusable_ratio_threshold",
        "label": "Confusable Ratio Threshold",
        "description": "Ratio of top-1 to top-2 belief below which predictions are flagged confusable",
        "group": "evidence",
        "type": "float",
        "min": 1.5,
        "max": 5.0,
        "step": 0.1,
        "default": 3.0,
        "caption_template": "Flag as confusable when top-1 / top-2 belief < {value}×.",
    },
    # ── Sampling (Monte Carlo + Row-level) ────────────────────────
    "mc_min_corpus_size": {
        "hocon_path": "classify.monte_carlo.min_corpus_size",
        "label": "MC Min Corpus Size",
        "description": "Below this column count, Monte Carlo stratification is skipped (full sweep)",
        "group": "sampling",
        "type": "int",
        "min": 50,
        "max": 2000,
        "step": 10,
        "default": 200,
        "caption_template": "Skip stratified sampling when column count < {value} — full-sweep regime.",
    },
    "mc_sample_fraction": {
        "hocon_path": "classify.monte_carlo.sample_fraction",
        "label": "Sample Fraction",
        "description": "Fraction of discovered columns sent to LLM sweep per iteration",
        "group": "sampling",
        "type": "float",
        "min": 0.08,
        "max": 0.40,
        "step": 0.01,
        "default": 0.15,
        "default_focus": True,
        "caption_template": "{value_pct}% of columns reach LLM per iteration — raise for denser first-pass coverage.",
    },
    "mc_min_per_stratum": {
        "hocon_path": "classify.monte_carlo.min_per_stratum",
        "label": "Min Per Stratum",
        "description": "Minimum columns sampled from each stratum",
        "group": "sampling",
        "type": "int",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 3,
        "default_focus": True,
        "caption_template": "At least {value} column per stratum reaches LLM — lower bound on sparse-stratum coverage.",
    },
    "mc_max_frontier_columns": {
        "hocon_path": "classify.monte_carlo.max_frontier_columns",
        "label": "Max Frontier Columns",
        "description": "Hard cap on LLM-swept columns regardless of fraction",
        "group": "sampling",
        "type": "int",
        "min": 100,
        "max": 5000,
        "step": 50,
        "default": 500,
        "caption_template": "Cap LLM sweep at {value} columns even when sample_fraction × corpus would be higher.",
    },
    "mc_propagation_threshold": {
        "hocon_path": "classify.monte_carlo.propagation_threshold",
        "label": "Propagation Threshold",
        "description": "Minimum cosine similarity for label propagation to unswept columns",
        "group": "sampling",
        "type": "float",
        "min": 0.75,
        "max": 0.95,
        "step": 0.01,
        "default": 0.85,
        "caption_template": "Propagate labels only when neighbor cosine ≥ {value} — raise to restrict propagation reach.",
    },
    "row_mc_enabled": {
        "hocon_path": "classify.row_mc.enabled",
        "label": "Row Monte Carlo",
        "description": "Rotate row subsets across iterations for unstable columns",
        "group": "sampling",
        "type": "switch",
        "default": False,
        "captions": {
            True: "Row-level Monte Carlo active — values rotate per iteration; surfaces variance on unstable columns.",
            False: "Row values held constant across iterations — cheaper, but masks variance on ambiguous columns.",
        },
    },
    "row_mc_k": {
        "hocon_path": "classify.row_mc.k",
        "label": "Row MC K",
        "description": "Number of rotated row subsets for row-level MC",
        "group": "sampling",
        "type": "int",
        "min": 3,
        "max": 50,
        "step": 1,
        "default": 10,
        "caption_template": "Rotate through {value} row subsets — higher = smoother variance estimate, slower.",
    },
    "row_mc_strategy": {
        "hocon_path": "classify.row_mc.strategy",
        "label": "Row MC Strategy",
        "description": "How row subsets are selected across iterations",
        "group": "sampling",
        "type": "choice",
        "choices": ["head", "random", "stratified"],
        "default": "stratified",
        "captions": {
            "head": "Head: take top-N rows — deterministic, biased toward table ordering.",
            "random": "Random: uniform row sample — unbiased but high variance.",
            "stratified": "Stratified: balanced by value distribution — best signal-to-noise (default).",
        },
    },
    "row_mc_adaptive_escalation": {
        "hocon_path": "classify.row_mc.adaptive_escalation",
        "label": "Row MC Adaptive Escalation",
        "description": "Promote unstable columns to deeper row MC automatically",
        "group": "sampling",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Escalate row_mc_k for high-variance columns automatically.",
            False: "Apply uniform row_mc_k to all columns regardless of per-column variance.",
        },
    },
    # ── Training (CatBoost + SAGE + SHAP) ─────────────────────────
    "classify_catboost_iterations": {
        "hocon_path": "classify.catboost.iterations",
        "label": "CatBoost Iterations",
        "description": "Boosting rounds for CatBoost training",
        "group": "training",
        "type": "int",
        "min": 100,
        "max": 5000,
        "step": 50,
        "default": 1000,
        "caption_template": "Train {value} boosting rounds — more = better fit, longer training, higher overfit risk.",
    },
    "classify_catboost_depth": {
        "hocon_path": "classify.catboost.depth",
        "label": "CatBoost Depth",
        "description": "Tree depth for CatBoost",
        "group": "training",
        "type": "int",
        "min": 3,
        "max": 10,
        "step": 1,
        "default": 6,
        "caption_template": "Tree depth {value} — deeper = finer decision boundaries, higher variance.",
    },
    "classify_catboost_learning_rate": {
        "hocon_path": "classify.catboost.learning_rate",
        "label": "CatBoost Learning Rate",
        "description": "Shrinkage applied to each boosting round",
        "group": "training",
        "type": "float",
        "min": 0.01,
        "max": 0.30,
        "step": 0.01,
        "default": 0.10,
        "caption_template": "Shrink each round's contribution by {value} — lower = smoother fit, pair with higher iterations.",
    },
    "classify_sage_enabled": {
        "hocon_path": "classify.sage.enabled",
        "label": "SAGE Feature Importance",
        "description": "Run SAGE global feature-importance after classification",
        "group": "training",
        "type": "switch",
        "default": False,
        "captions": {
            True: "Run SAGE tour after classification — slow on CPU, auto-enabled on GPU.",
            False: "Skip SAGE — feature importance comes from SHAP only.",
        },
    },
    "classify_sage_permutations": {
        "hocon_path": "classify.sage.permutations",
        "label": "SAGE Permutations",
        "description": "Monte Carlo budget for SAGE — higher = tighter importance estimates",
        "group": "training",
        "type": "int",
        "min": 64,
        "max": 2048,
        "step": 32,
        "default": 512,
        "caption_template": "Run {value} permutations per feature — raise for tighter importance CIs.",
    },
    "classify_shap_enabled": {
        "hocon_path": "classify.shap.enabled",
        "label": "SHAP Explanations",
        "description": "Compute per-prediction SHAP values",
        "group": "training",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Compute SHAP for every prediction — explains 'why this code' per column.",
            False: "Skip SHAP — faster, but loses per-column attribution.",
        },
    },
    "classify_shap_top_k": {
        "hocon_path": "classify.shap.top_k",
        "label": "SHAP Top-K",
        "description": "Number of top contributing features to retain per prediction",
        "group": "training",
        "type": "int",
        "min": 1,
        "max": 5,
        "step": 1,
        "default": 3,
        "caption_template": "Retain top {value} contributing features per prediction.",
    },
    # ── LLM & System ──────────────────────────────────────────────
    "classify_llm_discount": {
        "hocon_path": "classify.llm.discount",
        "label": "LLM Discount",
        "description": "Mass allocated to ignorance from LLM evidence",
        "group": "llm_system",
        "type": "float",
        "min": 0.05,
        "max": 0.20,
        "step": 0.01,
        "default": 0.10,
        "caption_template": "{value_pct}% of LLM mass allocated to ignorance.",
    },
    "classify_llm_max_tokens": {
        "hocon_path": "classify.llm.max_tokens",
        "label": "LLM Max Tokens",
        "description": "Upper bound on tokens per LLM response",
        "group": "llm_system",
        "type": "int",
        "min": 4096,
        "max": 131072,
        "step": 1024,
        "default": 65536,
        "caption_template": "Cap LLM response at {value} tokens per batch.",
    },
    "classify_llm_temperature": {
        "hocon_path": "classify.llm.temperature",
        "label": "LLM Temperature",
        "description": "Sampling temperature for the classification LLM",
        "group": "llm_system",
        "type": "float",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": 0.0,
        "caption_template": "Sample at temperature {value} — 0 = deterministic, 1 = maximally creative.",
    },
    "classify_llm_columns_per_call": {
        "hocon_path": "classify.llm.columns_per_call",
        "label": "Columns per LLM Call",
        "description": "Batch size — columns packed into one LLM request",
        "group": "llm_system",
        "type": "int",
        "min": 10,
        "max": 200,
        "step": 5,
        "default": 50,
        "caption_template": "Pack {value} columns per LLM call — higher = fewer calls, larger context per call.",
    },
    "classify_llm_max_retries": {
        "hocon_path": "classify.llm.max_retries",
        "label": "LLM Max Retries",
        "description": "Retries per LLM batch on transient failures",
        "group": "llm_system",
        "type": "int",
        "min": 1,
        "max": 10,
        "step": 1,
        "default": 3,
        "caption_template": "Retry each LLM batch up to {value} times on transient failure.",
    },
    "classify_embedding_batch_size": {
        "hocon_path": "classify.embedding_batch_size",
        "label": "Embedding Batch Size",
        "description": "Sentence-transformer encoding batch size",
        "group": "llm_system",
        "type": "int",
        "min": 16,
        "max": 1024,
        "step": 16,
        "default": 32,
        "caption_template": "Encode {value} texts per transformer batch — raise for GPU throughput.",
    },
    "classify_gpu_enabled": {
        "hocon_path": "classify.gpu.enabled",
        "label": "GPU Mode",
        "description": "GPU routing policy — auto-detect CUDA or force on/off",
        "group": "llm_system",
        "type": "choice",
        "choices": ["auto", "true", "false"],
        "default": "auto",
        "captions": {
            "auto": "Auto-detect CUDA — GPU kernels when available, CPU fallback otherwise (default).",
            "true": "Force GPU — fails if CUDA is absent.",
            "false": "Force CPU — disables GPU kernels even when CUDA is present.",
        },
    },
    "classify_gpu_shard_threshold": {
        "hocon_path": "classify.gpu.shard_threshold",
        "label": "GPU Shard Threshold",
        "description": "Column count above which embedding is sharded across multiple GPUs",
        "group": "llm_system",
        "type": "int",
        "min": 1500,
        "max": 1000000,
        "step": 500,
        "default": 200000,
        "caption_template": "Engage multi-GPU sharding above {value} columns — lower for larger embedding models.",
    },
    "classify_gpu_sage_chunk": {
        "hocon_path": "classify.gpu.sage_chunk_permutations",
        "label": "GPU SAGE Chunk Permutations",
        "description": "Permutations per GPU kernel launch for SAGE",
        "group": "llm_system",
        "type": "int",
        "min": 4,
        "max": 64,
        "step": 4,
        "default": 16,
        "caption_template": "Bundle {value} permutations per GPU kernel — higher = more VRAM, fewer launch stalls.",
    },
}


# ── Curated starter focus set ─────────────────────────────────────
#
# Keys returned by /api/settings/focus when no pipeline has produced
# overwatch-driven recommendations yet. Defined here so the Python
# focus module + gateway share the same list.

STARTER_FOCUS_KEYS: tuple[str, ...] = tuple(
    k for k, v in SETTINGS_METADATA.items() if v.get("default_focus")
)


# ── Overlay state ──────────────────────────────────────────────────

_overlay: dict[str, Any] = {}


def get_overlay() -> dict[str, Any]:
    """Return a copy of the current overlay (empty = all defaults)."""
    return dict(_overlay)


def clear_overlay() -> None:
    """Remove all overlay entries — reverts to HOCON/env defaults."""
    _overlay.clear()


def set_overlay(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate and apply overlay updates.

    Raises ValueError on invalid keys or out-of-range values.
    Returns the resulting overlay (merged with prior state).
    """
    for key, value in updates.items():
        _validate(key, value)
    _overlay.update(updates)
    log.info("Config overlay updated: %s", sorted(updates.keys()))
    return dict(_overlay)


def _validate(key: str, value: Any) -> None:
    """Raise ValueError if key is unknown or value is out of range."""
    if key not in SETTINGS_METADATA:
        raise ValueError(f"Unknown setting: {key!r}")
    meta = SETTINGS_METADATA[key]
    kind = meta["type"]

    if kind == "choice":
        if value not in meta["choices"]:
            raise ValueError(
                f"{key}: invalid choice {value!r}, expected one of "
                f"{meta['choices']}"
            )
    elif kind == "float":
        # bool is a subclass of int — reject it explicitly for float controls.
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key}: expected number, got {type(value).__name__}")
        fv = float(value)
        if fv < meta["min"] or fv > meta["max"]:
            raise ValueError(
                f"{key}: {fv} out of range [{meta['min']}, {meta['max']}]"
            )
    elif kind == "int":
        # Same bool-as-int hazard here.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{key}: expected int, got {type(value).__name__}")
        if value < meta["min"] or value > meta["max"]:
            raise ValueError(
                f"{key}: {value} out of range [{meta['min']}, {meta['max']}]"
            )
    elif kind == "switch":
        if not isinstance(value, bool):
            raise ValueError(f"{key}: expected bool, got {type(value).__name__}")
    else:
        raise ValueError(f"{key}: unsupported type {kind}")


def apply_to_config(cfg):  # type: ignore[no-untyped-def]
    """Return a copy of cfg with overlay values applied.

    No-op when overlay is empty.  Callers that need the overlay
    (e.g. the classification pipeline) should wrap their load_config()
    result with this function.
    """
    if not _overlay:
        return cfg
    return dataclasses.replace(cfg, **_overlay)
