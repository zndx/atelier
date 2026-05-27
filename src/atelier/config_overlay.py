# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
        "description": "Upper bound on bootstrap revisit iterations. Floor of 2 enforced by project directive — a single-iteration run skips the revisit pass and is not the pipeline we publish accuracy numbers for.",
        "group": "convergence",
        "type": "int",
        "min": 2,
        "max": 15,
        "step": 1,
        "default": 5,
        "caption_template": "Stop the bootstrap loop after {value} revisit iterations even if not converged.",
    },
    "classify_bootstrap_min_iterations": {
        "hocon_path": "classify.bootstrap.min_iterations",
        "label": "Min Iterations",
        "description": "Floor on bootstrap iterations — forces at least this many revisit passes even when the initial disagreement set is empty. Floor of 2 enforced by project directive so the iterative DST-fusion component always runs.",
        "group": "convergence",
        "type": "int",
        "min": 2,
        "max": 10,
        "step": 1,
        "default": 2,
        "caption_template": "Run at least {value} revisit iterations before the loop is allowed to declare convergence — prevents iter-1 early-exit when LLM and ML coincidentally agree.",
    },
    "classify_cautious_review_enabled": {
        "hocon_path": "classify.cautious_review.enabled",
        "label": "Cautious-Code Review (PROVEN HARMFUL)",
        "description": "PROVEN HARMFUL — destroys accuracy. Run ce4f3777 (2026-05-20): reroute 76.1% miss rate, backoff 78.8% miss rate, net −13.6pp vs LLM-only. Default OFF. Do not enable unless you are specifically re-validating this logic with a fix in hand.",
        "group": "convergence",
        "type": "switch",
        "default": False,
        "captions": {
            True: "CAUTION: Cautious review is enabled. This has been empirically proven to destroy accuracy (−13.6pp in ce4f3777). Proceed only if you have a validated fix.",
            False: "Cautious review disabled (recommended). DST fusion predictions are emitted as-is.",
        },
    },
    "classify_cautious_review_bel_threshold": {
        "hocon_path": "classify.cautious_review.bel_threshold",
        "label": "Cautious Review Belief Threshold (PROVEN HARMFUL)",
        "description": "Set to 0.0 (unreachable) as a safety guard. Even if cautious review is enabled, no column has belief < 0.0, so review never fires. Override only for controlled re-validation experiments.",
        "group": "convergence",
        "type": "float",
        "min": 0.0,
        "max": 0.99,
        "step": 0.05,
        "default": 0.0,
        "caption_template": "Review predictions with belief < {value}. Default 0.0 = never fires. The prior default of 0.80 reviewed 68% of columns and destroyed 13.6pp of accuracy.",
    },
    "classify_cautious_review_backend": {
        "hocon_path": "classify.cautious_review.backend",
        "label": "Cautious Review Backend (PROVEN HARMFUL)",
        "description": "Moot when cautious review is disabled (the recommended state). If re-enabling for controlled experiments, selects which backend handles the review LLM calls.",
        "group": "convergence",
        "type": "choice",
        "choices": ["default", "anthropic_direct", "bedrock"],
        "default": "default",
        "captions": {
            "default": "Follow whatever backend the classify pipeline uses.",
            "anthropic_direct": "Force direct Anthropic API.",
            "bedrock": "Route review calls through Bedrock.",
        },
    },
    # ── audit_2026-05-06_a remediations — part of the proven-harmful
    # cautious review system.  These mitigations failed to overcome
    # the fundamental problem.  Retained for re-validation only. ──
    "classify_resolve_llm_annotation_mnemonic": {
        "hocon_path": "classify.resolve_llm_annotation_mnemonic",
        "label": "R1 — Resolve LLM annotation mnemonics",
        "description": "When the LLM emits a mnemonic (``EMPDET``, ``BAN``, ``SHIPADDR``) instead of a numeric code, resolve it to the corresponding category via ``all_by_abbrev``. Strictly recovers evidence; toggle off only for ablation runs measuring the recovery delta.",
        "group": "evidence",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Mnemonic fallback active — LLM evidence recovered for ~13% of corpus on the audit baseline.",
            False: "Mnemonic emissions become vacuous LLM mass; recovery cohort falls back to cosine + ML alone.",
        },
    },
    "classify_exclude_temp_tables": {
        "hocon_path": "classify.exclude_temp_tables",
        "label": "R6 — Skip Hive/Hue temp tables",
        "description": "Drop tables whose name carries the Hive/Hue temp-table prefix (``__tmp_*``, ``tmp_*``) at discovery. In the audit baseline, a single ``hue__tmp_ecommerce_orders`` contributed 14 of the run's annotation-hallucination cases.",
        "group": "sampling",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Temp tables filtered out of classification — Hive query scratch space stays out of the corpus.",
            False: "All discovered tables classify, including ``__tmp_*``. Use only when a customer's data convention legitimately uses that prefix.",
        },
    },
    "classify_cautious_review_shortlist_permissive": {
        "hocon_path": "classify.cautious_review.shortlist_permissive",
        "label": "R2c — Shortlist-permissive reroute (PROVEN HARMFUL)",
        "description": "Part of the proven-harmful cautious review system. This mitigation failed to overcome the fundamental accuracy-destruction problem. Moot when cautious review is disabled.",
        "group": "convergence",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Reroutes to valid taxonomy codes outside the shortlist are accepted.",
            False: "Strict closed-set: only shortlist codes accepted.",
        },
    },
    "classify_cautious_review_exclude_opaque_siblings": {
        "hocon_path": "classify.cautious_review.exclude_opaque_siblings",
        "label": "R3 — Exclude opaque siblings (PROVEN HARMFUL)",
        "description": "Part of the proven-harmful cautious review system. This mitigation failed to overcome the fundamental accuracy-destruction problem. Moot when cautious review is disabled.",
        "group": "convergence",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Opaque-named siblings filtered from reviewer context.",
            False: "All same-table siblings included.",
        },
    },
    "classify_cautious_review_stability_guard_enabled": {
        "hocon_path": "classify.cautious_review.stability_guard_enabled",
        "label": "R2a — Stability guard (PROVEN HARMFUL)",
        "description": "Part of the proven-harmful cautious review system. This mitigation failed to overcome the fundamental accuracy-destruction problem. Moot when cautious review is disabled.",
        "group": "convergence",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Cross-subtree reroutes are downgraded to keep when LLM and pre-review code agreed at high confidence.",
            False: "All reroute decisions accepted regardless of fusion-LLM agreement.",
        },
    },
    "classify_cautious_review_stability_guard_llm_conf": {
        "hocon_path": "classify.cautious_review.stability_guard_llm_conf",
        "label": "R2a — Stability guard confidence (PROVEN HARMFUL)",
        "description": "Part of the proven-harmful cautious review system. Moot when cautious review is disabled.",
        "group": "convergence",
        "type": "float",
        "min": 0.50,
        "max": 0.99,
        "step": 0.05,
        "default": 0.80,
        "caption_template": "Stability guard fires only when the LLM's pre-review code carries confidence ≥ {value}.",
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
        "description": "Hard ceiling on successful LLM batch invocations per run",
        "group": "convergence",
        "type": "int",
        "min": 500,
        "max": 50000,
        "step": 100,
        "default": 5000,
        "caption_template": "Hard stop after {value} successful LLM batch calls — budget guard for large corpora.",
    },
    "classify_bootstrap_max_total_llm_attempts": {
        "hocon_path": "classify.bootstrap.max_total_llm_attempts",
        "label": "Max Total LLM Attempts",
        "description": "Hard ceiling on LLM call attempts (success + failure + halved retry)",
        "group": "convergence",
        "type": "int",
        "min": 1000,
        "max": 100000,
        "step": 500,
        "default": 10000,
        "caption_template": "Circuit-break after {value} attempts — catches retry storms that don't increment the success counter (every failing call halves, never succeeds).",
    },
    "classify_bootstrap_sweep_deadline_s": {
        "hocon_path": "classify.bootstrap.sweep_deadline_s",
        "label": "Sweep Deadline (s)",
        "description": "Wall-clock ceiling on the LLM sweep phase; 0 disables the brake. When positive, raises SweepDeadlineError and transitions FSM to ERROR on breach.",
        "group": "convergence",
        "type": "float",
        "min": 0.0,
        "max": 21600.0,
        "step": 60.0,
        "default": 0.0,
        "caption_template": "Abort the sweep after {value}s of wall-clock (0 = disabled; attempts cap + consecutive-failure breaker still apply). Re-enable with a positive value to hard-cap a slow-but-healthy sweep.",
    },
    "classify_bootstrap_max_consecutive_halve_failures": {
        "hocon_path": "classify.bootstrap.max_consecutive_halve_failures",
        "label": "Consecutive Failure Breaker",
        "description": "Abort after this many consecutive recoverable LLM failures with no successful response",
        "group": "convergence",
        "type": "int",
        "min": 2,
        "max": 32,
        "step": 1,
        "default": 8,
        "caption_template": "Fail the sweep after {value} recoverable failures in a row — fastest brake for a dead endpoint (~2 min on a 15s connect-timeout).",
    },
    "classify_bootstrap_clarity_target": {
        "hocon_path": "classify.bootstrap.clarity_target",
        "label": "Clarity Target",
        "description": "Max fraction of columns flagged unclear (high gap or low belief) before convergence accepts the iteration",
        "group": "convergence",
        "type": "float",
        "min": 0.05,
        "max": 0.30,
        "step": 0.01,
        "default": 0.25,
        "default_focus": True,
        "caption_template": "Tolerate up to {value_pct}% unclear columns at convergence — parent-aware fusion mechanically inflates the gap on specific-category predictions, so a higher tolerance avoids grinding through max_iterations for negligible accuracy gain.",
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
        "default": 0.18,
        "caption_template": "Converge once mean(Pl − Bel) < {value} — lower = demand tighter belief bands before stopping. Parent-aware fusion adds m(parent) to the prediction's plausibility, so values below 0.15 are usually unreachable on real corpora.",
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
        "default": 0.45,
        "caption_template": "Mark a prediction 'settled' once Bel ≥ {value} — under parent-aware fusion mass is shared between the prediction and its parent focal element, so settled-but-correct columns commonly land in [0.45, 0.50).",
    },
    "classify_bootstrap_indep_revisit_mass_threshold": {
        "hocon_path": "classify.bootstrap.indep_revisit_mass_threshold",
        "label": "Indep-Tier Revisit Threshold",
        "description": "Minimum independent-tier consensus mass to fire an LLM revisit on cosine/pattern disagreement",
        "group": "convergence",
        "type": "float",
        "min": 0.20,
        "max": 0.80,
        "step": 0.05,
        "default": 0.45,
        "caption_template": "Trigger an LLM revisit when {{cosine, pattern, name_match}} agree on a code at ≥ {value} mass that disagrees with the LLM — lower = more revisits, higher = more conservative.",
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
        "min": 0.10,
        "max": 0.45,
        "step": 0.01,
        "default": 0.20,
        "caption_template": "Reserve {value_pct}% of cosine mass as ignorance — lower = trust embedding similarity more, raise when cosine is misleading your labels.",
    },
    "classify_discount_svm": {
        "hocon_path": "classify.discounts.svm",
        "label": "SVM Discount",
        "description": "Mass allocated to ignorance from SVM evidence",
        "group": "evidence",
        "type": "float",
        "min": 0.10,
        "max": 0.85,
        "step": 0.01,
        "default": 0.30,
        "caption_template": "Reserve {value_pct}% of SVM mass as ignorance — the synth-trained SVM uses TF-IDF features and ICE-keyed labels translated into the user taxonomy via the LLM-mediated alignment, making it weakly non-distinct evidence at the vocabulary level (Denoeux 2008). The 0.30 default reflects this regime; raise toward 0.55 if reverting to per-column LLM-derivative training.",
    },
    "classify_svm_source": {
        "hocon_path": "classify.svm.source",
        "label": "SVM channel source",
        "description": "Which SVM head the pipeline loads for the SVM evidence channel",
        "group": "evidence",
        "type": "choice",
        "choices": ["registered", "per_vocab_legacy", "auto"],
        "default": "registered",
        "caption_template": "registered (strict) requires a current factorized NHSVM head in nhsvm_head_registry; per_vocab_legacy forces the legacy on-the-fly TF-IDF LinearSVC path; auto tries registered then falls back to legacy with a degradation tag.",
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
        "caption_template": "Reserve {value_pct}% of pattern mass as ignorance — raise when regex hits over-fire on format-only matches.",
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
        "caption_template": "Code-level name-match holds {value_pct}% as ignorance — lower this if your codes are reliable unique IDs.",
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
        "caption_template": "Alias-level name-match holds {value_pct}% as ignorance — raise when aliases are shared across multiple terms.",
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
        "caption_template": "Partial-overlap name-match holds {value_pct}% as ignorance — raise to de-weight fuzzy substring hits.",
    },
    "classify_discount_catboost_base": {
        "hocon_path": "classify.discounts.catboost_base",
        "label": "CatBoost Base Discount",
        "description": "Baseline CatBoost ignorance mass before variance inflation",
        "group": "evidence",
        "type": "float",
        "min": 0.05,
        "max": 0.85,
        "step": 0.01,
        "default": 0.55,
        "caption_template": "Floor CatBoost ignorance at {value_pct}% before variance scaling — when fit-to-LLM is on, CatBoost is non-distinct evidence (Shafer 1976 §11.3), so the principled default is high; lower only when training on labels independent of the LLM sweep.",
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
        "max": 0.95,
        "step": 0.01,
        "default": 0.75,
        "caption_template": "CatBoost discount capped at {value_pct}% — prevents runaway variance ignorance while preserving headroom above the non-distinct-evidence base.",
    },
    "classify_discount_catboost_fallback": {
        "hocon_path": "classify.discounts.catboost_fallback",
        "label": "CatBoost Fallback Discount",
        "description": "Discount used when CatBoost variance signal is unavailable",
        "group": "evidence",
        "type": "float",
        "min": 0.05,
        "max": 0.85,
        "step": 0.01,
        "default": 0.55,
        "caption_template": "When CatBoost variance is unavailable, discount at {value_pct}% — keeps the fallback aligned with the non-distinct-evidence base; the GPU-training path (where posterior_sampling is disabled) routes through this floor.",
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
    # ── Mass-magnitude calibration (Phase 1 operating point) ─────
    "classify_mass_calibration_cosine_alpha": {
        "hocon_path": "classify.mass_calibration.cosine_alpha",
        "label": "Cosine mass α",
        "description": "Post-discount multiplier on cosine channel mass; <1 down-weights, >1 up-weights",
        "group": "evidence",
        "type": "float",
        "min": 0.0,
        "max": 5.0,
        "step": 0.05,
        "default": 1.0,
        "caption_template": "Cosine channel mass × {value} — Phase 1 sweep found α=0.5 optimal (cosine's 53% standalone top-1 didn't justify its 0.80 max mass).",
    },
    "classify_mass_calibration_svm_alpha": {
        "hocon_path": "classify.mass_calibration.svm_alpha",
        "label": "SVM mass α",
        "description": "Post-discount multiplier on SVM channel mass; expands NHSVM's collapsed top-1 mass",
        "group": "evidence",
        "type": "float",
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
        "default": 1.0,
        "caption_template": "SVM channel mass × {value} — Phase 1 found α=15 optimal (exposes NHSVM's 90% same-dataset top-1 rank that the 0.03 mean mass was hiding).",
    },
    "classify_mass_calibration_catboost_alpha": {
        "hocon_path": "classify.mass_calibration.catboost_alpha",
        "label": "CatBoost mass α",
        "description": "Post-discount multiplier on CatBoost mass; tames its near-flat 0.45 contribution",
        "group": "evidence",
        "type": "float",
        "min": 0.0,
        "max": 3.0,
        "step": 0.05,
        "default": 1.0,
        "caption_template": "CatBoost channel mass × {value} — Phase 1 found α=0.7 optimal (CatBoost's flat ~0.45 top-1 mass inflates K without contributing direction).",
    },
    "classify_mass_calibration_llm_alpha": {
        "hocon_path": "classify.mass_calibration.llm_alpha",
        "label": "LLM mass α",
        "description": "Post-discount multiplier on LLM mass; α=0 effectively makes LLM appellate-only",
        "group": "evidence",
        "type": "float",
        "min": 0.0,
        "max": 2.0,
        "step": 0.05,
        "default": 1.0,
        "caption_template": "LLM channel mass × {value} — Phase 1 found α=0.0-0.1 optimal when supervised channels are calibrated; LLM becomes appellate referee, not equal voter.",
    },
    "classify_cosine_union_focal_k": {
        "hocon_path": "classify.cosine.union_focal_k",
        "label": "Cosine union-focal K",
        "description": "Build cosine mass on the top-K union focal element ('answer is in this set'); 0 disables",
        "group": "evidence",
        "type": "int",
        "min": 0,
        "max": 10,
        "step": 1,
        "default": 0,
        "caption_template": "K={value}: 0 keeps per-tag singletons; >0 builds a single top-K union focal. Phase 1 found K=3 structurally optimal; K>3 dilutes the focal even though raw recall keeps climbing.",
    },
    "classify_cosine_union_focal_alpha": {
        "hocon_path": "classify.cosine.union_focal_alpha",
        "label": "Cosine union-focal α",
        "description": "Mass assigned to the top-K union focal (rest goes to Θ); only used when K>0",
        "group": "evidence",
        "type": "float",
        "min": 0.10,
        "max": 0.90,
        "step": 0.05,
        "default": 0.45,
        "caption_template": "Top-K union focal mass = {value} — Phase 1 found 0.45 optimal at K=3; raise as K grows to compensate for diluted per-code commitment.",
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
        "description": "Fraction of discovered columns sent to LLM sweep per iteration (1.0 bypasses MC entirely and sends every column to the LLM)",
        "group": "sampling",
        "type": "float",
        "min": 0.08,
        "max": 1.00,
        "step": 0.01,
        "default": 1.00,
        "default_focus": True,
        "caption_template": "{value_pct}% of columns reach LLM per iteration — 100% disables Monte-Carlo stratification and sends every column to the LLM.",
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
        "caption_template": "Guarantee {value} LLM samples per stratum — raise when rare categories are being missed; the dominant stratum still gets sample_fraction × corpus.",
    },
    "mc_max_sampled_columns": {
        "hocon_path": "classify.monte_carlo.max_sampled_columns",
        "label": "Max Sampled Columns",
        "description": "Hard cap on directly-LLM-classified columns regardless of fraction",
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
        "default": 0.80,
        "caption_template": "Propagate labels only when neighbor cosine ≥ {value} — raise to restrict propagation reach. Lowered to 0.80 alongside the convergence-loop bel_floor under parent-aware fusion.",
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
    "classify_exclude_reference_columns": {
        "hocon_path": "classify.exclude_reference_columns",
        "label": "Exclude synth reference columns",
        "description": (
            "Drop synth-generator answer-key columns (names matching "
            "attr_*, code_*, col_*, data_*, field_*, item_*, key_*, "
            "ref_*, val_*, var_* with numeric suffixes) from the "
            "sample set before the LLM sweep.  Strictly a pre-filter; "
            "the prediction path never regex-decodes column names. "
            "No effect on production data (regex matches none of it)."
        ),
        "group": "sampling",
        "type": "switch",
        "default": True,
        "captions": {
            True: "Reference columns are excluded — classifier only sees natural-named columns on the UAT synth corpus (production default).",
            False: "Answer-key columns flow through the full classifier (LLM + cosine + CatBoost + SVM + DST fusion) as ordinary inputs and may influence classification results for adjacent columns.",
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
        "default": 8,
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
    # classify_catboost_fit_to_llm intentionally not user-tunable —
    # it is a project design directive (must be True).  Enforced at
    # pipeline entry in run_classification_pipeline.  See project memory
    # feedback_pipeline_invariants.md.  The companion knob
    # classify_catboost_fit_to_llm_min_labels tunes *when* fit-to-LLM
    # kicks in and is kept tunable below.
    "classify_catboost_fit_to_llm_min_labels": {
        "hocon_path": "classify.catboost.fit_to_llm_min_labels",
        "label": "Fit-to-LLM Min Labels",
        "description": "Minimum LLM-labeled columns required before fit-to-LLM kicks in",
        "group": "training",
        "type": "int",
        "min": 10,
        "max": 500,
        "step": 5,
        "default": 30,
        "caption_template": "Need ≥ {value} LLM labels before fitting — below this, fall back to the pre-trained model for this run.",
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
    "classify_shap_method": {
        "hocon_path": "classify.shap.method",
        "label": "SHAP Method",
        "description": "Which SHAP variant to run.  ``auto`` prefers GPU PermutationSHAP over the 12 named features (interpretable + fast on GPU; skips on CPU).  ``permutation`` forces PermutationSHAP regardless of hardware.  ``treeshap`` opts into CatBoost TreeSHAP — fast but attributes to the aggregate 'embedding' feature group, not the 12 source features.",
        "group": "training",
        "type": "choice",
        "choices": ["auto", "permutation", "treeshap"],
        "default": "auto",
        "captions": {
            "auto": "Auto: GPU PermutationSHAP over 12 named features when available, skips on CPU.",
            "permutation": "Permutation: force PermutationSHAP regardless of hardware.",
            "treeshap": "TreeSHAP: fast CatBoost-native — attributes to aggregate embedding group, not the 12 source features.",
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
        "caption_template": "Retain the top {value} feature attributions per column — raise for richer 'why?' explanations, lower to trim parquet payload.",
    },
    # ── LLM & System ──────────────────────────────────────────────
    "classify_llm_discount": {
        "hocon_path": "classify.llm.discount",
        "label": "LLM Discount",
        "description": "Mass allocated to ignorance from LLM evidence",
        "group": "llm_system",
        "type": "float",
        "min": 0.05,
        "max": 0.30,
        "step": 0.01,
        "default": 0.15,
        "caption_template": "Reserve {value_pct}% of LLM mass as ignorance — lower = trust the LLM's self-confidence, raise when you're seeing confident-but-wrong predictions or when LLM-derivative sources (CatBoost fit-to-LLM, SVM via the ontology→user-vocab alignment) are amplifying its vote.",
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
        "caption_template": "Bound LLM response at {value} tokens — raise only if you see truncated batches; larger values burn context quota without accuracy gain.",
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
        "default": 25,
        "caption_template": "Pack {value} columns per LLM call — higher = fewer calls but risks response truncation on vocabularies > 200 terms; 25 is safe for typical UAT corpora.",
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
        "caption_template": "Retry each batch up to {value} times — raise when the provider is flaky, lower to fail fast on persistent errors.",
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


def snapshot(cfg) -> dict[str, Any]:  # type: ignore[no-untyped-def]
    """Capture a point-in-time record of all tunable settings.

    Returns a dict suitable for writing as ``settings_snapshot.json``
    beside a pipeline run's other artifacts. Includes:

    - ``overlay_at_start``: the subset of keys the operator explicitly
      set via the Settings page (non-default).
    - ``resolved_values``: the final value each SETTINGS_METADATA key
      resolved to after HOCON + env + overlay merge.
    - ``default_values``: the SETTINGS_METADATA baseline, for diffing
      in the UI's "historical → current" view.

    Callers should pass the *already-overlaid* cfg (i.e. the return
    value of ``apply_to_config``) so ``resolved_values`` reflects
    what the run actually used.
    """
    overlay_at_start = dict(_overlay)
    resolved: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    for key, meta in SETTINGS_METADATA.items():
        resolved[key] = getattr(cfg, key, meta.get("default"))
        defaults[key] = meta.get("default")
    return {
        "overlay_at_start": overlay_at_start,
        "resolved_values": resolved,
        "default_values": defaults,
    }
