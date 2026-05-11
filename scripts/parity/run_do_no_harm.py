#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Phase-1 parity driver — do-no-harm check for DST fusion on UAT.

Runs the full classification pipeline once against the UAT meta-tagging
corpus and reports two arms computed from the same run's parquet:

  Arm A: LLM-only baseline   (predicted = llm_code, ignoring fusion)
  Arm B: DST-fused predicted (predicted = predicted_code, the fused result)

Thesis: Arm B ≥ Arm A on exact + hierarchical accuracy.  If so, DST fusion
adds signal (or at worst is neutral) over the Gopala-equivalent
"LLM everywhere" baseline — which is the precondition for later phases
that dial LLM coverage down.

Notes on configuration — the pipeline runs with:

  * Full coverage — no MC sample-fraction cap.
  * Fusion strategy: Yager (not Dempster).  Yager redirects conflict
    mass onto Θ (ignorance) instead of normalizing by ``(1-K)``; that
    matters on this corpus because cosine dispersal inflates K for
    correctly-LLM-labeled columns, and Dempster's normalization then
    over-confidently reshapes the posterior.
  * Default discounts from config/base.conf (no per-source tuning for
    Phase 1; tuning is a Phase-2 concern alongside SHAP/SAGE-driven
    embedding_text engineering).

Usage::

    uv run python scripts/parity/run_do_no_harm.py

Produces ``build/results/{run_id}/``:

  * atelier_embeddings.parquet     (the authoritative artifact for UAT)
  * evaluation_report.json         (written by the pipeline)
  * parity_report.json             (written here: arm-A vs arm-B summary)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _apply_overrides() -> None:
    """Pin the config knobs for the do-no-harm run.

    Everything runs at config defaults (Dempster fusion).  Override
    the fusion strategy by setting ``ATELIER_FUSION_STRATEGY`` in the
    shell before invoking this script — e.g. for a Yager comparison
    run, ``ATELIER_FUSION_STRATEGY=yager uv run python …``.
    """
    pass


def _compute_parity(parquet_path: Path) -> dict:
    """Score arm A (LLM-only) vs arm B (DST-fused) from the parquet."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    if "reference_code" not in df.columns:
        return {
            "error": (
                "parquet missing 'reference_code' column — regenerate the run "
                "(predates the reference_code schema — re-run the pipeline)"
            )
        }
    with_ref = df[df["reference_code"].astype(str).str.len() > 0].copy()
    if with_ref.empty:
        return {"error": "no curated-reference columns in parquet"}

    ref = with_ref["reference_code"].astype(str).str.strip()

    def _exact(pred_col: str) -> float:
        pred = with_ref[pred_col].astype(str).str.strip()
        return float((pred == ref).mean())

    def _hierarchical(pred_col: str) -> float:
        pred = with_ref[pred_col].astype(str).str.strip()
        total = len(with_ref)
        hits = 0
        for p, g in zip(pred, ref):
            if p == g or g.startswith(p + ".") or p.startswith(g + "."):
                hits += 1
        return hits / total if total else 0.0

    arm_a_exact = _exact("llm_code")
    arm_b_exact = _exact("predicted_code")
    arm_a_hier = _hierarchical("llm_code")
    arm_b_hier = _hierarchical("predicted_code")

    mean_belief = float(with_ref["belief"].mean()) if "belief" in with_ref else None
    mean_conflict = float(with_ref["conflict"].mean()) if "conflict" in with_ref else None

    llm_covered = int((with_ref["llm_code"].astype(str).str.len() > 0).sum())
    llm_fraction = llm_covered / len(with_ref)

    # Disagreement cases: where DST flipped the LLM answer.
    flips = with_ref[
        with_ref["llm_code"].astype(str).str.strip()
        != with_ref["predicted_code"].astype(str).str.strip()
    ]
    flip_ref_match = (
        flips["predicted_code"].astype(str).str.strip() == flips["reference_code"].astype(str).str.strip()
    ).sum() if len(flips) else 0
    flip_llm_match = (
        flips["llm_code"].astype(str).str.strip() == flips["reference_code"].astype(str).str.strip()
    ).sum() if len(flips) else 0

    return {
        "total_with_reference": int(len(with_ref)),
        "llm_covered_columns": llm_covered,
        "llm_coverage_fraction": round(llm_fraction, 4),
        "arm_a_llm_only": {
            "exact_accuracy": round(arm_a_exact, 4),
            "hierarchical_accuracy": round(arm_a_hier, 4),
        },
        "arm_b_dst_fused": {
            "exact_accuracy": round(arm_b_exact, 4),
            "hierarchical_accuracy": round(arm_b_hier, 4),
        },
        "delta": {
            "exact": round(arm_b_exact - arm_a_exact, 4),
            "hierarchical": round(arm_b_hier - arm_a_hier, 4),
        },
        "fusion_flips": {
            "count": int(len(flips)),
            "dst_correct_vs_llm": int(flip_ref_match),
            "llm_correct_vs_dst": int(flip_llm_match),
        },
        "mean_belief": round(mean_belief, 4) if mean_belief is not None else None,
        "mean_conflict": round(mean_conflict, 4) if mean_conflict is not None else None,
    }


def main() -> int:
    _configure_logging()
    _apply_overrides()

    # Imports deferred so the env overrides are visible to load_config().
    from atelier.classify import get_fsm
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.config import load_config

    cfg = load_config()
    if not cfg.has_classify_llm:
        print(
            "No classification LLM configured — set ANTHROPIC_API_KEY or "
            "ATELIER_LLM_API_KEY.",
            file=sys.stderr,
        )
        return 2

    print(f"fusion_strategy = {cfg.classify_fusion_strategy}", file=sys.stderr)
    print(f"meta-tagging mount = {cfg.classify_meta_tagging_dir or 'auto-resolved'}", file=sys.stderr)

    fsm = get_fsm()
    result = run_classification_pipeline(
        cfg,
        fsm,
        source_id="meta-tagging",
    )

    run_id = result.get("run_id")
    if not run_id:
        print(f"Pipeline result missing run_id: {result}", file=sys.stderr)
        return 3

    run_dir = Path("build/results") / run_id
    parquet_path = run_dir / "atelier_embeddings.parquet"
    if not parquet_path.is_file():
        print(f"No parquet at {parquet_path}", file=sys.stderr)
        return 4

    parity = _compute_parity(parquet_path)
    parity["run_id"] = run_id
    parity["fusion_strategy"] = cfg.classify_fusion_strategy
    parity["parquet"] = str(parquet_path)

    out_json = run_dir / "parity_report.json"
    out_json.write_text(json.dumps(parity, indent=2))

    print("\n=== do-no-harm parity report ===")
    print(json.dumps(parity, indent=2))
    print(f"\nparquet : {parquet_path}")
    print(f"parity  : {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
