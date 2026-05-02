#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Reproducibility test — can CatBoost + SVM alone replicate the DST-fused
predictions from the authoritative Dempster run, without ever calling the
LLM?

Runs ``train_eval_cycle.run_real_data_eval(mode="ml_only")`` against the
UAT meta-tagging corpus.  The pipeline there generates synthetic training
data from the real annotations vocabulary, trains CatBoost + SVM on that
synth corpus, then classifies the real UAT columns via DST fusion over
{cosine, name_match, catboost, svm}.  No LLM evidence participates.

Then compares each column's prediction to the Dempster-fused prediction
in build/results/2bb1431b/atelier_embeddings.parquet:

  * Agreement rate:  ML-only's top-1 vs the LLM-involved fusion's top-1
  * ML-only accuracy vs curated reference (exact + hierarchical)
  * Reference-level comparison — when they disagree, who was right?

Usage::

    uv run python scripts/parity/run_ml_only.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    import pandas as pd
    from atelier.classify.train_eval_cycle import run_real_data_eval

    data_dir = Path("build/meta-tagging").resolve()
    if not (data_dir / "annotations.csv").is_file():
        print(f"No annotations.csv under {data_dir}", file=sys.stderr)
        return 1

    # The ML-only pipeline writes its parquet inside work_dir/results/{id}.
    # We pin work_dir to build/results/ml_only/ so the artifacts are
    # alongside the main runs.
    work_dir = Path("build/results/ml_only")
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running ML-only eval against {data_dir} ...", file=sys.stderr)
    result = run_real_data_eval(
        data_dir=data_dir,
        mode="ml_only",
        variants_per_category=30,
        seed=42,
        catboost_iterations=500,
        work_dir=work_dir,
    )

    # Find the parquet that was just produced under work_dir/results.
    produced = sorted(
        (work_dir / "results").glob("*/atelier_embeddings.parquet"),
        key=lambda p: p.stat().st_mtime,
    )
    if not produced:
        print("ML-only run produced no parquet", file=sys.stderr)
        print(f"work_dir: {work_dir}", file=sys.stderr)
        print(f"result keys: {list(result.keys())}", file=sys.stderr)
        return 2
    ml_parquet = produced[-1]
    print(f"ML-only parquet: {ml_parquet}", file=sys.stderr)

    dempster = Path("build/results/2bb1431b/atelier_embeddings.parquet")
    if not dempster.is_file():
        print(f"Reference run 2bb1431b not found at {dempster}", file=sys.stderr)
        return 3

    ml  = pd.read_parquet(ml_parquet)
    dst = pd.read_parquet(dempster)

    # Join on (table, column) — unambiguous key.
    ml_key  = ml.set_index(["table_name", "column_name"])
    dst_key = dst.set_index(["table_name", "column_name"])
    shared = ml_key.index.intersection(dst_key.index)
    if shared.empty:
        print("No overlapping columns between ML-only and Dempster runs", file=sys.stderr)
        return 4

    ml_sub  = ml_key.loc[shared]
    dst_sub = dst_key.loc[shared]

    ml_pred  = ml_sub["predicted_code"].astype(str).str.strip()
    dst_pred = dst_sub["predicted_code"].astype(str).str.strip()
    ref_col = "reference_code" if "reference_code" in dst_sub.columns else None
    if ref_col is None:
        print(
            "Reference parquet missing 'reference_code' column — "
            "regenerate (pre-rename artifact).",
            file=sys.stderr,
        )
        return 5
    ref_codes = dst_sub[ref_col].astype(str).str.strip()
    has_ref   = ref_codes.str.len() > 0

    def exact(pred):
        return float((pred[has_ref] == ref_codes[has_ref]).mean())

    def hier(pred):
        hits = 0
        total = int(has_ref.sum())
        for p, g in zip(pred[has_ref], ref_codes[has_ref]):
            if p == g or (p and g.startswith(p + ".")) or (g and p.startswith(g + ".")):
                hits += 1
        return hits / total if total else 0.0

    agreement = float((ml_pred == dst_pred).mean())

    # Disagreement breakdown
    disagree_mask = (ml_pred != dst_pred) & has_ref
    disagree_n = int(disagree_mask.sum())
    ml_right  = int(((ml_pred == ref_codes) & disagree_mask).sum())
    dst_right = int(((dst_pred == ref_codes) & disagree_mask).sum())
    both_wrong = disagree_n - ml_right - dst_right

    report = {
        "reference_run": "2bb1431b",
        "ml_only_parquet": str(ml_parquet),
        "shared_columns": int(len(shared)),
        "columns_with_reference": int(has_ref.sum()),
        "agreement_rate": round(agreement, 4),
        "ml_only_accuracy": {
            "exact": round(exact(ml_pred), 4),
            "hierarchical": round(hier(ml_pred), 4),
        },
        "dempster_accuracy": {
            "exact": round(exact(dst_pred), 4),
            "hierarchical": round(hier(dst_pred), 4),
        },
        "disagreements": {
            "total": disagree_n,
            "ml_only_correct": ml_right,
            "dempster_correct": dst_right,
            "both_wrong": both_wrong,
        },
        "ml_train_eval_summary": {
            k: result.get(k)
            for k in (
                "accuracy", "hierarchical_accuracy", "micro_f1",
                "macro_f1", "n_correct", "n_evaluated",
            )
            if k in result
        },
    }

    out_path = Path("build/results/2bb1431b/reproducibility_report.json")
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== ML-only reproducibility report ===")
    print(json.dumps(report, indent=2))
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
