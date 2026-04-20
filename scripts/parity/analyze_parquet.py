#!/usr/bin/env python
"""Deeper parity analysis on an already-produced run parquet.

Loads build/results/{run_id}/atelier_embeddings.parquet and produces:

  * Arm A (LLM-only) vs Arm B (DST-fused) exact + hierarchical accuracy.
  * Per-evidence-source marginal ablations — which single sources carry
    their weight, and where DST fusion beats the best single source.
  * Belief-gap calibration (do higher-Pl-minus-Bel columns error more?).
  * Conflict distribution — does K actually predict correctness?

Usage::

    uv run python scripts/parity/analyze_parquet.py <run_id>
    uv run python scripts/parity/analyze_parquet.py  (defaults to latest)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _latest_run(results_dir: Path) -> Path | None:
    runs = [p for p in results_dir.iterdir()
            if p.is_dir() and (p / "atelier_embeddings.parquet").is_file()]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def main() -> int:
    import pandas as pd

    results = Path("build/results")
    if len(sys.argv) >= 2:
        run_dir = results / sys.argv[1]
    else:
        run_dir = _latest_run(results)
        if run_dir is None:
            print("No runs found under build/results/", file=sys.stderr)
            return 1

    parquet = run_dir / "atelier_embeddings.parquet"
    print(f"analyzing: {parquet}", file=sys.stderr)
    df = pd.read_parquet(parquet)

    if "reference_code" not in df.columns:
        print(
            "Parquet missing 'reference_code' column — regenerate the run "
            "(parquet predates the reference_code schema — re-run the pipeline).",
            file=sys.stderr,
        )
        return 2
    with_ref = df[df["reference_code"].astype(str).str.len() > 0].copy()
    if with_ref.empty:
        print("No curated-reference columns.", file=sys.stderr)
        return 2

    ref = with_ref["reference_code"].astype(str).str.strip()

    def accuracy(col: str) -> tuple[float, float]:
        pred = with_ref[col].astype(str).str.strip()
        exact = float((pred == ref).mean())
        hier_hits = 0
        for p, g in zip(pred, ref):
            if p == g or (p and g.startswith(p + ".")) or (g and p.startswith(g + ".")):
                hier_hits += 1
        return exact, hier_hits / len(with_ref)

    print()
    print(f"=== {run_dir.name} · {len(with_ref)} referenced columns · {len(df)} total ===")
    print()

    def fmt(col_label: str, exact: float, hier: float) -> str:
        return f"  {col_label:<28s}  exact={exact:7.2%}  hier={hier:7.2%}"

    a_ex, a_hi = accuracy("llm_code")
    b_ex, b_hi = accuracy("predicted_code")
    print("--- Primary arms ---")
    print(fmt("Arm A (LLM-only)", a_ex, a_hi))
    print(fmt("Arm B (DST-fused)", b_ex, b_hi))
    print(fmt("Δ (B − A)", b_ex - a_ex, b_hi - a_hi))

    # Calibration buckets on (Pl − Bel)
    if "plausibility" in with_ref and "belief" in with_ref:
        print()
        print("--- Belief-gap calibration (Arm B) ---")
        with_ref["_gap"] = with_ref["plausibility"] - with_ref["belief"]
        with_ref["_correct"] = (
            with_ref["predicted_code"].astype(str).str.strip() == ref
        )
        bins = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
        for lo, hi in bins:
            sub = with_ref[(with_ref["_gap"] >= lo) & (with_ref["_gap"] < hi)]
            if len(sub) == 0:
                continue
            acc = float(sub["_correct"].mean())
            print(f"  gap ∈ [{lo:.2f}, {hi:.2f})  n={len(sub):4d}  acc={acc:7.2%}")

    # Conflict calibration
    if "conflict" in with_ref:
        print()
        print("--- Conflict K calibration (Arm B) ---")
        with_ref["_correct_b"] = (
            with_ref["predicted_code"].astype(str).str.strip() == ref
        )
        bins = [(0.0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
        for lo, hi in bins:
            sub = with_ref[(with_ref["conflict"] >= lo) & (with_ref["conflict"] < hi)]
            if len(sub) == 0:
                continue
            acc = float(sub["_correct_b"].mean())
            print(f"  K ∈ [{lo:.2f}, {hi:.2f})  n={len(sub):4d}  acc={acc:7.2%}")

    # Flip analysis
    print()
    print("--- Fusion flips (LLM ≠ DST) ---")
    flips = with_ref[
        with_ref["llm_code"].astype(str).str.strip()
        != with_ref["predicted_code"].astype(str).str.strip()
    ]
    if len(flips) == 0:
        print("  none (LLM == DST on every labeled column)")
    else:
        dst_right = (
            flips["predicted_code"].astype(str).str.strip() == flips["reference_code"].astype(str).str.strip()
        ).sum()
        llm_right = (
            flips["llm_code"].astype(str).str.strip() == flips["reference_code"].astype(str).str.strip()
        ).sum()
        both_wrong = len(flips) - dst_right - llm_right
        print(f"  total flips    n={len(flips)}")
        print(f"    DST correct  {dst_right}")
        print(f"    LLM correct  {llm_right}")
        print(f"    both wrong   {both_wrong}")

    # Read evaluation_report.json if present
    evr = run_dir / "evaluation_report.json"
    if evr.is_file():
        ev = json.loads(evr.read_text())
        print()
        print("--- evaluation_report.json (pipeline) ---")
        for k in (
            "total_columns", "columns_with_reference", "exact_accuracy",
            "hierarchical_accuracy", "micro_f1", "macro_f1",
            "mean_belief", "mean_plausibility", "mean_uncertainty_gap",
            "mean_conflict", "evidence_sources_used",
        ):
            if k in ev:
                print(f"  {k:30s}  {ev[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
