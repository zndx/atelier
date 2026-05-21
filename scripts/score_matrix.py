#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Score a matrix-sweep manifest with a harm-aware composite objective.

Reads each run's final-iteration ``scoring_trend.json`` (written by
the per-iteration scorer in ``atelier.classify.incremental_scoring``)
and ranks runs across the matrix grid by a configurable composite:

  composite = strict_pct
              − wrong_subtree_weight   × (wrong_subtree_count / scored × 100)
              − hallucinated_weight    × (hallucinated_count   / scored × 100)
              − child_match_weight     × (child_match_count    / scored × 100)
              − parent_match_weight    × (parent_match_count   / scored × 100)

Defaults map to the "first do no harm" framing the operator agreed
to: wrong_subtree and hallucinated_annotation are the highest-cost
errors (they cause downstream policy mismatch); child_instead_of_parent
is moderately costly (over-commit risk); parent_instead_of_leaf is
conservative under-commit and effectively zero-cost.  Weights are
expressed in "percentage points lost per percentage point of that
error mode" so the composite is on the same 0–100 scale as strict_pct.

Default weights:
  wrong_subtree_weight  = 2.0   # 1pp of wrong_subtree costs 2pp of composite
  hallucinated_weight   = 5.0
  child_match_weight    = 1.0   # over-commit, mid-cost
  parent_match_weight   = 0.0   # conservative under-commit is free

Sibling-within-subtree errors are NOT penalized — they share the
depth-3 prefix with the reference and typically fall within the same
downstream policy bucket (the wrong "leaf" of an EMAIL vs PHONE
within ICE.SENSITIVE.PID.CONTACT, say).

Usage
-----

::

    uv run python scripts/score_matrix.py \\
        --manifest build/sweeps/bel_x_gap-2026-05-16T...Z.json \\
        --output build/sweeps/scoring-bel_x_gap-{ts}.json

The manifest can be a matrix manifest (schema_version=2) or the
single-axis bel_threshold manifest — both expose enough state to
locate per-run trend files.

Output is a JSON document with one entry per run, sorted by composite
descending, plus a top-level summary line for the winning run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_WEIGHTS = {
    "wrong_subtree_weight": 2.0,
    "hallucinated_weight": 5.0,
    "child_match_weight": 1.0,
    "parent_match_weight": 0.0,
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_final_iteration(run_dir: Path) -> dict | None:
    """Return the final-iteration record from ``scoring_trend.json``.

    The trend file is appended per iteration AND on the final
    post-cautious-review tick.  The final tick has phase
    "final_post_cautious_review" and represents the predictions that
    actually landed in classifications.json — that's what we score.
    """
    trend_path = run_dir / "scoring_trend.json"
    if not trend_path.exists():
        return None
    try:
        payload = json.loads(trend_path.read_text())
    except json.JSONDecodeError:
        return None
    iters = payload.get("iterations") or []
    if not iters:
        return None
    # Prefer the final_post_cautious_review tick; fall back to the
    # last in-loop iteration if cautious review didn't run.
    for r in reversed(iters):
        if r.get("phase") == "final_post_cautious_review":
            return r
    return iters[-1]


def _compute_composite(final: dict, weights: dict[str, float]) -> dict:
    """Compute the harm-aware composite score for one run.

    Returns a flat dict of all the per-mode rates (as percentages of
    ``scored``) plus the composite.  Lets the manifest reader see the
    decomposition without recomputing.
    """
    scored = max(1, final.get("scored", 0))  # avoid div-by-zero
    strict_pct = float(final.get("strict_pct", 0.0))

    def _rate(field: str) -> float:
        return 100.0 * float(final.get(field, 0)) / scored

    wrong_subtree_pct = _rate("wrong_subtree")
    hallucinated_pct = _rate("hallucinated_annotation")
    child_match_pct = _rate("child_match")
    parent_match_pct = _rate("parent_match")
    sibling_pct = _rate("sibling_within_subtree")
    missing_pred_pct = _rate("missing_prediction")
    on_right_path_pct = float(final.get("on_right_path_pct", 0.0))

    composite = (
        strict_pct
        - weights["wrong_subtree_weight"] * wrong_subtree_pct
        - weights["hallucinated_weight"] * hallucinated_pct
        - weights["child_match_weight"] * child_match_pct
        - weights["parent_match_weight"] * parent_match_pct
    )

    return {
        "scored": scored,
        "strict_pct": strict_pct,
        "on_right_path_pct": on_right_path_pct,
        "wrong_subtree_pct": round(wrong_subtree_pct, 2),
        "hallucinated_pct": round(hallucinated_pct, 2),
        "child_match_pct": round(child_match_pct, 2),
        "parent_match_pct": round(parent_match_pct, 2),
        "sibling_pct": round(sibling_pct, 2),
        "missing_prediction_pct": round(missing_pred_pct, 2),
        "composite": round(composite, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("build/results"),
        help="Where run dirs live (default: build/results).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Output JSON path.  Defaults to "
            "build/sweeps/scoring-{manifest-stem}-{utc}.json."
        ),
    )
    for name, default in DEFAULT_WEIGHTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=float,
            default=default,
            help=f"Default: {default}",
        )
    args = parser.parse_args()

    weights = {name: getattr(args, name) for name in DEFAULT_WEIGHTS}

    manifest = json.loads(args.manifest.read_text())
    runs = manifest.get("runs", [])
    if not runs:
        print("No runs in manifest.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for run in runs:
        if run.get("status") != "ok" or not run.get("run_id"):
            continue
        run_id = run["run_id"]
        run_dir = args.results_dir / run_id
        final = _load_final_iteration(run_dir)
        if final is None:
            print(
                f"  skipping run {run_id}: no usable scoring_trend.json",
                file=sys.stderr,
            )
            continue
        scores = _compute_composite(final, weights)
        # Carry the sweep-axis params alongside so the ranking table
        # tells the operator WHICH parameter combination won.  Matrix
        # manifests put params in run['params']; legacy single-axis
        # manifests have run['threshold'] only.
        rows.append({
            "run_id": run_id,
            "params": run.get("params") or {"bel_threshold": run.get("threshold")},
            "duration_s": run.get("duration_s"),
            **scores,
            "final_iteration": final.get("iteration"),
        })

    if not rows:
        print("No usable runs (none had a scoring_trend.json).", file=sys.stderr)
        return 2

    rows.sort(key=lambda r: r["composite"], reverse=True)
    best = rows[0]

    output = {
        "scored_at": _utc_iso(),
        "manifest": str(args.manifest),
        "weights": weights,
        "best": {
            "run_id": best["run_id"],
            "params": best["params"],
            "composite": best["composite"],
            "strict_pct": best["strict_pct"],
        },
        "ranking": rows,
    }

    out_path = args.output or (
        Path("build/sweeps") / f"scoring-{args.manifest.stem}-{_utc_iso()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n")

    # Console table
    print(f"\nMatrix scoring -> {out_path}\n")
    header = (
        f"{'rank':>4}  {'composite':>9}  {'strict %':>9}  {'right_path %':>12}  "
        f"{'wrong_st %':>10}  {'hallu %':>8}  {'child %':>8}  {'parent %':>9}  "
        f"params"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        param_str = " ".join(f"{k.split('.')[-1]}={v}" for k, v in r["params"].items())
        print(
            f"{i:>4}  {r['composite']:>9.3f}  {r['strict_pct']:>9.2f}  "
            f"{r['on_right_path_pct']:>12.2f}  "
            f"{r['wrong_subtree_pct']:>10.2f}  {r['hallucinated_pct']:>8.2f}  "
            f"{r['child_match_pct']:>8.2f}  {r['parent_match_pct']:>9.2f}  "
            f"{param_str}"
        )
    print()
    print(
        f"Best: {best['run_id']} composite={best['composite']:.3f} "
        f"strict={best['strict_pct']:.2f}% with {best['params']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
