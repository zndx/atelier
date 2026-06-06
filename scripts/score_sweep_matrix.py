#!/usr/bin/env python
"""Score a 2-D matrix sweep against an LLM-mediated reference artifact.

Companion to ``scripts/score_sweep.py``.  That script consumes the
legacy 1-D ``threshold``-keyed manifest produced by an earlier sweep
runner; this one consumes the ``schema_version: 2, kind: matrix``
manifest produced by the bel × gap (and future N-D) sweeps.

Inputs
------

  --manifest        Path to ``bel_x_gap-<ts>.json`` (matrix-shaped).
  --reference       Path to the reference annotation JSON.  Format
                    identical to ``score_sweep.py``'s ``--agent-mediated``
                    arg: flat ``{"table.col": "ANN", ...}`` or one
                    level of nesting auto-flattened.  Reference values
                    of ``null`` or ``""`` are loaded but skipped at
                    scoring time.
  --baseline-cell   Which axis-tuple to treat as the diff baseline.
                    Format ``"axis1=v1,axis2=v2"`` matching exactly
                    one cell's ``params`` map.  Defaults to the
                    *last* run in the manifest.

Note on terminology
-------------------

The reference is an LLM-mediated artifact (Opus annotations on the
920-column test set), not an objective ground truth in the ML sense.  Field names in
code retain ``agent_mediated`` for compatibility with the existing
``agent_mediated.json`` filename on disk; output labels say
``reference``.

Outputs
-------

Two artifacts written to ``--output-dir`` (default ``build/sweeps``):

  ``scoring_matrix-<utc-iso>.json``
      Per-cell summary keyed by stringified axis-tuple.

  ``per_column_diff_matrix-<utc-iso>.csv``
      Wide table with one row per scored column, one ``pred@<cell>``
      column per matrix cell, plus baseline/correct/recovers-at
      annotations.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SENSITIVE_PREFIXES = ("A_", "C_", "S_", "P_")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _norm(v) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _is_sensitive(annotation: str) -> bool:
    return annotation.startswith(SENSITIVE_PREFIXES)


def _load_reference(path: Path) -> dict[str, str | None]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Reference root must be an object, got {type(data)}")
    flat: dict[str, str | None] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for col, ann in v.items():
                flat[f"{k}.{col}"] = None if ann in (None, "") else _norm(ann)
        else:
            flat[k] = None if v in (None, "") else _norm(v)
    return flat


def _load_classifications(run_dir: Path) -> dict[str, dict]:
    path = run_dir / "classifications.json"
    if not path.exists():
        raise FileNotFoundError(f"No classifications.json at {path}")
    rows = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for row in rows:
        table = _norm(row.get("table_name"))
        col = _norm(row.get("column_name"))
        if not table or not col:
            continue
        out[f"{table}.{col}"] = row
    return out


def _cell_key(params: dict[str, str], axis_paths: list[str]) -> str:
    """Stable display key like ``bel=0.55,gap=0.12`` (short names)."""
    short = []
    for p in axis_paths:
        last = p.rsplit(".", 1)[-1]
        if last == "bel_threshold":
            tag = "bel"
        elif last == "gap_threshold":
            tag = "gap"
        else:
            tag = last
        short.append(f"{tag}={params[p]}")
    return ",".join(short)


def _score_one_run(
    classifications: dict[str, dict],
    reference: dict[str, str | None],
    *,
    also_pre_review: bool,
    binary_sensitivity: bool,
) -> dict:
    scored = 0
    strict_hits = 0
    binary_hits = 0
    pre_review_hits = 0
    missing_in_run = 0

    for key, expected in reference.items():
        if expected is None:
            continue
        row = classifications.get(key)
        if row is None:
            missing_in_run += 1
            continue
        scored += 1
        predicted = _norm(row.get("predicted_annotation"))
        if predicted == expected:
            strict_hits += 1
        if binary_sensitivity and (
            _is_sensitive(predicted) == _is_sensitive(expected)
        ):
            binary_hits += 1
        if also_pre_review:
            pre = _norm(
                row.get("predicted_annotation_pre_review")
                or row.get("predicted_code_pre_review")
            )
            if pre and pre == expected:
                pre_review_hits += 1

    summary = {
        "scored_columns": scored,
        "missing_in_run": missing_in_run,
        "strict_match": strict_hits,
        "strict_pct": round(100.0 * strict_hits / scored, 2) if scored else None,
    }
    if binary_sensitivity:
        summary["binary_sensitive_match"] = binary_hits
        summary["binary_sensitive_pct"] = (
            round(100.0 * binary_hits / scored, 2) if scored else None
        )
    if also_pre_review:
        summary["pre_review_strict_match"] = pre_review_hits
        summary["pre_review_strict_pct"] = (
            round(100.0 * pre_review_hits / scored, 2) if scored else None
        )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("build/results"),
    )
    parser.add_argument(
        "--baseline-cell", type=str, default=None,
        help='Cell to use as diff baseline, e.g. "bel=0.65,gap=0.18". '
        "Defaults to the last run in the manifest.",
    )
    parser.add_argument(
        "--also-score-pre-review", action="store_true",
    )
    parser.add_argument(
        "--binary-sensitivity", action="store_true",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/sweeps"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("kind") != "matrix":
        print(
            f"Expected kind=matrix, got kind={manifest.get('kind')!r}. "
            "Use scripts/score_sweep.py for 1-D manifests.",
            file=sys.stderr,
        )
        return 2

    axis_paths = [a["path"] for a in manifest["axes"]]
    reference = _load_reference(args.reference)
    scoreable = sum(1 for v in reference.values() if v is not None)
    print(
        f"Loaded {scoreable} scoreable columns from {args.reference.name} "
        f"(LLM-mediated reference, not objective agent-mediated reference)",
        file=sys.stderr,
    )

    cells: dict[str, dict] = {}  # cell_key -> {run_id, classifications, params}
    for run in manifest.get("runs", []):
        if run.get("status") != "ok" or not run.get("run_id"):
            continue
        run_id = run["run_id"]
        params = run["params"]
        key = _cell_key(params, axis_paths)
        run_dir = args.results_dir / run_id
        try:
            classifications = _load_classifications(run_dir)
        except FileNotFoundError as exc:
            print(f"  skipping {key}: {exc}", file=sys.stderr)
            continue
        cells[key] = {
            "run_id": run_id,
            "params": params,
            "classifications": classifications,
        }

    if not cells:
        print("No usable runs in manifest.", file=sys.stderr)
        return 1

    cell_keys_ordered = list(cells.keys())  # preserve manifest order
    baseline_key = args.baseline_cell or cell_keys_ordered[-1]
    if baseline_key not in cells:
        print(
            f"--baseline-cell {baseline_key!r} not found in manifest. "
            f"Available: {cell_keys_ordered}",
            file=sys.stderr,
        )
        return 2

    per_cell = {}
    for key in cell_keys_ordered:
        c = cells[key]
        score = _score_one_run(
            c["classifications"],
            reference,
            also_pre_review=args.also_score_pre_review,
            binary_sensitivity=args.binary_sensitivity,
        )
        score["run_id"] = c["run_id"]
        score["params"] = c["params"]
        per_cell[key] = score

    summary = {
        "scored_at": _utc_iso(),
        "manifest": str(args.manifest),
        "reference": str(args.reference),
        "reference_kind": "llm-mediated",
        "baseline_cell": baseline_key,
        "axis_paths": axis_paths,
        "cells": cell_keys_ordered,
        "per_cell": per_cell,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_iso()
    json_path = args.output_dir / f"scoring_matrix-{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    baseline_classifications = cells[baseline_key]["classifications"]
    csv_path = args.output_dir / f"per_column_diff_matrix-{ts}.csv"
    fieldnames = ["table_column", "reference"]
    for key in cell_keys_ordered:
        fieldnames.append(f"pred@{key}")
    fieldnames += [
        "baseline_correct",
        "any_cell_correct",
        "flipped_vs_baseline",
        "recovers_at",
    ]

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, expected in sorted(reference.items()):
            row = {"table_column": key, "reference": expected or ""}
            preds_by_cell: dict[str, str] = {}
            for ck in cell_keys_ordered:
                pred = _norm(
                    cells[ck]["classifications"]
                    .get(key, {})
                    .get("predicted_annotation")
                )
                preds_by_cell[ck] = pred
                row[f"pred@{ck}"] = pred
            baseline_pred = preds_by_cell[baseline_key]
            baseline_correct = (
                expected is not None and baseline_pred == expected
            )
            any_correct = (
                expected is not None
                and any(p == expected for p in preds_by_cell.values())
            )
            flipped = any(p != baseline_pred for p in preds_by_cell.values())
            recovers_at = ""
            if expected is not None and not baseline_correct and any_correct:
                for ck in cell_keys_ordered:
                    if preds_by_cell[ck] == expected:
                        recovers_at = ck
                        break
            row["baseline_correct"] = "1" if baseline_correct else "0"
            row["any_cell_correct"] = "1" if any_correct else "0"
            row["flipped_vs_baseline"] = "1" if flipped else "0"
            row["recovers_at"] = recovers_at
            writer.writerow(row)

    print(f"\nMatrix scoring -> {json_path}")
    print(f"Per-column diff -> {csv_path}\n")
    print(f"{'cell':<22}  {'strict %':>9}  {'scored':>7}  {'missing':>7}  run_id")
    for key in cell_keys_ordered:
        s = per_cell[key]
        print(
            f"  {key:<20}  {str(s.get('strict_pct')):>9}  "
            f"{s['scored_columns']:>7}  {s['missing_in_run']:>7}  {s['run_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
