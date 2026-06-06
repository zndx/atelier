#!/usr/bin/env python
"""Score a bel_threshold sweep against an agent-mediated reference set.

Consumes a sweep manifest produced by ``sweep_bel_threshold.py`` and a
agent-mediated reference JSON, and produces two artifacts:

  1. ``scoring-<utc-iso>.json`` — per-threshold aggregate scores.
  2. ``per_column_diff-<utc-iso>.csv`` — wide table with one row per
     scored column and one prediction column per threshold (plus the
     agent-mediated reference annotation and a ``flipped_vs_baseline`` flag).

Agent-mediated reference JSON formats accepted
-----------------------------------------------

Flat (preferred for review-driven curation, easy to diff in git)::

  {
    "table_a.col_x": "INOS",
    "table_a.col_y": "TRANSID",
    "table_b.col_z": null,            # explicitly "we don't know"
    ...
  }

Nested (acceptable; auto-flattened)::

  {
    "table_a": {"col_x": "INOS", "col_y": "TRANSID"},
    "table_b": {"col_z": null}
  }

Keys with ``null`` (or empty-string) values are loaded but skipped at
scoring time — they appear in the diff CSV so reviewers can see what
the pipeline produced for unreviewed columns, but they don't count
toward accuracy.

What "strict match" means
-------------------------

``predicted_annotation == agent_mediated_annotation`` (case-sensitive,
whitespace-stripped).  We score the post-cautious-review annotation
because that's what reaches production.  Pre-review predictions live
in ``predicted_code_pre_review`` if your build emits them — the
``--also-score-pre-review`` flag adds a second column comparing
that field, which is useful for confirming the "review is over-
correcting" hypothesis directly.

Sensitive-vs-public binary
--------------------------

Optional second scoring axis.  A column is "sensitive" if its
agent-mediated reference annotation starts with one of ``A_``, ``C_``, ``S_``,
``P_`` (the prefix families that denote regulated / personally
identifiable / commercially sensitive).  Treats any prediction with
the same coarse prefix family as a binary match.  Enable with
``--binary-sensitivity``.

Usage
-----

::

  uv run python scripts/score_sweep.py \\
      --manifest build/sweeps/bel_threshold-2026-05-15T...Z.json \\
      --agent-mediated path/to/agent_mediated.json \\
      --baseline 0.80

The ``--baseline`` flag selects which threshold's predictions are
treated as "the comparison" for the per-column diff (defaults to the
largest threshold in the manifest, which is the current production
default).
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


def _load_agent_mediated(path: Path) -> dict[str, str | None]:
    """Return a flat ``{table.col: annotation_or_None}`` map."""
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Agent-mediated reference root must be an object, got {type(data)}")

    flat: dict[str, str | None] = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for col, ann in v.items():
                flat[f"{k}.{col}"] = None if ann in (None, "") else _norm(ann)
        else:
            flat[k] = None if v in (None, "") else _norm(v)
    return flat


def _load_classifications(run_dir: Path) -> dict[str, dict]:
    """Return ``{table.col: classification_row}`` for a run."""
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


def _score_one_run(
    classifications: dict[str, dict],
    agent_mediated: dict[str, str | None],
    *,
    also_pre_review: bool,
    binary_sensitivity: bool,
) -> dict:
    """Compute strict + (optionally) binary-sensitivity accuracy."""
    scored = 0
    strict_hits = 0
    binary_hits = 0
    pre_review_hits = 0
    missing_in_run = 0

    for key, expected in agent_mediated.items():
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
            pre = _norm(row.get("predicted_annotation_pre_review")
                        or row.get("predicted_code_pre_review"))
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
    parser.add_argument("--agent-mediated", required=True, type=Path)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("build/results"),
        help="Where run dirs live (default: build/results).",
    )
    parser.add_argument(
        "--baseline", type=float, default=None,
        help="Threshold to use as the diff baseline (default: largest in manifest).",
    )
    parser.add_argument(
        "--also-score-pre-review", action="store_true",
        help="Additionally score predicted_*_pre_review when present.",
    )
    parser.add_argument(
        "--binary-sensitivity", action="store_true",
        help="Add a binary sensitive-vs-public scoring column.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/sweeps"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    agent_mediated = _load_agent_mediated(args.agent_mediated)
    print(
        f"Loaded {sum(1 for v in agent_mediated.values() if v is not None)} "
        f"scoreable columns from {args.agent_mediated.name}",
        file=sys.stderr,
    )

    # Map threshold -> classifications dict
    runs_by_threshold: dict[float, tuple[str, dict[str, dict]]] = {}
    for run in manifest.get("runs", []):
        if run.get("status") != "ok" or not run.get("run_id"):
            continue
        threshold = float(run["threshold"])
        run_id = run["run_id"]
        run_dir = args.results_dir / run_id
        try:
            classifications = _load_classifications(run_dir)
        except FileNotFoundError as exc:
            print(f"  skipping threshold={threshold}: {exc}", file=sys.stderr)
            continue
        runs_by_threshold[threshold] = (run_id, classifications)

    if not runs_by_threshold:
        print("No usable runs in manifest.", file=sys.stderr)
        return 1

    thresholds_sorted = sorted(runs_by_threshold)
    baseline_threshold = (
        args.baseline if args.baseline is not None else thresholds_sorted[-1]
    )
    if baseline_threshold not in runs_by_threshold:
        print(
            f"--baseline {baseline_threshold} not found in manifest "
            f"(have {thresholds_sorted})",
            file=sys.stderr,
        )
        return 2

    # ── Per-threshold aggregate scores ────────────────────────────
    per_threshold = {}
    for t in thresholds_sorted:
        run_id, classifications = runs_by_threshold[t]
        score = _score_one_run(
            classifications,
            agent_mediated,
            also_pre_review=args.also_score_pre_review,
            binary_sensitivity=args.binary_sensitivity,
        )
        score["run_id"] = run_id
        per_threshold[f"{t:.2f}"] = score

    summary = {
        "scored_at": _utc_iso(),
        "manifest": str(args.manifest),
        "agent_mediated": str(args.agent_mediated),
        "baseline_threshold": baseline_threshold,
        "thresholds": thresholds_sorted,
        "per_threshold": per_threshold,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ts = _utc_iso()
    json_path = args.output_dir / f"scoring-{ts}.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ── Per-column diff CSV ───────────────────────────────────────
    baseline_classifications = runs_by_threshold[baseline_threshold][1]
    csv_path = args.output_dir / f"per_column_diff-{ts}.csv"
    fieldnames = ["table_column", "agent_mediated"]
    for t in thresholds_sorted:
        fieldnames.append(f"pred@{t:.2f}")
    fieldnames.append("baseline_correct")
    fieldnames.append("any_threshold_correct")
    fieldnames.append("flipped_vs_baseline")
    fieldnames.append("recovers_at")

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for key, expected in sorted(agent_mediated.items()):
            row = {"table_column": key, "agent_mediated": expected or ""}
            preds_by_t: dict[float, str] = {}
            for t in thresholds_sorted:
                pred = _norm(
                    runs_by_threshold[t][1].get(key, {}).get("predicted_annotation")
                )
                preds_by_t[t] = pred
                row[f"pred@{t:.2f}"] = pred
            baseline_pred = preds_by_t[baseline_threshold]
            baseline_correct = (
                expected is not None and baseline_pred == expected
            )
            any_correct = (
                expected is not None
                and any(p == expected for p in preds_by_t.values())
            )
            flipped = any(p != baseline_pred for p in preds_by_t.values())
            recovers_at = ""
            if expected is not None and not baseline_correct and any_correct:
                # Lowest threshold that produces the correct annotation
                for t in thresholds_sorted:
                    if preds_by_t[t] == expected:
                        recovers_at = f"{t:.2f}"
                        break
            row["baseline_correct"] = "1" if baseline_correct else "0"
            row["any_threshold_correct"] = "1" if any_correct else "0"
            row["flipped_vs_baseline"] = "1" if flipped else "0"
            row["recovers_at"] = recovers_at
            writer.writerow(row)

    # ── Console summary ───────────────────────────────────────────
    print(f"\nSweep scoring -> {json_path}")
    print(f"Per-column diff -> {csv_path}\n")
    print(f"{'threshold':>10}  {'strict %':>9}  {'scored':>7}  {'missing':>7}  run_id")
    for t in thresholds_sorted:
        s = per_threshold[f"{t:.2f}"]
        print(
            f"  {t:>8.2f}  {str(s.get('strict_pct')):>9}  "
            f"{s['scored_columns']:>7}  {s['missing_in_run']:>7}  {s['run_id']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
