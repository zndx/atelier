#!/usr/bin/env python
"""Curate agent-mediated reference from human-reviewed HTML scoring files.

Extracts a refined reference from Atelier v0.3.0-rc1 HTML external-classifier
evaluation files.  Each HTML row evaluates one annotation type with an Expected
mnemonic, a Generated prediction, an Outcome, and sample column values.  The
script matches HTML rows to specific columns via sample-value overlap with
``embedding_text`` from the run's ``classifications.json`` (inside a ZIP bundle).

The reference is assembled in three tiers:

  Tier 1 — Columns directly matched to an HTML row with Pass outcome.
           Uses the HTML Expected annotation (human-reviewed).
  Tier 2 — Columns whose annotation type has 100% Pass rate in the HTML
           but the specific column was not directly matched.  Retains
           the current agent_mediated.json tag (transitive trust).
  Tier 3 — Columns whose annotation type was not evaluated in the HTML.
           Retains the current agent_mediated.json tag (fallback).

Usage::

    python scripts/curate_from_scoring.py --dry-run
    python scripts/curate_from_scoring.py
    python scripts/curate_from_scoring.py --min-sample-matches 3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path("build/data/agent_mediated")
ARCHIVE = ROOT / "archive"
AGENT_MEDIATED = ROOT / "agent_mediated.json"
WORKING_SET = ROOT / "working_set.json"
AUDIT = ROOT / "audit.json"
REVIEW_STATE = ROOT / "review_state.json"

DEFAULT_HTML = Path(
    "build/resources/hx_v0.3.0-rc1/"
    "data3_with_training_external_classifier_evaluation.html"
)
DEFAULT_ZIP = Path("build/resources/atelier-v0.3.0-rc1_12cf7a10.zip")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# HTML parser
# ---------------------------------------------------------------------------

class _ScoringTableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self._table_idx = 0
        self._in_cell = False
        self._cell_text = ""
        self._current_row: list[str] = []
        self._scoring_tables: list[list[list[str]]] = []
        self._current_table_rows: list[list[str]] = []
        self._is_scoring_table = False

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._table_idx += 1
            self._current_table_rows = []
            self._is_scoring_table = False
        elif tag in ("td", "th"):
            self._in_cell = True
            self._cell_text = ""
        elif tag == "tr":
            self._current_row = []

    def handle_endtag(self, tag):
        if tag in ("td", "th"):
            self._in_cell = False
            self._current_row.append(self._cell_text.strip())
        elif tag == "tr":
            self._current_table_rows.append(self._current_row)
            if (
                len(self._current_row) == 11
                and self._current_row[0] == "Expected"
            ):
                self._is_scoring_table = True
        elif tag == "table":
            if self._is_scoring_table:
                self._scoring_tables.append(self._current_table_rows)

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text += data


def _parse_html_scoring(path: Path) -> list[dict]:
    parser = _ScoringTableParser()
    parser.feed(path.read_text(encoding="utf-8"))

    rows = []
    for table in parser._scoring_tables:
        for row in table:
            if len(row) != 11 or row[0] == "Expected":
                continue
            gen_match = re.match(r"^(\S+)\s*\(([0-9.]+)\)", row[3])
            rows.append({
                "expected": row[0],
                "parent_annotations": [
                    s.strip() for s in row[1].split(";") if s.strip()
                ],
                "other_acceptable": [
                    s.strip() for s in row[2].split(";") if s.strip()
                ],
                "generated_annotation": gen_match.group(1) if gen_match else row[3].strip(),
                "generated_confidence": (
                    float(gen_match.group(2)) if gen_match else None
                ),
                "matched_annotation": row[4].strip(),
                "classification_result": row[6].strip(),
                "outcome": row[7].strip(),
                "reason": row[8].strip(),
                "samples": [
                    s.strip() for s in row[10].split(";") if s.strip()
                ],
            })
    return rows


# ---------------------------------------------------------------------------
# ZIP classifications loader
# ---------------------------------------------------------------------------

def _load_zip_classifications(zip_path: Path) -> dict[str, dict]:
    with zipfile.ZipFile(zip_path) as zf:
        cj_paths = [n for n in zf.namelist() if n.endswith("classifications.json")]
        if not cj_paths:
            print("ERROR: no classifications.json in ZIP", file=sys.stderr)
            sys.exit(1)
        data = json.loads(zf.read(cj_paths[0]))

    index: dict[str, dict] = {}
    for entry in data:
        key = f"{entry['table_name']}.{entry['column_name']}"
        index[key] = entry
    return index


# ---------------------------------------------------------------------------
# Matching algorithm
# ---------------------------------------------------------------------------

def _match_html_to_columns(
    scoring_rows: list[dict],
    col_index: dict[str, dict],
    min_sample_matches: int,
) -> list[dict]:
    matched: list[dict] = []

    for row_idx, row in enumerate(scoring_rows):
        html_samples = row["samples"]
        if len(html_samples) < min_sample_matches:
            matched.append({
                **row,
                "row_index": row_idx,
                "column_key": None,
                "match_confidence": None,
                "sample_overlap": 0,
                "match_status": "insufficient_samples",
            })
            continue

        candidates: list[tuple[str, int]] = []
        for col_key, entry in col_index.items():
            embed = entry.get("embedding_text", "")
            overlap = sum(1 for s in html_samples if s in embed)
            if overlap >= min_sample_matches:
                candidates.append((col_key, overlap))

        if not candidates:
            matched.append({
                **row,
                "row_index": row_idx,
                "column_key": None,
                "match_confidence": None,
                "sample_overlap": 0,
                "match_status": "no_match",
            })
            continue

        if len(candidates) == 1:
            col_key, overlap = candidates[0]
            confidence = "high" if overlap >= 3 else "medium"
            matched.append({
                **row,
                "row_index": row_idx,
                "column_key": col_key,
                "match_confidence": confidence,
                "sample_overlap": overlap,
                "match_status": "unique",
            })
            continue

        gen_ann = row["generated_annotation"]
        ann_filtered = [
            (k, o) for k, o in candidates
            if col_index[k].get("predicted_annotation") == gen_ann
        ]
        if len(ann_filtered) == 1:
            col_key, overlap = ann_filtered[0]
            matched.append({
                **row,
                "row_index": row_idx,
                "column_key": col_key,
                "match_confidence": "medium",
                "sample_overlap": overlap,
                "match_status": "resolved_by_annotation",
            })
            continue

        pool = ann_filtered if ann_filtered else candidates
        pool.sort(key=lambda x: x[1], reverse=True)
        col_key, overlap = pool[0]
        matched.append({
            **row,
            "row_index": row_idx,
            "column_key": col_key,
            "match_confidence": "low",
            "sample_overlap": overlap,
            "match_status": "best_overlap",
            "candidate_count": len(candidates),
        })

    return matched


# ---------------------------------------------------------------------------
# Annotation reliability
# ---------------------------------------------------------------------------

def _build_annotation_reliability(
    scoring_rows: list[dict],
) -> dict[str, dict]:
    by_ann: dict[str, list[dict]] = defaultdict(list)
    for row in scoring_rows:
        by_ann[row["expected"]].append(row)

    reliability: dict[str, dict] = {}
    for ann, rows in sorted(by_ann.items()):
        correct = sum(1 for r in rows if r["outcome"] == "Correct")
        hierarchy = sum(1 for r in rows if r["outcome"] == "Hierarchy Match")
        other_ok = sum(1 for r in rows if r["outcome"] == "Other Acceptable")
        failure = sum(1 for r in rows if r["outcome"] == "Failure")
        ontology = sum(1 for r in rows if r["outcome"] == "Ontology Match")
        total = len(rows)
        pass_count = sum(
            1 for r in rows if r["classification_result"] == "Pass"
        )
        reliability[ann] = {
            "total": total,
            "correct": correct,
            "hierarchy_match": hierarchy,
            "other_acceptable": other_ok,
            "ontology_match": ontology,
            "failure": failure,
            "pass_count": pass_count,
            "pass_rate": pass_count / total if total else 0,
            "all_pass": pass_count == total,
        }
    return reliability


# ---------------------------------------------------------------------------
# Tiered reference assembly
# ---------------------------------------------------------------------------

def _assemble_reference(
    current_am: dict[str, str | None],
    matches: list[dict],
    reliability: dict[str, dict],
    vocab: dict[str, dict],
) -> tuple[dict[str, str | None], dict[str, dict]]:
    tier1_by_col: dict[str, dict] = {}
    for m in matches:
        col = m.get("column_key")
        if col is None:
            continue
        is_pass = m["classification_result"] == "Pass"
        if is_pass:
            if col not in tier1_by_col:
                tier1_by_col[col] = m

    reliable_anns = {
        ann for ann, r in reliability.items() if r["all_pass"]
    }

    new_am: dict[str, str | None] = {}
    audit: dict[str, dict] = {}

    for col_key, current_tag in current_am.items():
        entry: dict = {"previous_tag": current_tag, "changed": False}

        if col_key in tier1_by_col:
            m = tier1_by_col[col_key]
            new_tag = m["expected"]
            if new_tag not in vocab:
                new_tag = current_tag
                entry["tier"] = 3
                entry["tag"] = current_tag
                entry["skipped_html_tag"] = m["expected"]
                entry["skip_reason"] = "html_tag_not_in_vocabulary"
                new_am[col_key] = current_tag
                audit[col_key] = entry
                continue
            entry["tier"] = 1
            entry["tag"] = new_tag
            entry["html_expected"] = m["expected"]
            entry["html_generated"] = m["generated_annotation"]
            entry["html_outcome"] = m["outcome"]
            entry["match_confidence"] = m["match_confidence"]
            entry["match_status"] = m["match_status"]
            entry["sample_overlap"] = m["sample_overlap"]
            if new_tag != current_tag:
                entry["changed"] = True
                entry["disagreement"] = {
                    "previous": current_tag,
                    "new": new_tag,
                }
            new_am[col_key] = new_tag

        elif current_tag is not None and current_tag in reliable_anns:
            entry["tier"] = 2
            entry["tag"] = current_tag
            entry["annotation_pass_rate"] = reliability[current_tag]["pass_rate"]
            new_am[col_key] = current_tag

        else:
            entry["tier"] = 3
            entry["tag"] = current_tag
            if current_tag is not None and current_tag in reliability:
                entry["annotation_pass_rate"] = reliability[current_tag]["pass_rate"]
            new_am[col_key] = current_tag

        audit[col_key] = entry

    return new_am, audit


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------

def _archive_artifacts() -> Path | None:
    if not AGENT_MEDIATED.exists():
        return None
    date_tag = _utc_date()
    dest = ARCHIVE / date_tag
    sentinel = dest / f"{date_tag}_agent_mediated.json"
    if sentinel.exists():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    for src in (AGENT_MEDIATED, AUDIT, REVIEW_STATE):
        if src.exists():
            shutil.copy2(src, dest / f"{date_tag}_{src.name}")
    print(f"Archived pre-update snapshot -> {dest}", file=sys.stderr)
    return dest


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_report(
    matches: list[dict],
    reliability: dict[str, dict],
    audit: dict[str, dict],
    vocab: dict[str, dict],
):
    total_cols = len(audit)
    tier_counts = defaultdict(int)
    changed_count = 0
    for entry in audit.values():
        tier_counts[entry["tier"]] += 1
        if entry.get("changed"):
            changed_count += 1

    matched = sum(1 for m in matches if m.get("column_key"))
    unmatched = sum(1 for m in matches if not m.get("column_key"))

    print(f"\n=== Curation Summary ===")
    print(f"HTML scoring rows:   {len(matches)}")
    print(f"  Matched to column: {matched}")
    print(f"  Unmatched:         {unmatched}")
    print(f"\nAnnotation types evaluated: {len(reliability)}")
    print(
        f"  All-Pass types:    "
        f"{sum(1 for r in reliability.values() if r['all_pass'])}"
    )
    print(
        f"  Mixed/Fail types:  "
        f"{sum(1 for r in reliability.values() if not r['all_pass'])}"
    )

    print(f"\nReference columns:   {total_cols}")
    print(f"  Tier 1 (HTML match):     {tier_counts[1]}")
    print(f"  Tier 2 (transitive):     {tier_counts[2]}")
    print(f"  Tier 3 (fallback):       {tier_counts[3]}")
    print(f"  Tags changed:            {changed_count}")

    if changed_count:
        print(f"\n--- First 20 changes ---")
        shown = 0
        for col, entry in sorted(audit.items()):
            if entry.get("changed") and shown < 20:
                d = entry["disagreement"]
                print(f"  {col}: {d['previous']} -> {d['new']}")
                shown += 1

    invalid = []
    for col, entry in audit.items():
        tag = entry.get("tag")
        if tag is not None and tag not in vocab:
            invalid.append((col, tag))
    if invalid:
        print(f"\nWARNING: {len(invalid)} tags not in vocabulary:")
        for col, tag in invalid[:10]:
            print(f"  {col}: {tag}")
    else:
        print(f"\nAll tags validated against vocabulary.")

    match_status_counts = defaultdict(int)
    for m in matches:
        match_status_counts[m.get("match_status", "unknown")] += 1
    print(f"\nMatch quality:")
    for status, count in sorted(match_status_counts.items()):
        print(f"  {status}: {count}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curate agent-mediated reference from HTML scoring"
    )
    parser.add_argument(
        "--html", type=Path, default=DEFAULT_HTML,
        help="Path to HTML scoring file",
    )
    parser.add_argument(
        "--zip", type=Path, default=DEFAULT_ZIP,
        help="Path to ZIP bundle with classifications.json",
    )
    parser.add_argument(
        "--agent-mediated", type=Path, default=AGENT_MEDIATED,
        help="Path to current agent_mediated.json",
    )
    parser.add_argument(
        "--working-set", type=Path, default=WORKING_SET,
        help="Path to working_set.json (vocabulary)",
    )
    parser.add_argument(
        "--output", type=Path, default=AGENT_MEDIATED,
        help="Output path for new agent_mediated.json",
    )
    parser.add_argument(
        "--audit-output", type=Path,
        default=ROOT / "curation_audit.json",
        help="Output path for curation audit trail",
    )
    parser.add_argument(
        "--min-sample-matches", type=int, default=2,
        help="Minimum sample values that must match (default: 2)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without writing output",
    )
    args = parser.parse_args()

    if not args.html.exists():
        print(f"ERROR: HTML file not found: {args.html}", file=sys.stderr)
        return 1
    if not args.zip.exists():
        print(f"ERROR: ZIP file not found: {args.zip}", file=sys.stderr)
        return 1

    current_am: dict[str, str | None] = json.loads(
        args.agent_mediated.read_text()
    )
    vocab: dict[str, dict] = json.loads(
        args.working_set.read_text()
    )["vocabulary"]

    print(f"Parsing HTML: {args.html}")
    scoring_rows = _parse_html_scoring(args.html)
    print(f"  {len(scoring_rows)} scoring rows extracted")

    print(f"Loading classifications from: {args.zip}")
    col_index = _load_zip_classifications(args.zip)
    print(f"  {len(col_index)} columns indexed")

    print(f"Matching HTML rows to columns (min_overlap={args.min_sample_matches})...")
    matches = _match_html_to_columns(
        scoring_rows, col_index, args.min_sample_matches
    )

    reliability = _build_annotation_reliability(scoring_rows)

    new_am, audit = _assemble_reference(
        current_am, matches, reliability, vocab
    )

    _print_report(matches, reliability, audit, vocab)

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return 0

    _archive_artifacts()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(new_am, indent=2, sort_keys=True))
    print(f"\nWrote reference: {args.output}")

    audit_doc = {
        "metadata": {
            "curated_at": _utc_iso(),
            "html_file": str(args.html),
            "zip_file": str(args.zip),
            "min_sample_matches": args.min_sample_matches,
            "scoring_rows_total": len(scoring_rows),
            "matched_rows": sum(
                1 for m in matches if m.get("column_key")
            ),
            "unmatched_rows": sum(
                1 for m in matches if not m.get("column_key")
            ),
            "annotation_types_evaluated": len(reliability),
            "annotation_types_all_pass": sum(
                1 for r in reliability.values() if r["all_pass"]
            ),
            "tier_counts": {
                str(t): sum(
                    1 for e in audit.values() if e["tier"] == t
                )
                for t in (1, 2, 3)
            },
            "tags_changed": sum(
                1 for e in audit.values() if e.get("changed")
            ),
            "total_columns": len(new_am),
        },
        "annotation_reliability": reliability,
        "columns": audit,
        "disagreements": [
            {
                "column": col,
                "previous_tag": e["disagreement"]["previous"],
                "new_tag": e["disagreement"]["new"],
                "html_outcome": e.get("html_outcome"),
                "match_confidence": e.get("match_confidence"),
            }
            for col, e in sorted(audit.items())
            if e.get("changed")
        ],
        "unmatched_html_rows": [
            {
                "expected": m["expected"],
                "samples_snippet": "; ".join(m["samples"][:3]),
                "match_status": m["match_status"],
            }
            for m in matches
            if not m.get("column_key")
        ],
    }
    args.audit_output.write_text(json.dumps(audit_doc, indent=2))
    print(f"Wrote audit:     {args.audit_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
