#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Apples-to-apples reconciliation: Atelier vs. baseline UAT xlsx.

The baseline xlsx only contains rows for columns that were in the
baseline pipeline's actual input. Columns absent from the baseline
output are NOT failures — they were never seen. This script separates:

  * **baseline_seen**   — column appears in baseline's section of the xlsx
                          (``Column Name`` cell non-null), regardless of
                          whether it produced a tag
  * **baseline_tagged** — baseline_seen AND ``Suggested Classification Tags``
                          non-null and resolvable to a code
  * **baseline_declined** — baseline_seen but not tagged (the true
                          "refusal" bucket)
  * **baseline_absent** — no row in baseline's section of the sheet for
                          this (table, column) — never in baseline input

Then it reports:

  1. Overall Atelier performance on all curated-reference columns.
  2. Baseline performance on the intersection (baseline_seen).
  3. Atelier performance on the same intersection (apples-to-apples).
  4. Atelier performance on baseline_declined (cols baseline saw but
     declined to tag) — where Atelier adds genuine coverage.
  5. Per-table breakdown of seen / absent / declined so the phase-gate
     review can show exactly which tables baseline did and didn't see.

Outputs:

  build/results/parity/reconciliation.json
  build/results/parity/reconciliation.md
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


log = logging.getLogger("reconcile")


def _load_curated_reference(path: Path) -> dict[tuple[str, str], dict]:
    gt: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["derivation"] == "unresolved":
                continue
            gt[(row["table"], row["column"])] = row
    return gt


def _mnemonic_to_code() -> dict[str, str]:
    from atelier.classify.meta_tagging_source import _read_annotation_rows
    records = _read_annotation_rows(Path("build/meta-tagging/annotations.csv"))
    m: dict[str, str] = {}
    for r in records:
        ann = (r.get("annotation") or "").strip()
        code = r.get("id")
        if ann and code and ann not in m:
            m[ann] = code
    return m


def _is_reference_col(name: str) -> bool:
    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE
    return bool(_REFERENCE_COL_RE.match(name))


def _normalize_table(xlsx_sheet_name: str) -> str:
    return {
        "customer_pci_data": "customer_pii_pci_data",
    }.get(xlsx_sheet_name.lower(), xlsx_sheet_name.lower())


def _hier_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    return gt.startswith(pred + ".") or pred.startswith(gt + ".")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    import pandas as pd

    xlsx_path = Path("build/Atelier_Results_Default_DB_4-16.xlsx")
    gt_path = Path("build/meta-tagging-clean/curated_reference.csv")
    parquet_path = Path("build/results/323cfbbc/atelier_embeddings.parquet")
    for p in (xlsx_path, gt_path, parquet_path):
        if not p.is_file():
            log.error("missing %s", p)
            return 1

    gt = _load_curated_reference(gt_path)
    mnem_to_code = _mnemonic_to_code()

    # --- Atelier predictions from parquet ---
    df_parq = pd.read_parquet(parquet_path)
    atelier_preds: dict[tuple[str, str], str] = {}
    for _, row in df_parq.iterrows():
        t = str(row.get("table_name", "")).strip()
        c = str(row.get("column_name", "")).strip()
        code = str(row.get("predicted_code", "")).strip()
        if t and c and code and not _is_reference_col(c):
            atelier_preds[(t, c)] = code

    # --- Baseline coverage from xlsx ---
    baseline_seen: set[tuple[str, str]] = set()
    baseline_preds: dict[tuple[str, str], str] = {}
    per_table_audit: dict[str, dict] = {}

    xl = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names
                   if s not in ("Overview", "Missed Classifications")]
    for sheet in data_sheets:
        table = _normalize_table(sheet)
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

        header_row_idx = None
        for i in range(min(3, len(df))):
            v0 = df.iloc[i].iloc[0]
            if pd.notna(v0) and str(v0).strip() == "column_name":
                header_row_idx = i
                break
        if header_row_idx is None:
            log.warning("no header in %s; skipping", sheet)
            continue
        hdr = df.iloc[header_row_idx]

        def _col_for(label: str) -> int | None:
            for j in range(len(hdr)):
                v = hdr.iloc[j]
                if pd.notna(v) and str(v).strip() == label:
                    return j
            return None

        col_gopala_col = _col_for("Column Name")
        col_gopala_tags = _col_for("Suggested Classification Tags")

        seen_n = tagged_n = declined_n = 0
        if col_gopala_col is None:
            per_table_audit[table] = {
                "baseline_section_present": False,
                "baseline_seen": 0,
                "baseline_tagged": 0,
                "baseline_declined": 0,
            }
            continue

        for i in range(header_row_idx + 1, len(df)):
            row = df.iloc[i]
            gval = row.iloc[col_gopala_col]
            if not pd.notna(gval):
                continue
            col_name = str(gval).strip()
            if not col_name or _is_reference_col(col_name):
                continue
            baseline_seen.add((table, col_name))
            seen_n += 1

            tag_val = (row.iloc[col_gopala_tags]
                       if col_gopala_tags is not None else None)
            if pd.notna(tag_val):
                tag_clean = str(tag_val).replace("\n", "").strip()
                code = mnem_to_code.get(tag_clean, "")
                if code:
                    baseline_preds[(table, col_name)] = code
                    tagged_n += 1
                else:
                    declined_n += 1
            else:
                declined_n += 1

        per_table_audit[table] = {
            "baseline_section_present": True,
            "baseline_seen": seen_n,
            "baseline_tagged": tagged_n,
            "baseline_declined": declined_n,
        }

    # --- Intersection with curated reference ---
    gt_keys = set(gt.keys())
    intersection = gt_keys & baseline_seen
    absent_from_baseline = gt_keys - baseline_seen
    declined_by_baseline = intersection - set(baseline_preds.keys())

    # Per-table absent counts (curated-ref cols only)
    absent_per_table: dict[str, int] = {}
    for t, _c in absent_from_baseline:
        absent_per_table[t] = absent_per_table.get(t, 0) + 1

    # Per-table intersection audit — restricted to curated-reference cols,
    # not "every row baseline produced". The phase-gate question is how
    # much of the curated reference baseline covered, not whether baseline
    # tagged extra columns we don't resolve.
    intersect_seen_per_table: dict[str, int] = {}
    intersect_tagged_per_table: dict[str, int] = {}
    for k in intersection:
        intersect_seen_per_table[k[0]] = intersect_seen_per_table.get(k[0], 0) + 1
        if k in baseline_preds:
            intersect_tagged_per_table[k[0]] = intersect_tagged_per_table.get(k[0], 0) + 1

    def _score(preds, keys):
        n = len(keys)
        n_scored = n_exact = n_hier = n_overspec = n_wrong = 0
        for k in keys:
            gt_code = gt[k]["reference_code"]
            pred = preds.get(k, "")
            if not pred:
                continue
            n_scored += 1
            if pred == gt_code:
                n_exact += 1; n_hier += 1
            elif _hier_match(pred, gt_code):
                n_hier += 1; n_overspec += 1
            else:
                n_wrong += 1
        return {
            "n": n,
            "scored": n_scored,
            "exact": n_exact,
            "hier": n_hier,
            "overspec": n_overspec,
            "wrong_class": n_wrong,
            "unpred": n - n_scored,
            "exact_pct": round(n_exact / n, 4) if n else 0.0,
            "hier_pct": round(n_hier / n, 4) if n else 0.0,
            "exact_pct_when_predicts": round(n_exact / n_scored, 4) if n_scored else 0.0,
            "hier_pct_when_predicts": round(n_hier / n_scored, 4) if n_scored else 0.0,
            "coverage_pct": round(n_scored / n, 4) if n else 0.0,
        }

    # Segments
    atelier_full = _score(atelier_preds, gt_keys)
    atelier_intersect = _score(atelier_preds, intersection)
    atelier_declined = _score(atelier_preds, declined_by_baseline)
    atelier_absent = _score(atelier_preds, absent_from_baseline)
    baseline_full = _score(baseline_preds, gt_keys)          # naive, vs. all 239
    baseline_intersect = _score(baseline_preds, intersection)

    # Per-table matrix on intersection only
    per_table_intersection: dict[str, dict] = {}
    for (t, c) in intersection:
        d = per_table_intersection.setdefault(
            t, {"n": 0, "a_exact": 0, "a_hier": 0, "b_exact": 0, "b_hier": 0}
        )
        d["n"] += 1
        gt_code = gt[(t, c)]["reference_code"]
        a = atelier_preds.get((t, c), "")
        b = baseline_preds.get((t, c), "")
        if a == gt_code: d["a_exact"] += 1
        if _hier_match(a, gt_code): d["a_hier"] += 1
        if b == gt_code: d["b_exact"] += 1
        if _hier_match(b, gt_code): d["b_hier"] += 1

    # Summarize
    out = {
        "inputs": {
            "xlsx": str(xlsx_path),
            "curated_reference": str(gt_path),
            "atelier_parquet": str(parquet_path),
        },
        "curated_reference_resolvable": len(gt_keys),
        "baseline_seen_total": len(baseline_seen),
        "intersection_size": len(intersection),
        "baseline_absent_count": len(absent_from_baseline),
        "baseline_declined_count": len(declined_by_baseline),
        "per_table_audit": per_table_audit,
        "absent_per_table": absent_per_table,
        "atelier_vs_curated_full": atelier_full,
        "baseline_vs_curated_naive_full": baseline_full,
        "atelier_on_intersection": atelier_intersect,
        "baseline_on_intersection": baseline_intersect,
        "atelier_on_baseline_declined": atelier_declined,
        "atelier_on_baseline_absent": atelier_absent,
        "per_table_intersection": per_table_intersection,
    }

    out_dir = Path("build/results/parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconciliation.json").write_text(json.dumps(out, indent=2))

    # Markdown report
    def pct(x): return f"{x:.2%}"
    lines = [
        "# Apples-to-apples reconciliation — Atelier vs. baseline classifier",
        "",
        "Both pipelines (the UAT xlsx baseline and the Atelier parquet) are "
        "scored here against the same curated reference, on the same columns. "
        "The three sections that matter:",
        "",
        "1. How much of the curated reference each pipeline saw.",
        "2. How each pipeline performed on the shared intersection.",
        "3. Where Atelier extends coverage beyond what the baseline produced.",
        "",
        "**Sources**",
        "",
        f"- curated reference: `{gt_path.name}` — {len(gt_keys)} resolvable columns",
        f"- baseline output:   `{xlsx_path.name}`",
        f"- atelier output:    `{parquet_path.name}` (run 323cfbbc)",
        "",
        "## 1. Coverage accounting",
        "",
        f"- Curated-reference resolvable columns: **{len(gt_keys)}**",
        f"- In both baseline input and curated reference: **{len(intersection)}**",
        f"- In curated reference, not in baseline input: **{len(absent_from_baseline)}**",
        f"- Baseline saw but did not tag: **{len(declined_by_baseline)}**",
        "",
        "### Per-table audit (curated-reference cols only)",
        "",
        "| Table | curated-ref cols | baseline saw | baseline tagged | baseline declined | absent from baseline |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted({k[0] for k in gt_keys}):
        gt_n = sum(1 for k in gt_keys if k[0] == t)
        seen_n = intersect_seen_per_table.get(t, 0)
        tagged_n = intersect_tagged_per_table.get(t, 0)
        declined_n = seen_n - tagged_n
        absent_n = absent_per_table.get(t, 0)
        lines.append(
            f"| {t} | {gt_n} | {seen_n} | {tagged_n} | {declined_n} | {absent_n} |"
        )

    lines += [
        "",
        "## 2. Performance on the shared intersection",
        "",
        f"Both pipelines scored on the same **{len(intersection)}** curated-reference "
        "columns that appear in baseline's input.",
        "",
        "| Metric | Baseline | Atelier | Δ |",
        "|---|---|---|---|",
        f"| coverage (tagged / eligible)  | {pct(baseline_intersect['coverage_pct'])} ({baseline_intersect['scored']}/{baseline_intersect['n']}) | {pct(atelier_intersect['coverage_pct'])} ({atelier_intersect['scored']}/{atelier_intersect['n']}) | "
        f"{(atelier_intersect['coverage_pct']-baseline_intersect['coverage_pct'])*100:+.2f} pp |",
        f"| exact accuracy (of eligible)  | {pct(baseline_intersect['exact_pct'])} ({baseline_intersect['exact']}/{baseline_intersect['n']}) | {pct(atelier_intersect['exact_pct'])} ({atelier_intersect['exact']}/{atelier_intersect['n']}) | "
        f"{(atelier_intersect['exact_pct']-baseline_intersect['exact_pct'])*100:+.2f} pp |",
        f"| hierarchical (of eligible)    | {pct(baseline_intersect['hier_pct'])} ({baseline_intersect['hier']}/{baseline_intersect['n']}) | {pct(atelier_intersect['hier_pct'])} ({atelier_intersect['hier']}/{atelier_intersect['n']}) | "
        f"{(atelier_intersect['hier_pct']-baseline_intersect['hier_pct'])*100:+.2f} pp |",
        f"| exact-when-predicts (precision) | {pct(baseline_intersect['exact_pct_when_predicts'])} | {pct(atelier_intersect['exact_pct_when_predicts'])} | "
        f"{(atelier_intersect['exact_pct_when_predicts']-baseline_intersect['exact_pct_when_predicts'])*100:+.2f} pp |",
        "",
        "Two observations:",
        "",
        f"- Baseline's exact-when-predicts is **{pct(baseline_intersect['exact_pct_when_predicts'])}** — it is highly accurate on the columns it chooses to tag.",
        f"- Atelier tags every eligible column ({pct(atelier_intersect['coverage_pct'])} coverage) and trades a small precision margin for {(atelier_intersect['exact_pct']-baseline_intersect['exact_pct'])*100:+.2f} pp net-exact gain.",
        "",
        "## 3. Columns baseline saw but did not tag",
        "",
        f"- N = **{atelier_declined['n']}**",
        f"- Atelier exact:        **{pct(atelier_declined['exact_pct'])}** ({atelier_declined['exact']}/{atelier_declined['n']})",
        f"- Atelier hierarchical: **{pct(atelier_declined['hier_pct'])}** ({atelier_declined['hier']}/{atelier_declined['n']})",
        "",
        "## 4. Columns not in baseline's input",
        "",
        f"- N = **{atelier_absent['n']}**",
        f"- Atelier exact:        **{pct(atelier_absent['exact_pct'])}** ({atelier_absent['exact']}/{atelier_absent['n']})",
        f"- Atelier hierarchical: **{pct(atelier_absent['hier_pct'])}** ({atelier_absent['hier']}/{atelier_absent['n']})",
        "",
        "## 5. Per-table side-by-side on the intersection",
        "",
        "| Table | N (shared) | Baseline exact | Baseline hier | Atelier exact | Atelier hier |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted(per_table_intersection):
        d = per_table_intersection[t]
        n = d["n"] or 1
        lines.append(
            f"| {t} | {d['n']} | "
            f"{d['b_exact']}/{d['n']} ({d['b_exact']/n:.2%}) | "
            f"{d['b_hier']}/{d['n']} ({d['b_hier']/n:.2%}) | "
            f"{d['a_exact']}/{d['n']} ({d['a_exact']/n:.2%}) | "
            f"{d['a_hier']}/{d['n']} ({d['a_hier']/n:.2%}) |"
        )

    lines += [
        "",
        "## Summary",
        "",
        f"- On the shared **{len(intersection)}-column** intersection, Atelier is "
        f"{pct(atelier_intersect['exact_pct'])} exact vs. baseline "
        f"{pct(baseline_intersect['exact_pct'])} exact "
        f"({(atelier_intersect['exact_pct']-baseline_intersect['exact_pct'])*100:+.2f} pp).",
        f"- Baseline's precision-when-predicts is {pct(baseline_intersect['exact_pct_when_predicts'])}; "
        "it is conservative and almost never wrong when it commits.",
        f"- Atelier additionally covers **{atelier_declined['n']}** columns baseline declined "
        f"(at {pct(atelier_declined['exact_pct'])} exact) and "
        f"**{atelier_absent['n']}** columns baseline was never asked about "
        f"(at {pct(atelier_absent['exact_pct'])} exact).",
        "- Both pipelines are measured against the same curated reference; any "
        "improvements to that reference will re-score both uniformly.",
    ]

    (out_dir / "reconciliation.md").write_text("\n".join(lines) + "\n")

    print("\n=== reconciliation ===")
    print(f"curated-ref resolvable: {len(gt_keys)}")
    print(f"baseline saw: {len(intersection)}  absent: {len(absent_from_baseline)}  declined: {len(declined_by_baseline)}")
    print(f"baseline intersect: exact={pct(baseline_intersect['exact_pct'])} hier={pct(baseline_intersect['hier_pct'])} cov={pct(baseline_intersect['coverage_pct'])}")
    print(f"atelier  intersect: exact={pct(atelier_intersect['exact_pct'])} hier={pct(atelier_intersect['hier_pct'])} cov={pct(atelier_intersect['coverage_pct'])}")
    print(f"atelier on baseline-declined: exact={pct(atelier_declined['exact_pct'])} ({atelier_declined['exact']}/{atelier_declined['n']})")
    print(f"atelier on baseline-absent:   exact={pct(atelier_absent['exact_pct'])} ({atelier_absent['exact']}/{atelier_absent['n']})")
    print(f"\n  md  : {out_dir / 'reconciliation.md'}")
    print(f"  json: {out_dir / 'reconciliation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
