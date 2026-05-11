#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Score the UAT-produced xlsx against our curated reference.

Input:
  build/Atelier_Results_Default_DB_4-16.xlsx
  build/meta-tagging-clean/curated_reference.csv   (from build_curated_reference.py)

The xlsx contains *both* pipelines' predictions side-by-side on each
data sheet (row 0 = section titles, row 1 = sub-headers, rows 2+ = data):

    col 0-4     Atelier:   column_name, predicted_code, predicted_label,
                           confidence, is_correct
    col 7-12    Gopala LLM: Asset, Column Name, Suggested Classification
                           Tags (mnemonic), Data Sensitivity,
                           Confidence Score, Explanation

We score both arms against our authoritative GT on exactly the same
set of columns (natural-named; reference columns excluded by
invariant).

Gopala's output uses Annotation *mnemonics* (``TRADESEC``,
``FINDOC``, etc.), not numeric codes.  We map mnemonic → code via
``annotations.csv``.

Output:
    build/results/parity/uat_scored.json      machine-readable
    build/results/parity/uat_scored.md        summary
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


log = logging.getLogger("score_uat_xlsx")


def _load_curated_reference(path: Path) -> dict[tuple[str, str], dict]:
    gt: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["derivation"] == "unresolved":
                continue
            gt[(row["table"], row["column"])] = row
    return gt


def _mnemonic_to_code() -> dict[str, str]:
    """Build {Annotation → ID} from build/meta-tagging/annotations.csv."""
    from atelier.classify.meta_tagging_source import _read_annotation_rows
    records = _read_annotation_rows(
        Path("build/meta-tagging/annotations.csv")
    )
    m: dict[str, str] = {}
    for r in records:
        ann = (r.get("annotation") or "").strip()
        code = r.get("id")
        if ann and code and ann not in m:
            m[ann] = code
    return m


def _normalize_table(xlsx_sheet_name: str) -> str:
    """Map xlsx sheet name → GT table name.

    The xlsx uses ``customer_pci_data`` and ``Identity_data`` (mixed
    case); our source corpus uses ``customer_pii_pci_data`` and
    ``identity_data``.
    """
    lowered = xlsx_sheet_name.lower()
    return {
        "customer_pci_data": "customer_pii_pci_data",
    }.get(lowered, lowered)


def _hier_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    return gt.startswith(pred + ".") or pred.startswith(gt + ".")


def _is_reference_col(name: str) -> bool:
    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE
    return bool(_REFERENCE_COL_RE.match(name))


def _score_arm(arm: str, preds: dict[tuple[str, str], str],
               gt: dict[tuple[str, str], dict]) -> dict:
    """Compute per-table + overall exact / hierarchical accuracy."""
    per_table: dict[str, dict] = {}
    n_total = n_scored = n_exact = n_hier = n_unresolved = 0
    n_overspec = n_wrong_class = 0
    for (table, col), row in gt.items():
        n_total += 1
        pred = preds.get((table, col), "")
        if not pred:
            n_unresolved += 1
            continue
        n_scored += 1
        ex = pred == row["reference_code"]
        hi = _hier_match(pred, row["reference_code"])
        if ex:
            n_exact += 1
        elif hi:
            n_overspec += 1
        else:
            n_wrong_class += 1
        if hi:
            n_hier += 1
        pt = per_table.setdefault(
            table, {"n": 0, "exact": 0, "hier": 0, "unpred": 0,
                    "overspec": 0, "wrong_class": 0}
        )
        pt["n"] += 1
        if ex: pt["exact"] += 1
        if hi: pt["hier"] += 1
        if not ex and hi: pt["overspec"] += 1
        if not hi: pt["wrong_class"] += 1
    # Track unpredicted per table separately
    for (table, col), _ in gt.items():
        if (table, col) not in preds or not preds.get((table, col)):
            per_table.setdefault(
                table, {"n": 0, "exact": 0, "hier": 0, "unpred": 0,
                        "overspec": 0, "wrong_class": 0}
            )["unpred"] += 1

    return {
        "arm": arm,
        "total_gt_columns": n_total,
        "scored": n_scored,
        "unpredicted": n_unresolved,
        "exact_accuracy": round(n_exact / n_total, 4) if n_total else 0.0,
        "exact_accuracy_excluding_unpred": round(n_exact / n_scored, 4) if n_scored else 0.0,
        "hierarchical_accuracy": round(n_hier / n_total, 4) if n_total else 0.0,
        "over_specified": n_overspec,
        "wrong_class": n_wrong_class,
        "per_table": {
            t: {
                "n": v["n"],
                "exact": v["exact"],
                "exact_pct": round(v["exact"] / v["n"], 4) if v["n"] else 0.0,
                "hier": v["hier"],
                "hier_pct": round(v["hier"] / v["n"], 4) if v["n"] else 0.0,
                "overspec": v["overspec"],
                "wrong_class": v["wrong_class"],
                "unpred": v["unpred"],
            }
            for t, v in sorted(per_table.items())
        },
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    import pandas as pd

    xlsx_path = Path("build/Atelier_Results_Default_DB_4-16.xlsx")
    gt_path = Path("build/meta-tagging-clean/curated_reference.csv")
    if not xlsx_path.is_file():
        log.error("missing %s", xlsx_path); return 1
    if not gt_path.is_file():
        log.error("missing %s — run build_curated_reference.py first", gt_path)
        return 1

    gt = _load_curated_reference(gt_path)
    mnem_to_code = _mnemonic_to_code()
    log.info("gt columns (resolvable): %d", len(gt))
    log.info("mnemonic→code map: %d entries", len(mnem_to_code))

    # Collect both arms' predictions from the xlsx.
    # Sheets have inconsistent layouts; locate column indices per-sheet
    # by finding the sub-header row (contains 'column_name' at col 0).
    atelier_preds: dict[tuple[str, str], str] = {}
    gopala_preds: dict[tuple[str, str], str] = {}
    per_sheet_counts = {}
    xl = pd.ExcelFile(xlsx_path)
    data_sheets = [s for s in xl.sheet_names
                   if s not in ("Overview", "Missed Classifications")]
    for sheet in data_sheets:
        table = _normalize_table(sheet)
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

        # Find the sub-header row: first row with 'column_name' at col 0.
        header_row_idx = None
        for i in range(min(3, len(df))):
            v0 = df.iloc[i].iloc[0]
            if pd.notna(v0) and str(v0).strip() == "column_name":
                header_row_idx = i
                break
        if header_row_idx is None:
            log.warning("no header in sheet %s; skipping", sheet)
            continue
        hdr = df.iloc[header_row_idx]

        def _col_for(label: str) -> int | None:
            for j in range(len(hdr)):
                v = hdr.iloc[j]
                if pd.notna(v) and str(v).strip() == label:
                    return j
            return None

        col_atelier_id = _col_for("column_name")
        col_atelier_code = _col_for("predicted_code")
        col_gopala_col = _col_for("Column Name")
        col_gopala_tags = _col_for("Suggested Classification Tags")

        atelier_n = gopala_n = skipped_refs = 0
        for i in range(header_row_idx + 1, len(df)):
            row = df.iloc[i]
            # Atelier side
            if col_atelier_id is not None and col_atelier_code is not None:
                ate_id = row.iloc[col_atelier_id]
                if pd.notna(ate_id) and "." in str(ate_id):
                    col_name = str(ate_id).split(".", 1)[1].strip()
                    if _is_reference_col(col_name):
                        skipped_refs += 1
                    else:
                        code_val = row.iloc[col_atelier_code]
                        if pd.notna(code_val):
                            code = str(code_val).strip()
                            if code:
                                atelier_preds[(table, col_name)] = code
                                atelier_n += 1
            # Gopala side
            if col_gopala_col is not None and col_gopala_tags is not None:
                gopala_col = row.iloc[col_gopala_col]
                gopala_tag = row.iloc[col_gopala_tags]
                if pd.notna(gopala_col) and pd.notna(gopala_tag):
                    col_name = str(gopala_col).strip()
                    tag_clean = (
                        str(gopala_tag).replace("\n", "").strip()
                    )
                    if col_name and not _is_reference_col(col_name) and tag_clean:
                        code = mnem_to_code.get(tag_clean, "")
                        if code:
                            gopala_preds[(table, col_name)] = code
                            gopala_n += 1
        per_sheet_counts[table] = {
            "atelier_preds": atelier_n,
            "gopala_preds": gopala_n,
            "reference_cols_skipped": skipped_refs,
            "has_gopala_section": col_gopala_col is not None,
        }

    log.info("atelier predictions: %d · gopala predictions: %d",
             len(atelier_preds), len(gopala_preds))

    atelier_scored = _score_arm("atelier_xlsx", atelier_preds, gt)
    gopala_scored = _score_arm("gopala_llm_xlsx", gopala_preds, gt)

    out_dir = Path("build/results/parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "uat_scored.json"
    out_json.write_text(json.dumps({
        "input_xlsx": str(xlsx_path),
        "authoritative_gt_csv": str(gt_path),
        "sheet_counts": per_sheet_counts,
        "atelier": atelier_scored,
        "gopala_llm": gopala_scored,
    }, indent=2))

    # Short markdown summary
    lines = [
        "# UAT xlsx scored against curated reference",
        "",
        f"Source: `{xlsx_path}`   ·   GT: `{gt_path}`",
        f"GT resolvable columns: **{len(gt)}**  (reference columns excluded by invariant)",
        "",
        "## Atelier (predicted_code column)",
        "",
        f"- exact:      **{atelier_scored['exact_accuracy']:.2%}**  ({atelier_scored['scored']}/{atelier_scored['total_gt_columns']} scored)",
        f"- hierarchical: **{atelier_scored['hierarchical_accuracy']:.2%}**",
        f"- over-specified: {atelier_scored['over_specified']}   wrong-class: {atelier_scored['wrong_class']}   unpredicted: {atelier_scored['unpredicted']}",
        "",
        "## Gopala LLM (Suggested Classification Tags → code)",
        "",
        f"- exact:      **{gopala_scored['exact_accuracy']:.2%}**  ({gopala_scored['scored']}/{gopala_scored['total_gt_columns']} scored)",
        f"- hierarchical: **{gopala_scored['hierarchical_accuracy']:.2%}**",
        f"- over-specified: {gopala_scored['over_specified']}   wrong-class: {gopala_scored['wrong_class']}   unpredicted: {gopala_scored['unpredicted']}",
        "",
        "## Per-table",
        "",
        "| Table | N | Atelier exact / hier | Gopala exact / hier |",
        "|---|---|---|---|",
    ]
    tables = sorted(set(atelier_scored["per_table"]) | set(gopala_scored["per_table"]))
    for t in tables:
        a = atelier_scored["per_table"].get(t, {})
        g = gopala_scored["per_table"].get(t, {})
        lines.append(
            f"| {t} | {a.get('n') or g.get('n')} | "
            f"{a.get('exact_pct', 0):.2%} / {a.get('hier_pct', 0):.2%} | "
            f"{g.get('exact_pct', 0):.2%} / {g.get('hier_pct', 0):.2%} |"
        )
    (out_dir / "uat_scored.md").write_text("\n".join(lines) + "\n")

    print("\n=== UAT xlsx scored ===")
    print(f"Atelier side    exact={atelier_scored['exact_accuracy']:.4f}  hier={atelier_scored['hierarchical_accuracy']:.4f}")
    print(f"Gopala LLM side exact={gopala_scored['exact_accuracy']:.4f}  hier={gopala_scored['hierarchical_accuracy']:.4f}")
    print(f"\n  json: {out_json}")
    print(f"  md  : {out_dir / 'uat_scored.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
