#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Produce the final Atelier vs UAT delta report.

Reads ``uat_scored.json`` (Gopala LLM extracted from the UAT xlsx)
and ``atelier_scored.json`` (our 323cfbbc parquet) — both already
scored against the same authoritative GT — and emits a single
side-by-side markdown that can be shared with UAT.

Tabulates:
  * overall exact + hierarchical for both arms
  * per-table breakdown
  * coverage (Gopala often declines to classify; Atelier predicts
    for every column)
  * consequential disagreements — per-column list showing GT,
    Atelier's prediction, and Gopala's prediction, grouped by
    "Atelier wins", "Gopala wins", "both wrong"
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


log = logging.getLogger("report_delta")


def _load_gt(path: Path) -> dict[tuple[str, str], dict]:
    gt: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["derivation"] == "unresolved":
                continue
            gt[(row["table"], row["column"])] = row
    return gt


def _hier_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    return gt.startswith(pred + ".") or pred.startswith(gt + ".")


def _is_reference_col(name: str) -> bool:
    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE
    return bool(_REFERENCE_COL_RE.match(name))


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    import pandas as pd

    gt = _load_gt(Path("build/meta-tagging-clean/curated_reference.csv"))

    atelier_json = json.loads(
        Path("build/results/parity/atelier_scored.json").read_text()
    )
    uat_json = json.loads(
        Path("build/results/parity/uat_scored.json").read_text()
    )

    atelier_run = atelier_json["run_id"]
    atelier_dst = atelier_json["atelier_dst_fused"]
    gopala = uat_json["gopala_llm"]

    # Re-collect per-column predictions for the disagreements listing.
    parquet = Path(f"build/results/{atelier_run}/atelier_embeddings.parquet")
    df = pd.read_parquet(parquet)
    atelier_preds: dict[tuple[str, str], str] = {}
    for _, r in df.iterrows():
        table = str(r["table_name"]).strip()
        col = str(r["column_name"]).strip()
        if _is_reference_col(col):
            continue
        code = str(r.get("predicted_code", "") or "").strip()
        if code:
            atelier_preds[(table, col)] = code

    # Parse xlsx again for Gopala per-column predictions.
    gopala_preds: dict[tuple[str, str], str] = {}
    from atelier.classify.meta_tagging_source import _read_annotation_rows
    records = _read_annotation_rows(Path("build/meta-tagging/annotations.csv"))
    mnem_to_code = {}
    for r in records:
        a = (r.get("annotation") or "").strip()
        c = r.get("id")
        if a and c and a not in mnem_to_code:
            mnem_to_code[a] = c

    xlsx = Path("build/Atelier_Results_Default_DB_4-16.xlsx")
    xl = pd.ExcelFile(xlsx)
    name_map = {"customer_pci_data": "customer_pii_pci_data"}
    for sheet in xl.sheet_names:
        if sheet in ("Overview", "Missed Classifications"):
            continue
        table = name_map.get(sheet.lower(), sheet.lower())
        xdf = pd.read_excel(xlsx, sheet_name=sheet, header=None)
        # Find sub-header
        hdr_idx = None
        for i in range(min(3, len(xdf))):
            if pd.notna(xdf.iloc[i].iloc[0]) and str(xdf.iloc[i].iloc[0]).strip() == "column_name":
                hdr_idx = i
                break
        if hdr_idx is None:
            continue
        hdr = xdf.iloc[hdr_idx]

        def col_for(label: str) -> int | None:
            for j in range(len(hdr)):
                v = hdr.iloc[j]
                if pd.notna(v) and str(v).strip() == label:
                    return j
            return None

        c_col = col_for("Column Name")
        c_tag = col_for("Suggested Classification Tags")
        if c_col is None or c_tag is None:
            continue
        for i in range(hdr_idx + 1, len(xdf)):
            row = xdf.iloc[i]
            cn, tg = row.iloc[c_col], row.iloc[c_tag]
            if pd.isna(cn) or pd.isna(tg):
                continue
            col_name = str(cn).strip()
            if not col_name or _is_reference_col(col_name):
                continue
            tag_clean = str(tg).replace("\n", "").strip()
            code = mnem_to_code.get(tag_clean, "")
            if code:
                gopala_preds[(table, col_name)] = code

    # Build per-column comparison
    both_right: list[tuple] = []
    atelier_only: list[tuple] = []
    gopala_only: list[tuple] = []
    both_wrong: list[tuple] = []
    atelier_covered_gopala_didnt: list[tuple] = []
    both_uncovered: list[tuple] = []
    for (table, col), row in gt.items():
        gt_code = row["reference_code"]
        a = atelier_preds.get((table, col), "")
        g = gopala_preds.get((table, col), "")
        a_right = a == gt_code
        g_right = g == gt_code
        if a and g:
            if a_right and g_right:
                both_right.append((table, col, gt_code, a, g, row["derivation"]))
            elif a_right:
                atelier_only.append((table, col, gt_code, a, g, row["derivation"]))
            elif g_right:
                gopala_only.append((table, col, gt_code, a, g, row["derivation"]))
            else:
                both_wrong.append((table, col, gt_code, a, g, row["derivation"]))
        elif a and not g:
            # Gopala declined; track separately
            atelier_covered_gopala_didnt.append(
                (table, col, gt_code, a, g, row["derivation"], a_right)
            )
        elif g and not a:
            # Shouldn't happen — Atelier predicts for everything
            pass
        else:
            both_uncovered.append((table, col, gt_code, row["derivation"]))

    # Delta summary
    overall = {
        "authoritative_gt_total": len(gt),
        "atelier": {
            "exact": atelier_dst["exact_accuracy"],
            "hierarchical": atelier_dst["hierarchical_accuracy"],
            "coverage": round(len(atelier_preds) / len(gt), 4),
        },
        "gopala": {
            "exact": gopala["exact_accuracy"],
            "hierarchical": gopala["hierarchical_accuracy"],
            "coverage": round(len(gopala_preds) / len(gt), 4),
            "precision_when_predicts": round(
                gopala["exact_accuracy"] * len(gt) /
                (gopala["scored"] or 1), 4
            ),
        },
        "delta_overall_exact": round(
            atelier_dst["exact_accuracy"] - gopala["exact_accuracy"], 4
        ),
        "delta_overall_hier": round(
            atelier_dst["hierarchical_accuracy"] - gopala["hierarchical_accuracy"], 4
        ),
        "disagreements": {
            "both_right": len(both_right),
            "atelier_only_right": len(atelier_only),
            "gopala_only_right": len(gopala_only),
            "both_wrong": len(both_wrong),
            "atelier_covered_gopala_didnt": len(atelier_covered_gopala_didnt),
            "both_uncovered": len(both_uncovered),
        },
    }

    out_dir = Path("build/results/parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "delta_summary.json").write_text(json.dumps(overall, indent=2))

    # Render the markdown
    id_to_label = {r["id"]: (r.get("ontology") or "") for r in records}

    def _fmt(p, gt):
        mark = "✓" if p == gt else ("○" if _hier_match(p, gt) else "✗")
        return f"{mark} {p or '(none)'}"

    lines = [
        "# Atelier vs UAT (Gopala LLM) — scored against authoritative GT",
        "",
        f"Curated reference: `build/meta-tagging-clean/curated_reference.csv` ({len(gt)} resolvable columns; reference columns excluded by invariant)",
        f"Atelier parquet: `build/results/{atelier_run}/atelier_embeddings.parquet`",
        f"UAT xlsx:        `build/Atelier_Results_Default_DB_4-16.xlsx`",
        "",
        "## Overall",
        "",
        "| | Exact | Hierarchical | Coverage |",
        "|---|---|---|---|",
        f"| **Atelier (DST-fused)** | **{atelier_dst['exact_accuracy']:.2%}** | **{atelier_dst['hierarchical_accuracy']:.2%}** | {len(atelier_preds) / len(gt):.2%} ({len(atelier_preds)}/{len(gt)}) |",
        f"| UAT / Gopala LLM        | {gopala['exact_accuracy']:.2%}     | {gopala['hierarchical_accuracy']:.2%} | {len(gopala_preds) / len(gt):.2%} ({len(gopala_preds)}/{len(gt)}) |",
        f"| **Δ (Atelier − Gopala)** | **{atelier_dst['exact_accuracy'] - gopala['exact_accuracy']:+.2%}** | **{atelier_dst['hierarchical_accuracy'] - gopala['hierarchical_accuracy']:+.2%}** | +{(len(atelier_preds) - len(gopala_preds))/len(gt):.2%} |",
        "",
        f"Gopala precision-when-predicts: **{(overall['gopala']['exact'] * len(gt)) / (gopala['scored'] or 1):.2%}** — when Gopala makes a prediction, it's nearly always right; the gap is dominated by coverage, not accuracy.",
        "",
        "## Per-table",
        "",
        "| Table | GT cols | Atelier exact / hier | Gopala exact / hier | Gopala coverage |",
        "|---|---|---|---|---|",
    ]
    tables = sorted(
        set(atelier_dst["per_table"]) | set(gopala["per_table"])
    )
    for t in tables:
        a = atelier_dst["per_table"].get(t, {})
        g = gopala["per_table"].get(t, {})
        n = a.get("n") or g.get("n") or 0
        gop_cov = (n - g.get("unpred", n)) / n if n else 0
        lines.append(
            f"| {t} | {n} | "
            f"{a.get('exact_pct', 0):.2%} / {a.get('hier_pct', 0):.2%} | "
            f"{g.get('exact_pct', 0):.2%} / {g.get('hier_pct', 0):.2%} | "
            f"{gop_cov:.2%} |"
        )

    lines += [
        "",
        "## Disagreement topology",
        "",
        "| Bucket | n |",
        "|---|---|",
        f"| both arms correct | {len(both_right)} |",
        f"| **Atelier correct, Gopala wrong** | **{len(atelier_only)}** |",
        f"| Gopala correct, Atelier wrong | {len(gopala_only)} |",
        f"| both wrong (different answers) | {len(both_wrong)} |",
        f"| Atelier predicted, Gopala declined | {len(atelier_covered_gopala_didnt)}  (Atelier correct on {sum(1 for r in atelier_covered_gopala_didnt if r[-1])}) |",
        f"| both declined / unpredictable | {len(both_uncovered)} |",
        "",
    ]

    if atelier_only:
        lines += ["## Columns where Atelier is right and Gopala is wrong", "",
                  "| Table | Column | GT | Atelier | Gopala | Derivation |",
                  "|---|---|---|---|---|---|"]
        for t, c, g, a, gl, d in atelier_only[:20]:
            lines.append(
                f"| {t} | {c} | `{g}` ({id_to_label.get(g, '?')}) | `{a}` | `{gl}` | {d} |"
            )
        if len(atelier_only) > 20:
            lines.append(f"| ... ({len(atelier_only) - 20} more) | | | | | |")
        lines.append("")

    if gopala_only:
        lines += ["## Columns where Gopala is right and Atelier is wrong", "",
                  "| Table | Column | GT | Atelier | Gopala | Derivation |",
                  "|---|---|---|---|---|---|"]
        for t, c, g, a, gl, d in gopala_only[:20]:
            lines.append(
                f"| {t} | {c} | `{g}` ({id_to_label.get(g, '?')}) | `{a}` | `{gl}` | {d} |"
            )
        if len(gopala_only) > 20:
            lines.append(f"| ... ({len(gopala_only) - 20} more) | | | | | |")
        lines.append("")

    if atelier_covered_gopala_didnt:
        lines += [
            "## Columns Atelier classified but Gopala declined",
            "",
            "Atelier's do-no-harm coverage here is the quantitative expression of reproducibility — these are the columns where the DST-fused pipeline makes a call that UAT's LLM-only run left unanswered.",
            "",
            "| Table | Column | GT | Atelier | Atelier correct? |",
            "|---|---|---|---|---|",
        ]
        correct_count = sum(1 for r in atelier_covered_gopala_didnt if r[-1])
        for t, c, g, a, _, d, right in atelier_covered_gopala_didnt[:20]:
            mark = "✓" if right else "✗"
            lines.append(f"| {t} | {c} | `{g}` ({id_to_label.get(g, '?')}) | `{a}` | {mark} |")
        if len(atelier_covered_gopala_didnt) > 20:
            lines.append(f"| ... ({len(atelier_covered_gopala_didnt) - 20} more) | | | | |")
        lines.append("")
        lines.append(f"Atelier accuracy on columns Gopala declined: **{correct_count}/{len(atelier_covered_gopala_didnt)} = {correct_count/len(atelier_covered_gopala_didnt):.2%}**")
        lines.append("")

    (out_dir / "delta_report.md").write_text("\n".join(lines) + "\n")

    print("\n=== Atelier vs UAT delta ===")
    print(f"Atelier exact {atelier_dst['exact_accuracy']:.4f}   Gopala exact {gopala['exact_accuracy']:.4f}   Δ {atelier_dst['exact_accuracy'] - gopala['exact_accuracy']:+.4f}")
    print(f"Atelier hier  {atelier_dst['hierarchical_accuracy']:.4f}   Gopala hier  {gopala['hierarchical_accuracy']:.4f}   Δ {atelier_dst['hierarchical_accuracy'] - gopala['hierarchical_accuracy']:+.4f}")
    print(f"Atelier coverage {len(atelier_preds)}/{len(gt)}  ·  Gopala coverage {len(gopala_preds)}/{len(gt)}")
    print(f"Disagreements — Atelier-only-right: {len(atelier_only)}   Gopala-only-right: {len(gopala_only)}   both-wrong: {len(both_wrong)}")
    print(f"\n  summary : {out_dir / 'delta_summary.json'}")
    print(f"  report  : {out_dir / 'delta_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
