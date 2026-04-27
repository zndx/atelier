#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Re-score an Atelier run's parquet against the curated reference.

Reuses the scoring structure from ``score_uat_xlsx.py`` so the two
outputs are directly comparable.

Usage::

    uv run python scripts/parity/rescore_parquet.py [run_id]

Defaults to ``323cfbbc``.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


log = logging.getLogger("rescore_parquet")


def _hier_match(pred: str, gt: str) -> bool:
    if not pred or not gt:
        return False
    if pred == gt:
        return True
    return gt.startswith(pred + ".") or pred.startswith(gt + ".")


def _is_reference_col(name: str) -> bool:
    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE
    return bool(_REFERENCE_COL_RE.match(name))


def _load_curated_reference(path: Path) -> dict[tuple[str, str], dict]:
    ref: dict[tuple[str, str], dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["derivation"] == "unresolved":
                continue
            ref[(row["table"], row["column"])] = row
    return ref


def _score_arm(arm: str, preds: dict[tuple[str, str], str],
               reference: dict[tuple[str, str], dict]) -> dict:
    per_table: dict[str, dict] = {}
    n_total = n_scored = n_exact = n_hier = 0
    n_overspec = n_wrong = n_unpred = 0
    for (table, col), row in reference.items():
        n_total += 1
        ref_code = row["reference_code"]
        pt = per_table.setdefault(
            table,
            {"n": 0, "exact": 0, "hier": 0, "unpred": 0,
             "overspec": 0, "wrong_class": 0},
        )
        pt["n"] += 1
        pred = preds.get((table, col), "")
        if not pred:
            n_unpred += 1
            pt["unpred"] += 1
            continue
        n_scored += 1
        ex = pred == ref_code
        hi = _hier_match(pred, ref_code)
        if ex:
            n_exact += 1; pt["exact"] += 1
        elif hi:
            n_overspec += 1; pt["overspec"] += 1
        else:
            n_wrong += 1; pt["wrong_class"] += 1
        if hi:
            n_hier += 1; pt["hier"] += 1
    return {
        "arm": arm,
        "total_reference_columns": n_total,
        "scored": n_scored,
        "unpredicted": n_unpred,
        "exact_accuracy": round(n_exact / n_total, 4) if n_total else 0.0,
        "exact_accuracy_excluding_unpred": (
            round(n_exact / n_scored, 4) if n_scored else 0.0
        ),
        "hierarchical_accuracy": round(n_hier / n_total, 4) if n_total else 0.0,
        "over_specified": n_overspec,
        "wrong_class": n_wrong,
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

    run_id = sys.argv[1] if len(sys.argv) > 1 else "323cfbbc"
    parquet = Path(f"build/results/{run_id}/atelier_embeddings.parquet")
    ref_path = Path("build/meta-tagging-clean/curated_reference.csv")
    if not parquet.is_file():
        log.error("missing %s", parquet); return 1
    if not ref_path.is_file():
        log.error("missing %s", ref_path); return 1

    reference = _load_curated_reference(ref_path)
    df = pd.read_parquet(parquet)

    # Collect predictions, skipping reference columns.
    llm_preds: dict[tuple[str, str], str] = {}
    dst_preds: dict[tuple[str, str], str] = {}
    for _, r in df.iterrows():
        table = str(r["table_name"]).strip()
        col = str(r["column_name"]).strip()
        if _is_reference_col(col):
            continue
        llm_code = str(r.get("llm_code", "") or "").strip()
        dst_code = str(r.get("predicted_code", "") or "").strip()
        if llm_code:
            llm_preds[(table, col)] = llm_code
        if dst_code:
            dst_preds[(table, col)] = dst_code

    atelier_llm = _score_arm("atelier_llm_only", llm_preds, reference)
    atelier_dst = _score_arm("atelier_dst_fused", dst_preds, reference)

    out_dir = Path("build/results/parity")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "atelier_scored.json").write_text(json.dumps({
        "run_id": run_id,
        "parquet": str(parquet),
        "curated_reference_csv": str(ref_path),
        "atelier_llm_only": atelier_llm,
        "atelier_dst_fused": atelier_dst,
    }, indent=2))

    # Short markdown
    lines = [
        f"# Atelier parquet `{run_id}` scored against curated reference",
        "",
        f"Source: `{parquet}`   ·   reference: `{ref_path}`",
        f"Resolvable columns: **{len(reference)}** (reference columns excluded)",
        "",
        "## Arm A — LLM-only (`llm_code`)",
        "",
        f"- exact:        **{atelier_llm['exact_accuracy']:.2%}**  ({atelier_llm['scored']}/{atelier_llm['total_reference_columns']} scored)",
        f"- hierarchical: **{atelier_llm['hierarchical_accuracy']:.2%}**",
        f"- over-specified: {atelier_llm['over_specified']}   wrong-class: {atelier_llm['wrong_class']}   unpredicted: {atelier_llm['unpredicted']}",
        "",
        "## Arm B — DST-fused (`predicted_code`)",
        "",
        f"- exact:        **{atelier_dst['exact_accuracy']:.2%}**  ({atelier_dst['scored']}/{atelier_dst['total_reference_columns']} scored)",
        f"- hierarchical: **{atelier_dst['hierarchical_accuracy']:.2%}**",
        f"- over-specified: {atelier_dst['over_specified']}   wrong-class: {atelier_dst['wrong_class']}   unpredicted: {atelier_dst['unpredicted']}",
        "",
        "## Per-table",
        "",
        "| Table | N | LLM-only exact / hier | DST-fused exact / hier |",
        "|---|---|---|---|",
    ]
    tables = sorted(set(atelier_llm["per_table"]) | set(atelier_dst["per_table"]))
    for t in tables:
        a = atelier_llm["per_table"].get(t, {})
        b = atelier_dst["per_table"].get(t, {})
        lines.append(
            f"| {t} | {a.get('n') or b.get('n')} | "
            f"{a.get('exact_pct', 0):.2%} / {a.get('hier_pct', 0):.2%} | "
            f"{b.get('exact_pct', 0):.2%} / {b.get('hier_pct', 0):.2%} |"
        )
    (out_dir / "atelier_scored.md").write_text("\n".join(lines) + "\n")

    print(f"\n=== Atelier {run_id} scored ===")
    print(f"LLM-only    exact={atelier_llm['exact_accuracy']:.4f}  hier={atelier_llm['hierarchical_accuracy']:.4f}")
    print(f"DST-fused   exact={atelier_dst['exact_accuracy']:.4f}  hier={atelier_dst['hierarchical_accuracy']:.4f}")
    print(f"\n  json: {out_dir / 'atelier_scored.json'}")
    print(f"  md  : {out_dir / 'atelier_scored.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
