#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Phase-gate validation: LLM-agreement thesis on the meta-tagging source.

Drives the pipeline end-to-end against the locally-mounted meta-tagging
source with:
  - 100% LLM coverage (mc_sample_fraction = 1.0)
  - CatBoost fit-to-LLM enabled
  - Quarantined patterns (already compiled-in)

Reports accuracy per table vs. raw LLM baseline and produces a
local summary.  No data from the meta-tagging mount is written back
to the repository — all results land in build/results/{run_id}/
(gitignored).

Usage::
    uv run python scripts/validate_phase_gate.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def main() -> int:
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.meta_tagging_source import resolve_meta_tagging_mount
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.config import load_config
    from atelier.config_overlay import apply_to_config, clear_overlay, set_overlay

    mount = resolve_meta_tagging_mount()
    if mount is None:
        log.error("Meta-tagging mount not found — set ATELIER_META_TAGGING_DIR")
        return 2

    log.info("Mount: %s", mount)

    clear_overlay()
    set_overlay({
        "mc_sample_fraction": 1.0,
        "classify_catboost_fit_to_llm": True,
        "classify_catboost_fit_to_llm_min_labels": 30,
    })

    cfg = apply_to_config(load_config())
    log.info(
        "Overlay applied: sample_fraction=%s  fit_to_llm=%s  min_labels=%s",
        cfg.mc_sample_fraction, cfg.classify_catboost_fit_to_llm,
        cfg.classify_catboost_fit_to_llm_min_labels,
    )

    fsm = AgentFSM()
    result = run_classification_pipeline(
        cfg, fsm, source_id="meta-tagging",
    )

    run_id = result["run_id"]
    log.info("Run complete: %s", run_id)
    log.info("State: %s", result.get("state"))

    results_dir = Path("build/results") / run_id
    classifications_path = results_dir / "classifications.json"
    if not classifications_path.exists():
        log.error("classifications.json missing at %s", classifications_path)
        return 1

    classifications = json.loads(classifications_path.read_text())
    log.info("Classifications: %d", len(classifications))

    # Per-table accuracy on columns with a curated reference
    per_table: dict[str, list[bool]] = defaultdict(list)
    per_table_llm_only: dict[str, list[bool]] = defaultdict(list)
    for c in classifications:
        ref_code = c.get("reference_code")
        if not ref_code:
            continue
        table = c.get("table_name", "?")
        per_table[table].append(bool(c.get("matches_reference")))
        # Raw LLM vote is the top code in evidence_sources.llm
        srcs = c.get("evidence_sources") or {}
        llm_map = srcs.get("llm") or {}
        if llm_map:
            llm_top = max(llm_map.items(), key=lambda kv: kv[1])[0]
            per_table_llm_only[table].append(llm_top == ref_code)
        else:
            per_table_llm_only[table].append(False)

    print("\n" + "=" * 72)
    print(f"{'Table':<22} {'Fused':>10} {'LLM-only':>10} {'N':>6}  {'Fused vs LLM':>14}")
    print("-" * 72)
    for table in sorted(per_table):
        fused = per_table[table]
        llm_only = per_table_llm_only[table]
        fused_acc = sum(fused) / len(fused) if fused else 0.0
        llm_acc = sum(llm_only) / len(llm_only) if llm_only else 0.0
        delta = fused_acc - llm_acc
        print(
            f"{table:<22} {fused_acc:>10.1%} {llm_acc:>10.1%} "
            f"{len(fused):>6}  {delta:+14.1%}"
        )

    all_fused = [v for lst in per_table.values() for v in lst]
    all_llm = [v for lst in per_table_llm_only.values() for v in lst]
    print("-" * 72)
    if all_fused:
        print(
            f"{'OVERALL':<22} {sum(all_fused)/len(all_fused):>10.1%} "
            f"{sum(all_llm)/len(all_llm):>10.1%} "
            f"{len(all_fused):>6}  {(sum(all_fused)/len(all_fused) - sum(all_llm)/len(all_llm)):+14.1%}"
        )
    print("=" * 72)

    # SAGE / SHAP artifacts presence
    sage = results_dir / "sage_importance.json"
    shap = results_dir / "shap_summary.json"
    print(f"\nSAGE: {sage.exists()}  SHAP: {shap.exists()}")
    if sage.exists():
        sage_data = json.loads(sage.read_text())
        names = sage_data.get("feature_names", [])
        vals = sage_data.get("importance_values", [])
        if names and vals:
            ranked = sorted(zip(names, vals), key=lambda kv: -kv[1])
            print("Top SAGE features:")
            for name, v in ranked[:5]:
                print(f"  {name:<22} {v:+.4f}")

    return 0 if result.get("state") == "CONVERGED" else 1


if __name__ == "__main__":
    sys.exit(main())
