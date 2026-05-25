#!/usr/bin/env python3
"""scripts/audit_generator_coverage.py — Phase A of the corpus expansion plan.

Deterministic, no-LLM audit of synth generator coverage across the full
287-node taxonomy.  For every node:

  1. Resolve a generator via ``GeneratorRegistry.from_enrichment_payloads``
     (the same registry the production pipeline uses).
  2. If covered, run a 50-trial diversity probe with a fixed-seed RNG
     and record: distinct-output ratio, mean output length, top-1 share,
     sample outputs.
  3. Classify each node into one of:
       - ``missing``               — no generator at any layer
       - ``inferred_only``         — only regex-keyword-matched fallback
       - ``template_only``         — only sample-perturbation generator
       - ``low_diversity_handcoded`` — hand-coded but distinct_ratio < 0.5
       - ``ok``                    — hand-coded with adequate diversity

Outputs:
  build/svm_corpus_v2/coverage_audit.json  — full per-code metadata
                                              + priority-sorted gap list

The gap list drives Phase B (Agent SDK-driven generator authorship).

Usage:
  python scripts/audit_generator_coverage.py
  python scripts/audit_generator_coverage.py --payloads <path>
  python scripts/audit_generator_coverage.py --diversity-threshold 0.6
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflect_nhsvm import build_category_set
from atelier.classify.synth_registry import GeneratorRegistry
from atelier.classify.enrichment_loader import load_enrichment_payloads

log = logging.getLogger("audit_generator_coverage")

REPORT_DIR = Path("build/svm_corpus_v2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_JSON = REPORT_DIR / "coverage_audit.json"
DEFAULT_PAYLOADS = Path("build/data/svm_training/enrichment_payloads.json")

DIVERSITY_PROBE_N = 50
DIVERSITY_SEED = 7


def probe_generator(gen, n: int = DIVERSITY_PROBE_N, seed: int = DIVERSITY_SEED) -> dict:
    """Sample a generator n times; return diversity statistics."""
    rng = random.Random(seed)
    outputs: list[str] = []
    errors = 0
    for _ in range(n):
        try:
            out = gen(rng)
            outputs.append(str(out))
        except Exception as exc:  # noqa: BLE001 — generator may misbehave
            errors += 1
            outputs.append(f"<error: {type(exc).__name__}>")
    counts = Counter(outputs)
    distinct = len(counts)
    distinct_ratio = distinct / n if n else 0.0
    top1_share = (counts.most_common(1)[0][1] / n) if counts else 0.0
    mean_len = (sum(len(o) for o in outputs) / n) if outputs else 0.0
    return {
        "n_trials": n,
        "n_errors": errors,
        "distinct_outputs": distinct,
        "distinct_ratio": round(distinct_ratio, 4),
        "top1_share": round(top1_share, 4),
        "mean_length": round(mean_len, 2),
        "sample_outputs": outputs[:5],  # store first 5 for inspection
    }


def classify_gap(source: str | None, diversity_ratio: float, threshold: float) -> str:
    if source is None:
        return "missing"
    if source == "inferred":
        return "inferred_only"
    if source == "template":
        return "template_only"
    if source == "hand-coded":
        if diversity_ratio < threshold:
            return "low_diversity_handcoded"
        return "ok"
    return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS,
                    help=f"Enrichment payloads JSON (default: {DEFAULT_PAYLOADS})")
    ap.add_argument("--diversity-threshold", type=float, default=0.5,
                    help="distinct_ratio below this on hand-coded → low_diversity gap")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    log.info("=== Phase A: generator coverage audit ===")
    log.info("Loading category set...")
    cat_set = build_category_set()
    nodes = list(cat_set.all_categories)
    log.info("  %d total nodes", len(nodes))

    log.info("Loading enrichment payloads from %s ...", args.payloads)
    payloads = load_enrichment_payloads(json_path=args.payloads)
    log.info("  %d payload entries (mnemonic + code keyed; some duplicate)",
             len(payloads))

    log.info("Building registry via from_enrichment_payloads...")
    registry = GeneratorRegistry.from_enrichment_payloads(payloads, cat_set)
    coverage_summary = registry.coverage_summary(cat_set)
    log.info("  coverage by source: %s", coverage_summary)

    log.info("Probing generators for diversity (n=%d, seed=%d)...",
             DIVERSITY_PROBE_N, DIVERSITY_SEED)

    per_node: list[dict] = []
    gap_class_counts: Counter = Counter()
    for cat in nodes:
        code = cat.code
        mnemonic = getattr(cat, "abbrev", "") or ""
        spec = registry.get(code)
        source = spec.source if spec else None
        probe: dict | None = None
        diversity_ratio = 0.0
        if spec is not None:
            probe = probe_generator(spec.generator)
            diversity_ratio = probe["distinct_ratio"]
        gap_class = classify_gap(source, diversity_ratio, args.diversity_threshold)
        gap_class_counts[gap_class] += 1
        per_node.append({
            "code": code,
            "mnemonic": mnemonic,
            "label": getattr(cat, "label", "") or "",
            "source": source,
            "gap_class": gap_class,
            "probe": probe,
            "payload_present": mnemonic in payloads,
        })

    # Priority-sorted gap list: missing > inferred_only > template_only >
    # low_diversity_handcoded.  Within each tier, sort by mnemonic for
    # determinism.
    PRIORITY = {
        "missing": 0,
        "inferred_only": 1,
        "template_only": 2,
        "low_diversity_handcoded": 3,
        "ok": 99,
    }
    gap_nodes = [n for n in per_node if n["gap_class"] != "ok"]
    gap_nodes.sort(key=lambda n: (PRIORITY.get(n["gap_class"], 99), n["mnemonic"]))

    report = {
        "n_nodes_total": len(nodes),
        "n_payloads_loaded": len(payloads),
        "coverage_summary": dict(coverage_summary),
        "gap_class_counts": dict(gap_class_counts),
        "diversity_threshold": args.diversity_threshold,
        "diversity_probe_n": DIVERSITY_PROBE_N,
        "diversity_probe_seed": DIVERSITY_SEED,
        "per_node": per_node,
        "gap_list": [
            {"code": n["code"], "mnemonic": n["mnemonic"],
             "gap_class": n["gap_class"], "source": n["source"],
             "label": n["label"],
             "diversity_ratio": (n["probe"] or {}).get("distinct_ratio")}
            for n in gap_nodes
        ],
    }
    AUDIT_JSON.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", AUDIT_JSON)

    # Human-readable summary
    print()
    print("=== Coverage audit summary ===")
    print(f"  total nodes:        {len(nodes)}")
    print(f"  coverage by source:")
    for source, count in sorted(coverage_summary.items(), key=lambda kv: -kv[1]):
        print(f"    {source:<25} {count}")
    print(f"  gap classification:")
    for gap_class, count in sorted(gap_class_counts.items(),
                                    key=lambda kv: PRIORITY.get(kv[0], 99)):
        print(f"    {gap_class:<25} {count}")
    print()
    print(f"  total gaps to fill via Phase B: {len(gap_nodes)}")
    print(f"  audit JSON: {AUDIT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
