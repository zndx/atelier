#!/usr/bin/env python3
"""scripts/audit_generator_coverage.py — Phase A of the corpus expansion plan.

Deterministic, no-LLM audit of synth generator coverage across the full
taxonomy.  For every node:

  1. Resolve a generator via ``GeneratorRegistry.from_enrichment_payloads``
     (the upstream Aegir registry; includes BFO/CCO ICE generators,
     template-based generators, and inferred fallbacks).
  2. Check whether a v1 (agent-authored, SVM-channel-scoped) generator
     exists at ``build/lib/generated/generators_v1.py``.
  3. Classify each node.  Under ``--require-v1-coverage`` (default ON
     for SVM-stage usage):

       - ``ok``                          — v1 generator exists
       - ``missing``                     — no coverage anywhere
       - ``ice_upstream_needs_v1``       — upstream ICE coverage but
                                           no v1; agent must author
       - ``template_upstream_needs_v1``  — upstream template coverage
                                           but no v1; agent must author
       - ``inferred_upstream_needs_v1``  — upstream inferred coverage
                                           but no v1; agent must author

     Under ``--no-require-v1-coverage`` (legacy diagnostic mode), the
     classification uses upstream-registry coverage only (ICE counts
     as ok), without the v1 requirement.

About ICE: ICE = Information Content Entity, a BFO/CCO ontology class
(see CommonCoreOntology/CommonCoreOntologies/.../InformationEntityOntology.ttl).
ICE generators are NOT legacy — they are a first-class upstream
Aegir-managed surface, used in other contexts.  The SVM stage requires
its own v1 generators (refined under the metrology loop) because the
SVM channel must train on agent-authored generators that the metrology
+ refinement machinery can steer.  ICE generators continue to exist
upstream; this audit just enforces that they don't satisfy SVM-stage
coverage on their own.

Outputs:
  build/svm_corpus_v2/coverage_audit.json  — full per-code metadata
                                              + priority-sorted gap list

The gap list drives Phase B (Agent SDK-driven generator authorship).

Usage:
  python scripts/audit_generator_coverage.py
  python scripts/audit_generator_coverage.py --payloads <path>
  python scripts/audit_generator_coverage.py --diversity-threshold 0.6
  python scripts/audit_generator_coverage.py --no-require-v1-coverage
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


def classify_gap(
    source: str | None,
    diversity_ratio: float,
    threshold: float,
    *,
    has_v1: bool = False,
    require_v1_coverage: bool = False,
) -> str:
    """Classify per-code coverage status.

    With ``require_v1_coverage=True`` (the SVM stage's default), any code
    NOT covered by a v1 (agent-authored) generator is a gap the agent
    must fill — even if ICE / template / inferred coverage exists in the
    upstream Aegir registry.  The hand-coded ICE generators remain a
    first-class upstream surface but no longer satisfy SVM-stage
    coverage.
    """
    if require_v1_coverage:
        if has_v1:
            # v1 covers this code; it's done regardless of any
            # upstream ICE / template / inferred coverage that may
            # also exist via the Aegir-managed registry.
            return "ok"
        # No v1 coverage — surface the underlying upstream-registry
        # situation in the gap_class so the agent's prioritization can
        # distinguish missing-everywhere vs has-upstream-ICE-but-needs-v1.
        # ICE = Information Content Entity (BFO/CCO ontology surface
        # managed upstream in Aegir) — NOT legacy; co-exists with the
        # SVM stage's v1 channel, just outside its scope.
        if source is None:
            return "missing"
        if source == "hand-coded":
            return "ice_upstream_needs_v1"
        if source == "template":
            return "template_upstream_needs_v1"
        if source == "inferred":
            return "inferred_upstream_needs_v1"
        return "unknown_upstream_needs_v1"

    # Legacy (non-v1-requiring) classification: ICE counts as ok.
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


def _load_v1_codes() -> set[str]:
    """Return the set of taxonomy codes with v1 (agent-authored) generators."""
    v1_path = Path("build/lib/generated/generators_v1.py")
    if not v1_path.exists():
        return set()
    import importlib.util
    spec = importlib.util.spec_from_file_location("generators_v1", v1_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = set()
    for code, gens in getattr(mod, "GENERATORS_BY_CODE", {}).items():
        gens_list = gens if isinstance(gens, list) else [gens]
        if any(callable(g) for g in gens_list):
            out.add(code)
    return out


def _load_reference_value_samples(
    max_per_column: int = 8,
    max_columns_per_code: int = 4,
) -> dict[str, list[dict]]:
    """For each code present in the agent-mediated reference, pull
    actual hive-poc sample values from the reference columns tagged
    with that code.  Returns {code: [{table, column, values}, ...]}.

    These samples ground the agent's authorship in the actual hive-poc
    value distribution rather than (only) the taxonomy definition's
    semantic.  Without them the agent can interpret a definition
    correctly-in-the-abstract but produce values that don't match
    what's actually in hive-poc — e.g., generating international
    postal codes when hive-poc has only US 5-digit ZIPs.
    """
    am_path = Path("build/data/agent_mediated/agent_mediated.json")
    hive_cache_path = Path("build/reflect_nhsvm/hive_cache.json")
    if not am_path.exists() or not hive_cache_path.exists():
        return {}
    try:
        am = json.loads(am_path.read_text())
        hive_cache = json.loads(hive_cache_path.read_text())
    except Exception:  # noqa: BLE001
        return {}

    by_code: dict[str, list[dict]] = {}
    for key, entry in am.items():
        if not isinstance(entry, dict):
            continue
        code = entry.get("code")
        if not code:
            continue
        table, _, column = key.partition(".")
        if not table or not column:
            continue
        tdata = hive_cache.get(table) or {}
        samples = (tdata.get("samples") or {}).get(column) or []
        if not samples:
            continue
        # Truncate per-column samples; cap columns per code so the
        # JSON doesn't bloat for codes with many reference rows
        bucket = by_code.setdefault(code, [])
        if len(bucket) >= max_columns_per_code:
            continue
        bucket.append({
            "table": table,
            "column": column,
            "values": [str(v)[:200] for v in samples[:max_per_column]],
        })
    return by_code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--payloads", type=Path, default=DEFAULT_PAYLOADS,
                    help=f"Enrichment payloads JSON (default: {DEFAULT_PAYLOADS})")
    ap.add_argument("--diversity-threshold", type=float, default=0.5,
                    help="distinct_ratio below this on hand-coded → low_diversity gap")
    ap.add_argument("--require-v1-coverage", action="store_true", default=True,
                    help="SVM-stage default ON: codes with only ICE/template/"
                         "inferred coverage are reclassified as gaps the "
                         "agent must fill (the SVM channel does not train "
                         "on hand-coded ICE; that surface stays upstream "
                         "in Aegir)")
    ap.add_argument("--no-require-v1-coverage", dest="require_v1_coverage",
                    action="store_false",
                    help="Disable v1-coverage requirement (legacy audit mode)")
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

    if args.require_v1_coverage:
        v1_codes = _load_v1_codes()
        log.info("  v1 (agent-authored) coverage: %d codes", len(v1_codes))
    else:
        v1_codes = set()

    log.info("Loading actual hive-poc sample values from agent-mediated reference...")
    ref_value_samples = _load_reference_value_samples()
    log.info("  loaded reference values for %d codes", len(ref_value_samples))

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
        gap_class = classify_gap(
            source, diversity_ratio, args.diversity_threshold,
            has_v1=(code in v1_codes),
            require_v1_coverage=args.require_v1_coverage,
        )
        gap_class_counts[gap_class] += 1
        per_node.append({
            "code": code,
            "mnemonic": mnemonic,
            "label": getattr(cat, "label", "") or "",
            "source": source,
            "has_v1": (code in v1_codes),
            "gap_class": gap_class,
            "probe": probe,
            "payload_present": mnemonic in payloads,
            # Actual hive-poc value samples from the agent-mediated
            # reference rows tagged with this code.  Ground truth the
            # agent must match when authoring generators — without this,
            # the agent can interpret a definition correctly-in-the-
            # abstract but produce values that don't match what's
            # actually in hive-poc (e.g., international postal codes
            # for a US-only BILLPOSTAL column).
            "reference_value_samples": ref_value_samples.get(code, []),
        })

    # Priority-sorted gap list.  Under require_v1_coverage=True the gap
    # classes are *_upstream_needs_v1 variants (codes with upstream
    # Aegir-managed coverage but no v1 SVM-channel generator); legacy
    # classes also handled for backward compat when the flag is
    # disabled.
    PRIORITY = {
        # v1-required mode: every gap is high priority for the agent
        "missing": 0,
        "ice_upstream_needs_v1": 1,
        "template_upstream_needs_v1": 2,
        "inferred_upstream_needs_v1": 3,
        "unknown_upstream_needs_v1": 4,
        # legacy mode
        "inferred_only": 5,
        "template_only": 6,
        "low_diversity_handcoded": 7,
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
