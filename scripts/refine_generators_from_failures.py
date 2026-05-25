#!/usr/bin/env python3
"""scripts/refine_generators_from_failures.py — Phase E of the corpus
expansion plan.

Multi-pass iterative domain-adaptation loop.  Each pass:

  1. Reads the latest ``build/reflect_nhsvm_eval_shap_v2/results.json``
     to identify the bottom-N categories by held-out test accuracy
     and, for each, the neighbor categories the model predicted
     instead (the adjacent value-shape regions that need separation).
  2. Writes ``build/svm_corpus_v2/refinement_targets.json`` listing the
     target codes + neighbors.  Consumed by the evolve-generators skill
     in domain-adaptation mode (via the same wrapper as Phase B).
  3. Re-runs ``scripts/run_evolve_generators_sdk.py`` against the
     refinement targets (only — Phase B's other generators are
     preserved).
  4. Re-runs ``scripts/generate_corpus_v2.py`` (incremental — corpus_v2
     is regenerated fresh, but with the updated generators_v1.py).
  5. Re-runs ``scripts/reflect_nhsvm_eval_shap_v2.py`` to measure lift.

Loop stops on:
  - Test top-1 ≥ ``--target-accuracy`` (default 0.80), OR
  - Two consecutive passes with lift < 1pp, OR
  - ``--max-passes`` reached (default 5).

Outputs per-pass history at ``build/svm_corpus_v2/refinement_history.json``.

Usage:
  python scripts/refine_generators_from_failures.py
  python scripts/refine_generators_from_failures.py --max-passes 3
  python scripts/refine_generators_from_failures.py --target-accuracy 0.85
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("refine_generators_from_failures")

EVAL_RESULTS = Path("build/reflect_nhsvm_eval_shap_v2/results.json")
PER_CAT_JSON = Path("build/reflect_nhsvm_eval_shap_v2/per_category_accuracy.json")
REFINEMENT_TARGETS = Path("build/svm_corpus_v2/refinement_targets.json")
HISTORY_JSON = Path("build/svm_corpus_v2/refinement_history.json")


def identify_refinement_targets(top_n: int = 20) -> list[dict]:
    """Read latest eval results; identify bottom-N categories with their
    most-frequent prediction-neighbors (categories the model predicted
    instead — the adjacent value-shape regions the generator needs to
    move away from via domain adaptation)."""
    if not EVAL_RESULTS.exists():
        raise FileNotFoundError(
            f"Eval results missing at {EVAL_RESULTS}. "
            f"Run scripts/reflect_nhsvm_eval_shap_v2.py first."
        )
    results = json.loads(EVAL_RESULTS.read_text())
    per_cat = results.get("per_category_accuracy", {})

    # Sort by accuracy ascending; pick bottom_n with n_test >= 1
    sorted_cats = sorted(
        per_cat.items(),
        key=lambda kv: (kv[1]["accuracy"], -kv[1]["n_test"]),
    )
    bottom = [c for c, _info in sorted_cats[:top_n]]

    # Prediction-neighbors from per-row predictions
    # (categories the model picked instead of the true label — these are
    # the adjacent value-shape regions the refined generator must
    # separate from via domain adaptation)
    neighbors: dict[str, Counter] = defaultdict(Counter)
    if PER_CAT_JSON.exists():
        pc = json.loads(PER_CAT_JSON.read_text())
        for rec in pc.get("predictions", []):
            if not rec.get("correct"):
                neighbors[rec["true"]][rec["pred"]] += 1

    targets = []
    for code in bottom:
        neighbor_list = [
            {"pred_code": p, "count": n}
            for p, n in neighbors.get(code, Counter()).most_common(3)
        ]
        targets.append({
            "code": code,
            "accuracy": per_cat[code]["accuracy"],
            "n_test": per_cat[code]["n_test"],
            "n_correct": per_cat[code]["n_correct"],
            "neighbors": neighbor_list,
        })
    return targets


def write_refinement_targets(targets: list[dict]) -> None:
    REFINEMENT_TARGETS.parent.mkdir(parents=True, exist_ok=True)
    REFINEMENT_TARGETS.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(targets),
        "targets": targets,
        "instructions": (
            "These are the bottom-N codes from the latest held-out eval. "
            "The evolve-generators skill should re-author generators for "
            "each target code under DOMAIN-ADAPTATION mode: produce values "
            "that occupy a clearly separable value-shape region from the "
            "listed `neighbors` (the categories the model wrongly predicted "
            "for this code's test rows).  Re-author replaces existing "
            "variants in generators_v1.py."
        ),
    }, indent=2))


def run_command(cmd: list[str]) -> int:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def load_history() -> list[dict]:
    if not HISTORY_JSON.exists():
        return []
    return json.loads(HISTORY_JSON.read_text()).get("passes", [])


def save_history(passes: list[dict]) -> None:
    HISTORY_JSON.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_JSON.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "passes": passes,
    }, indent=2))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-passes", type=int, default=5)
    ap.add_argument("--target-accuracy", type=float, default=0.80)
    ap.add_argument("--bottom-n", type=int, default=20,
                    help="Per-pass: refine bottom-N worst-accuracy categories")
    ap.add_argument("--rows-per-node", type=int, default=200,
                    help="Passed to generate_corpus_v2.py on each rerun")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    passes = load_history()
    pass_idx = len(passes)
    consecutive_low_lift = 0

    while pass_idx < args.max_passes:
        log.info("\n=== Refinement pass %d ===", pass_idx + 1)

        # Read latest eval result
        if not EVAL_RESULTS.exists():
            log.error("No eval results to refine from. "
                      "Run reflect_nhsvm_eval_shap_v2.py first.")
            return 2
        prior = json.loads(EVAL_RESULTS.read_text())
        prior_top1 = prior["test"]["top1"]
        log.info("  prior test top-1: %.4f", prior_top1)

        if prior_top1 >= args.target_accuracy:
            log.info("✓ target accuracy met (%.4f ≥ %.2f); stopping",
                     prior_top1, args.target_accuracy)
            break

        # Identify targets
        targets = identify_refinement_targets(top_n=args.bottom_n)
        write_refinement_targets(targets)
        log.info("  refinement targets written: %d codes → %s",
                 len(targets), REFINEMENT_TARGETS)

        # Clear stale proposal files for target codes so the agent re-authors
        # them (it skips codes with existing proposals).
        proposals_dir = Path("build/svm_corpus_v2/proposals")
        n_cleared = 0
        for t in targets:
            sanitized = t["code"].replace(".", "_")
            stale = proposals_dir / f"{sanitized}.json"
            if stale.exists():
                stale.unlink()
                n_cleared += 1
        log.info("  cleared %d stale proposal files for re-authorship", n_cleared)

        # Re-run Phase B (Agent SDK re-authors the cleared codes; the skill
        # will pick up build/svm_corpus_v2/refinement_targets.json for the
        # domain-adaptation neighbors context)
        rc = run_command([sys.executable,
                          "scripts/run_evolve_generators_sdk.py"])
        if rc != 0:
            log.error("evolve_generators_sdk failed (rc=%d); aborting", rc)
            return rc

        # Re-run Phase C (corpus regeneration with updated generators_v1.py)
        rc = run_command([sys.executable,
                          "scripts/generate_corpus_v2.py",
                          "--rows-per-node", str(args.rows_per_node)])
        if rc != 0:
            log.error("generate_corpus_v2 failed (rc=%d); aborting", rc)
            return rc

        # Re-run Phase D (eval)
        rc = run_command([sys.executable,
                          "scripts/reflect_nhsvm_eval_shap_v2.py"])
        if rc != 0:
            log.error("reflect_nhsvm_eval_shap_v2 failed (rc=%d); aborting", rc)
            return rc

        # Compare
        post = json.loads(EVAL_RESULTS.read_text())
        post_top1 = post["test"]["top1"]
        lift = post_top1 - prior_top1
        log.info("  pass %d: prior=%.4f  post=%.4f  lift=%+.4f",
                 pass_idx + 1, prior_top1, post_top1, lift)

        passes.append({
            "pass_idx": pass_idx + 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior_top1": prior_top1,
            "post_top1": post_top1,
            "lift": round(lift, 4),
            "n_refinement_targets": len(targets),
            "target_codes": [t["code"] for t in targets],
        })
        save_history(passes)

        if lift < 0.01:
            consecutive_low_lift += 1
            log.info("  lift < 1pp (consecutive low-lift count: %d)",
                     consecutive_low_lift)
            if consecutive_low_lift >= 2:
                log.info("✗ stopped: 2 consecutive passes with lift < 1pp "
                         "(diminishing returns)")
                break
        else:
            consecutive_low_lift = 0

        pass_idx += 1

    log.info("\n=== Refinement complete ===")
    if passes:
        final = passes[-1]
        log.info("Total passes: %d", len(passes))
        log.info("Final test top-1: %.4f", final["post_top1"])
        if passes:
            cumulative = passes[-1]["post_top1"] - passes[0]["prior_top1"]
            log.info("Cumulative lift over baseline: %+.4f", cumulative)
    return 0


if __name__ == "__main__":
    sys.exit(main())
