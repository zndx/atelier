#!/usr/bin/env python3
"""scripts/refine_generators_from_failures.py — Phase E under the new framing.

Multi-pass iterative loop that drives off the VALIDATE accuracy signal
(full agent-mediated reference, not a held-out subset) and the
metrology-derived diagnostic feedback (fidelity / separability / spread
percentiles + shape exemplars from real reference rows).

Per pass:

  1. Read latest Phase D's ``results_pass{N-1}.json`` (or ``results.json``
     symlink) to identify per-code validate accuracy.
  2. Run ``scripts/corpus_metrology.py`` against the current corpus to
     produce ``metrology_report.json`` keyed by code with the diagnostic
     fields the agent needs.
  3. Identify bottom-N refinement targets:
     - filter by ``n_validate >= 3`` (fall back to >=2 if pool < 10)
     - exclude codes in permanent abandon_set
     - exclude codes in cooldown (non-improvers from recent passes)
     - join with metrology signals + shape exemplars
  4. Write ``refinement_targets.json`` with the rich schema the agent
     consumes.
  5. Re-author generators for target codes via Agent SDK.
  6. Regenerate corpus.
  7. Re-run Phase D with ``--pass-idx N`` so outputs are pass-numbered.
  8. Compare post vs prior on full_validate top-1; update cooldown for
     non-improvers; track best pass.

Loop stops on:
  - full-validate top-1 ≥ ``--target-accuracy`` (default 0.95, the
    SAME number as the test-stage deployment threshold), OR
  - Two consecutive passes with lift < 1pp, OR
  - ``--max-passes`` reached (default 5).

Outputs:
  build/svm_corpus_v2/refinement_history.json — per-pass deltas
  build/svm_corpus_v2/refinement_targets.json — current pass's targets
  build/svm_corpus_v2/abandon_history.json    — passes_with_problem accumulator
  build/svm_corpus_v2/best_pass.json          — best_pass_idx + best_top1

Usage:
  python scripts/refine_generators_from_failures.py
  python scripts/refine_generators_from_failures.py --max-passes 3
  python scripts/refine_generators_from_failures.py --target-accuracy 0.70
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

REPORT_DIR = Path("build/reflect_nhsvm_eval_shap_v2")
EVAL_RESULTS = REPORT_DIR / "results.json"
PER_CAT_JSON = REPORT_DIR / "per_category_accuracy.json"

SVM_DIR = Path("build/svm_corpus_v2")
REFINEMENT_TARGETS = SVM_DIR / "refinement_targets.json"
HISTORY_JSON = SVM_DIR / "refinement_history.json"
ABANDON_HISTORY = SVM_DIR / "abandon_history.json"
ABANDON_SET = SVM_DIR / "abandoned_codes.json"
COOLDOWN_STATE = SVM_DIR / "cooldown_state.json"
BEST_PASS = SVM_DIR / "best_pass.json"
METROLOGY_REPORT = SVM_DIR / "metrology_report.json"

PROPOSALS_DIR = SVM_DIR / "proposals"
SNAPSHOTS_DIR = SVM_DIR / "corpus_snapshots"

COOLDOWN_PASSES = 2  # Non-improvers skipped for this many passes


def snapshot_corpus(corpus_dir: Path, pass_idx: int) -> Path:
    """Snapshot the corpus's synth_rows.jsonl + manifest.json to a
    pass-numbered directory so the orchestrator can restore the best
    pass's corpus state on stop.
    """
    snap_dir = SNAPSHOTS_DIR / f"pass_{pass_idx}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for fname in ("synth_rows.jsonl", "manifest.json"):
        src = corpus_dir / fname
        if src.exists():
            shutil.copy2(src, snap_dir / fname)
    return snap_dir


# ──────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────

def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def _save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


# ──────────────────────────────────────────────────────────────────────
# Run helper
# ──────────────────────────────────────────────────────────────────────

def run_command(cmd: list[str]) -> int:
    log.info("$ %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


# ──────────────────────────────────────────────────────────────────────
# Target identification (metrology-driven)
# ──────────────────────────────────────────────────────────────────────

def identify_refinement_targets(
    *,
    bottom_n: int,
    n_validate_min: int,
    abandoned_codes: set[str],
    cooldown_state: dict[str, int],
    metrology: dict,
) -> tuple[list[dict], dict]:
    """Pick bottom-N codes from current Phase D results, joined with
    metrology signals.  Returns (targets_list, diagnostics).
    """
    if not EVAL_RESULTS.exists():
        raise FileNotFoundError(
            f"Eval results missing at {EVAL_RESULTS}. "
            f"Run scripts/reflect_nhsvm_eval_shap_v2.py first."
        )
    results = json.loads(EVAL_RESULTS.read_text())
    per_cat = results.get("per_category_accuracy", {})

    metrology_per_code = metrology.get("per_code", {}) if metrology else {}

    # Filter codes:
    eligible: list[tuple[str, dict]] = []
    skipped_low_n = 0
    skipped_abandoned = 0
    skipped_cooldown = 0
    for code, info in per_cat.items():
        n_val = info.get("n_test", 0)  # field name is per_category_accuracy's; semantically n_validate
        if code in abandoned_codes:
            skipped_abandoned += 1
            continue
        if cooldown_state.get(code, 0) > 0:
            skipped_cooldown += 1
            continue
        if n_val < n_validate_min:
            skipped_low_n += 1
            continue
        eligible.append((code, info))

    diagnostics = {
        "n_eligible_pre_sort": len(eligible),
        "skipped_low_n_validate": skipped_low_n,
        "skipped_abandoned": skipped_abandoned,
        "skipped_cooldown": skipped_cooldown,
    }

    # Fallback: if pool is too small, relax n_validate filter
    if len(eligible) < 10 and n_validate_min > 2:
        log.warning("  pool < 10 after n_validate>=%d filter; "
                    "relaxing to n_validate>=2", n_validate_min)
        eligible = []
        for code, info in per_cat.items():
            n_val = info.get("n_test", 0)
            if code in abandoned_codes:
                continue
            if cooldown_state.get(code, 0) > 0:
                continue
            if n_val < 2:
                continue
            eligible.append((code, info))
        diagnostics["fallback_n_validate_relaxed"] = True

    # Sort by accuracy ascending; pick bottom_n
    eligible.sort(key=lambda kv: (kv[1]["accuracy"], -kv[1]["n_test"]))
    selected = eligible[:bottom_n]

    # Prediction-neighbors from per-row predictions
    neighbors: dict[str, Counter] = defaultdict(Counter)
    if PER_CAT_JSON.exists():
        pc = json.loads(PER_CAT_JSON.read_text())
        for rec in pc.get("predictions", []):
            if not rec.get("correct"):
                neighbors[rec["true"]][rec["pred"]] += 1

    targets: list[dict] = []
    for code, info in selected:
        mt = metrology_per_code.get(code, {})

        # Build neighbor list with separability if metrology has it
        sep = mt.get("separability", {})  # {pred_code: distance}
        neighbor_list = []
        for p, n in neighbors.get(code, Counter()).most_common(5):
            neighbor_list.append({
                "pred_code": p,
                "count": n,
                "separability": sep.get(p),
            })

        targets.append({
            "code": code,
            "validate_accuracy": info["accuracy"],
            "n_validate": info["n_test"],
            "n_correct": info["n_correct"],
            "neighbors": neighbor_list,
            # Metrology fields
            "fidelity_vs_reference": mt.get("fidelity_vs_reference"),
            "fidelity_pct": mt.get("fidelity_pct"),
            "separability_min": mt.get("separability_min"),
            "separability_pct": mt.get("separability_pct"),
            "intra_code_mean_sim": mt.get("intra_code_mean_sim"),
            "spread_pct": mt.get("spread_pct"),
            "shape_exemplars_from_reference":
                mt.get("shape_exemplars_from_reference", []),
            "neighbor_source": mt.get("neighbor_source"),
            "passes_with_problem": mt.get("passes_with_problem", 0),
            "recommended_action": mt.get("recommended_action", "improve_separability"),
        })

    return targets, diagnostics


def write_refinement_targets(targets: list[dict], diagnostics: dict) -> None:
    REFINEMENT_TARGETS.parent.mkdir(parents=True, exist_ok=True)
    REFINEMENT_TARGETS.write_text(json.dumps({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_targets": len(targets),
        "diagnostics": diagnostics,
        "targets": targets,
        "instructions": (
            "These are the bottom-N codes from the latest validate eval, "
            "joined with metrology signals.  For each code, the "
            "`recommended_action` tells you what to optimize:\n"
            "  - improve_fidelity: your synth shape doesn't match the "
            "    `shape_exemplars_from_reference`; re-author for surface "
            "    similarity (format, character class, length distribution)\n"
            "  - improve_separability: your synth is on-shape but "
            "    indistinguishable from the listed `neighbors`; amplify "
            "    discriminative dimensions visible in their exemplars\n"
            "  - reduce_redundancy: generator has saturated coverage; "
            "    recommend lower volume, not more variants\n"
            "  - abandon_structural: this code requires non-value channels "
            "    (name/table-context/cosine); skip authorship, log a note\n"
            "The encoder sees only what fits in lean text — column_name, "
            "type, ~5 sample values, sibling names.  Optimize for that "
            "small window."
        ),
    }, indent=2))


# ──────────────────────────────────────────────────────────────────────
# Cooldown + abandon bookkeeping
# ──────────────────────────────────────────────────────────────────────

def update_cooldown(
    state: dict[str, int],
    selected_codes: set[str],
    improved_codes: set[str],
) -> dict[str, int]:
    """Decrement all cooldown counters by 1 each pass.  For selected codes
    that did NOT improve, set cooldown to COOLDOWN_PASSES.  Improved codes
    get cooldown cleared.
    """
    new_state: dict[str, int] = {}
    # Decrement existing
    for code, remaining in state.items():
        if remaining > 1:
            new_state[code] = remaining - 1
    # Apply cooldown to non-improvers among selected
    for code in selected_codes:
        if code not in improved_codes:
            new_state[code] = COOLDOWN_PASSES
    return new_state


def update_abandon_history(
    abandon_history: dict[str, int],
    metrology: dict,
) -> dict[str, int]:
    """Increment passes_with_problem for codes whose metrology shows
    bad_fidelity OR bad_separability; reset others.  Used to drive
    metrology's abandon_structural recommendation on the NEXT pass.
    """
    per_code = metrology.get("per_code", {})
    thresholds = metrology.get("thresholds", {})
    fid_bad_pct = thresholds.get("fidelity_bad_percentile", 90)
    sep_bad_pct = thresholds.get("separability_bad_percentile", 10)

    new_history: dict[str, int] = {}
    for code, entry in per_code.items():
        fid_pct = entry.get("fidelity_pct")
        sep_pct = entry.get("separability_pct")
        bad = ((fid_pct is not None and fid_pct >= fid_bad_pct)
               or (sep_pct is not None and sep_pct <= sep_bad_pct))
        if bad:
            new_history[code] = abandon_history.get(code, 0) + 1
    return new_history


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--max-passes", type=int, default=5)
    ap.add_argument("--target-accuracy", type=float, default=0.95,
                    help="Stop when full-validate top-1 reaches this "
                         "(SAME as the test-stage deployment threshold — "
                         "validate is the upper-bound proxy for test "
                         "accuracy since the reference is sampled from "
                         "hive-poc; plateauing below this means the SVM "
                         "channel is NOT deployment-ready)")
    ap.add_argument("--bottom-n", type=int, default=20,
                    help="Per-pass: refine bottom-N worst-accuracy codes")
    ap.add_argument("--n-validate-min", type=int, default=3,
                    help="Skip codes with fewer than this many validate "
                         "examples (avoids singleton-noise feedback loop)")
    ap.add_argument("--synth-examples-per-code", type=int, default=40,
                    help="Passed to generate_corpus_v2.py on each rerun")
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path("build/data/svm_training/corpus_v2"),
                    help="Target corpus directory")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    history_data = _load_json(HISTORY_JSON, {"passes": []})
    passes = history_data.get("passes", [])
    pass_idx = len(passes)

    abandon_history = _load_json(ABANDON_HISTORY, {})
    abandoned_codes: set[str] = set(_load_json(ABANDON_SET, []))
    cooldown_state = _load_json(COOLDOWN_STATE, {})
    best_pass = _load_json(BEST_PASS, {
        "best_pass_idx": None, "best_top1": -1.0, "best_corpus_hash": None,
    })

    consecutive_low_lift = 0

    while pass_idx < args.max_passes:
        log.info("\n=== Refinement pass %d ===", pass_idx + 1)

        # Read latest Phase D
        if not EVAL_RESULTS.exists():
            log.error("No eval results to refine from. "
                      "Run reflect_nhsvm_eval_shap_v2.py first.")
            return 2
        prior = json.loads(EVAL_RESULTS.read_text())
        # Support both new schema (validate.full_top1) and legacy (test.top1)
        if "validate" in prior:
            prior_top1 = prior["validate"]["full_top1"]
            prior_cont = prior["validate"].get("continuity_top1")
        else:
            prior_top1 = prior["test"]["top1"]
            prior_cont = None
        log.info("  prior full-validate top-1: %.4f", prior_top1)
        if prior_cont is not None:
            log.info("  prior continuity-271 top-1: %.4f", prior_cont)

        if prior_top1 >= args.target_accuracy:
            log.info("✓ target accuracy met (%.4f ≥ %.2f); stopping",
                     prior_top1, args.target_accuracy)
            break

        # Compute metrology for the current corpus + predictions
        log.info("  computing metrology...")
        _save_json(ABANDON_HISTORY, abandon_history)  # ensure latest on disk
        rc = run_command([
            sys.executable, "scripts/corpus_metrology.py",
            "--corpus", str(args.corpus_dir),
            "--output", str(METROLOGY_REPORT),
            "--per-category-json", str(PER_CAT_JSON),
            "--abandon-history", str(ABANDON_HISTORY),
        ])
        if rc != 0:
            log.error("corpus_metrology failed (rc=%d); aborting", rc)
            return rc
        metrology = _load_json(METROLOGY_REPORT, {})

        # Promote metrology-flagged abandon_structural codes to permanent set
        new_abandons = 0
        for code, entry in metrology.get("per_code", {}).items():
            if entry.get("recommended_action") == "abandon_structural":
                if code not in abandoned_codes:
                    abandoned_codes.add(code)
                    new_abandons += 1
        if new_abandons:
            log.warning("  abandoned %d new structurally-trapped codes "
                        "(metrology flagged abandon_structural after %d passes)",
                        new_abandons, COOLDOWN_PASSES)
            _save_json(ABANDON_SET, sorted(abandoned_codes))

        # Identify targets
        targets, diagnostics = identify_refinement_targets(
            bottom_n=args.bottom_n,
            n_validate_min=args.n_validate_min,
            abandoned_codes=abandoned_codes,
            cooldown_state=cooldown_state,
            metrology=metrology,
        )
        write_refinement_targets(targets, diagnostics)
        log.info("  refinement targets: %d codes (eligible %d, "
                 "skipped low-n=%d, abandoned=%d, cooldown=%d)",
                 len(targets), diagnostics["n_eligible_pre_sort"],
                 diagnostics["skipped_low_n_validate"],
                 diagnostics["skipped_abandoned"],
                 diagnostics["skipped_cooldown"])

        if not targets:
            log.warning("✗ no eligible refinement targets remain; stopping")
            break

        target_codes = {t["code"] for t in targets}

        # Clear stale proposal files for target codes (so agent re-authors)
        n_cleared = 0
        for t in targets:
            sanitized = t["code"].replace(".", "_")
            stale = PROPOSALS_DIR / f"{sanitized}.json"
            if stale.exists():
                stale.unlink()
                n_cleared += 1
        log.info("  cleared %d stale proposal files for re-authorship", n_cleared)

        # Re-run Phase B (Agent SDK)
        rc = run_command([
            sys.executable, "scripts/run_evolve_generators_sdk.py",
        ])
        if rc != 0:
            log.error("evolve_generators_sdk failed (rc=%d); aborting", rc)
            return rc

        # Re-run Phase C (corpus regeneration)
        rc = run_command([
            sys.executable, "scripts/generate_corpus_v2.py",
            "--synth-examples-per-code", str(args.synth_examples_per_code),
            "--output", str(args.corpus_dir),
        ])
        if rc != 0:
            log.error("generate_corpus_v2 failed (rc=%d); aborting", rc)
            return rc

        # Re-run Phase D with pass-numbered output
        new_pass_idx = pass_idx + 1
        rc = run_command([
            sys.executable, "scripts/reflect_nhsvm_eval_shap_v2.py",
            "--pass-idx", str(new_pass_idx),
            "--corpus-dir", str(args.corpus_dir),
        ])
        if rc != 0:
            log.error("reflect_nhsvm_eval_shap_v2 failed (rc=%d); aborting", rc)
            return rc

        # Snapshot this pass's corpus state for potential best-pass restore
        snap = snapshot_corpus(args.corpus_dir, new_pass_idx)
        log.info("  corpus snapshot: %s", snap)

        # Compare
        post = json.loads(EVAL_RESULTS.read_text())
        if "validate" in post:
            post_top1 = post["validate"]["full_top1"]
            post_cont = post["validate"].get("continuity_top1")
        else:
            post_top1 = post["test"]["top1"]
            post_cont = None
        lift = post_top1 - prior_top1
        log.info("  pass %d: prior=%.4f  post=%.4f  lift=%+.4f",
                 new_pass_idx, prior_top1, post_top1, lift)
        if post_cont is not None:
            log.info("  continuity-271 also: %.4f", post_cont)

        # Per-code improvement detection
        post_per_cat = post.get("per_category_accuracy", {})
        prior_per_cat = prior.get("per_category_accuracy", {})
        improved_codes: set[str] = set()
        for code in target_codes:
            new_acc = post_per_cat.get(code, {}).get("accuracy", 0.0)
            old_acc = prior_per_cat.get(code, {}).get("accuracy", 0.0)
            if new_acc > old_acc:
                improved_codes.add(code)
        log.info("  per-code improvements among targets: %d / %d",
                 len(improved_codes), len(target_codes))

        # Update cooldown state
        cooldown_state = update_cooldown(
            cooldown_state, target_codes, improved_codes,
        )
        _save_json(COOLDOWN_STATE, cooldown_state)

        # Update abandon-history accumulator for next pass's metrology
        abandon_history = update_abandon_history(abandon_history, metrology)
        _save_json(ABANDON_HISTORY, abandon_history)

        # Update best-pass tracking
        if post_top1 > best_pass.get("best_top1", -1.0):
            best_pass = {
                "best_pass_idx": new_pass_idx,
                "best_top1": round(post_top1, 4),
                "best_continuity_top1": (round(post_cont, 4)
                                          if post_cont is not None else None),
                "best_corpus_hash": post.get("corpus_hash"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            _save_json(BEST_PASS, best_pass)
            log.info("  ★ new best pass: %d  top-1=%.4f",
                     new_pass_idx, post_top1)

        # Record this pass
        passes.append({
            "pass_idx": new_pass_idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prior_full_top1": round(prior_top1, 4),
            "post_full_top1": round(post_top1, 4),
            "post_continuity_top1": (round(post_cont, 4)
                                      if post_cont is not None else None),
            "lift": round(lift, 4),
            "n_refinement_targets": len(targets),
            "target_codes": sorted(target_codes),
            "n_improved": len(improved_codes),
            "improved_codes": sorted(improved_codes),
            "n_abandoned_total": len(abandoned_codes),
            "n_new_abandoned_this_pass": new_abandons,
            "n_cooldown_active": len(cooldown_state),
        })
        _save_json(HISTORY_JSON, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "passes": passes,
        })

        if lift < 0.01:
            consecutive_low_lift += 1
            log.info("  lift < 1pp (consecutive low-lift count: %d)",
                     consecutive_low_lift)
            if consecutive_low_lift >= 2:
                log.info("✗ stopped: 2 consecutive passes with lift < 1pp "
                         "(diminishing returns); best pass was #%d at %.4f",
                         best_pass.get("best_pass_idx"),
                         best_pass.get("best_top1"))
                break
        else:
            consecutive_low_lift = 0

        pass_idx += 1

    log.info("\n=== Refinement complete ===")
    if passes:
        log.info("Total passes: %d", len(passes))
        log.info("Final pass top-1: %.4f", passes[-1]["post_full_top1"])
        log.info("BEST pass: #%d at top-1=%.4f (corpus_hash=%s)",
                 best_pass.get("best_pass_idx"),
                 best_pass.get("best_top1"),
                 best_pass.get("best_corpus_hash"))
        cumulative = (passes[-1]["post_full_top1"]
                       - passes[0]["prior_full_top1"])
        log.info("Cumulative lift vs first prior: %+.4f", cumulative)
    log.info("Abandoned codes (permanent): %d", len(abandoned_codes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
