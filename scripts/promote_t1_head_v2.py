#!/usr/bin/env python
"""promote_t1_head_v2.py — synth-primary mix + post-hoc temperature calibration.

Sibling of ``promote_t1_head.py`` (the reference-primary path that
produced c3cf4fce74ee00e6).  This script is the synth-primary
training entry point for the max-accuracy retrain: ALL synth rows
form the training base, reference is added only for codes flagged
``--targeted-reference-codes`` (INOS=0.1 by default plus PASS-1 weak
codes), and a stratified held-out slice fits the softmax temperature
that bakes calibrated mass into the saved adapter's
``training_metadata``.

The reference-primary path remains intact as a rollback option — do
NOT delete or merge into promote_t1_head.py.

Why this exists
---------------

c3cf4fce was trained reference-primary with synth-as-augmentation.
Despite fit_acc=0.9888 and held-out top1=0.7370, its runtime mass
was uncalibrated: top1 mass mean=0.029, max=0.175.  Phase 1 had to
apply post-hoc α_svm=15 to expose the ranking signal — that's
papering over miscalibration rather than fixing it.

This script fixes both problems at once:
  * Mix policy: synth corpus (5,148 rows, 100% code coverage) as the
    training base instead of as a 220-row augmentation, with reference
    added only for codes that need special supervision (INOS + weak
    codes).
  * Calibration: stratified 15% holdout from the training set, fit T
    via NLL minimization on raw logits, bake T into adapter metadata
    so runtime ``NHSVMHeadAdapter._encode_and_predict`` (factorized_nhsvm.py:421)
    picks it up automatically.

Usage
-----

  python scripts/promote_t1_head_v2.py \\
    --mix-policy synth-primary \\
    --targeted-reference-codes 0.1 \\
    --calibration-holdout 0.15 \\
    --no-promote-current        # default; require manual review

After training, the registered head sits in status='building' under
the registry (or saved to disk only if PGlite isn't reachable).  PASS
4 k-fold validation must pass the gate before manual promotion to
'current'.

Promotion gate (must all pass to flip status='current'):
  * held_out_top1 ≥ 0.74              (no accuracy regression vs c3cf4fce)
  * raw_top1_mass_mean ≥ 0.30         (~10× current 0.029)
  * raw_top1_mass_max ≥ 0.70          (~4× current 0.175)
  * ece ≤ 0.20                        (reasonable calibration)
  * held_out_top1_per_fold_std ≤ 0.05 (not fragile)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def _parse_codes_list(s: str) -> list[str]:
    """Parse comma-separated code list, stripping whitespace."""
    if not s or not s.strip():
        return []
    return [c.strip() for c in s.split(",") if c.strip()]


def _stratified_split_by_code(
    labels: list[str],
    holdout_fraction: float,
    seed: int = 42,
    min_train_per_code: int = 1,
) -> tuple[list[int], list[int]]:
    """Stratified split by code: each code's rows split proportionally.

    Returns (train_indices, holdout_indices).  Guarantees each code
    has at least ``min_train_per_code`` row in train (so the head
    sees every code during fitting); the rest go to holdout up to
    ``holdout_fraction``.
    """
    rng = np.random.RandomState(seed)
    by_code: dict[str, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_code[lbl].append(i)
    train_idx, holdout_idx = [], []
    for code, idxs in by_code.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_hold = int(n * holdout_fraction)
        if n - n_hold < min_train_per_code:
            n_hold = max(0, n - min_train_per_code)
        holdout_idx.extend(idxs[:n_hold])
        train_idx.extend(idxs[n_hold:])
    return train_idx, holdout_idx


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path("build/data/svm_training/corpus_v3"),
                    help="Synth corpus directory")
    ap.add_argument("--weights-path", type=Path,
                    default=Path("build/data/agent_mediated/training_weights.json"),
                    help="Audit-derived per-row reference weights")
    ap.add_argument("--mix-policy", default="synth-primary",
                    choices=("synth-primary", "synth-only"),
                    help="Training data mix. synth-primary: all synth + reference "
                         "for --targeted-reference-codes only. synth-only: "
                         "synth corpus exclusively (no reference at all).")
    ap.add_argument("--targeted-reference-codes", default="0.1",
                    help="Comma-separated codes whose reference rows are "
                         "included alongside the full synth corpus. INOS=0.1 "
                         "is the canonical default. Pass a PASS 1 weak-codes "
                         "list joined with commas (e.g. '0.1,1.2.6,1.3.2.1.3').")
    ap.add_argument("--calibration-holdout", type=float, default=0.15,
                    help="Fraction of training data to hold out (stratified "
                         "by code) for softmax temperature fitting. 0 disables "
                         "calibration entirely.")
    ap.add_argument("--taxonomy-id", default="default")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--no-promote-current", action="store_true", default=True,
                    help="Register in status='building' (default). Manual "
                         "promotion via promote_to_current after gate review.")
    ap.add_argument("--promote-current", action="store_true",
                    help="Auto-promote to current after training. NOT "
                         "recommended; requires gate metrics to also pass.")
    ap.add_argument("--no-registry", action="store_true",
                    help="Skip the nhsvm_head_registry write entirely (when "
                         "PGlite is unreachable). Saves adapter + manifest to "
                         "disk; operator runs promote_to_current later.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("promote_t1_head_v2")
    log.info("=" * 76)
    log.info("synth-primary NHSVM retrain — max-accuracy + calibrated mass")
    log.info("=" * 76)

    # Deferred imports so the help text doesn't pay the torch+numpy load
    from atelier.optimize.svm.reflect import (
        build_category_set, build_texts_and_labels,
    )
    from atelier.optimize.svm.reference import (
        load_reference_rows_with_weights, load_synth_rows,
        encode_with_cache, corpus_hash, reference_hash,
    )
    from atelier.optimize.svm.kfold import BEST_KNOB
    from atelier.classify.factorized_nhsvm import (
        fit_factorized_nhsvm, NHSVMHeadAdapter,
    )
    from atelier.optimize.svm.promote import promote_head
    from atelier.optimize.svm.calibrate import (
        fit_softmax_temperature, head_logits_batched,
        mass_metrics_at_T, expected_calibration_error,
    )

    targeted_codes = set(_parse_codes_list(args.targeted_reference_codes))
    log.info("mix-policy: %s", args.mix_policy)
    log.info("targeted reference codes: %s", sorted(targeted_codes) or "(none — pure synth)")
    log.info("calibration holdout: %.0f%% stratified by code",
             100 * args.calibration_holdout)

    t_start = time.time()

    # ── 1. Load + encode synth corpus (the training base) ──────────────
    synth_rows = load_synth_rows(args.corpus_dir)
    synth_labels_all = [r.code for r in synth_rows]
    synth_counts = Counter(synth_labels_all)
    # Drop singletons (codes with only 1 row — can't stratify)
    keep = [i for i, l in enumerate(synth_labels_all) if synth_counts[l] >= 2]
    synth_rows = [synth_rows[i] for i in keep]
    synth_labels = [synth_labels_all[i] for i in keep]
    synth_texts, _ = build_texts_and_labels(synth_rows)
    ch = corpus_hash(args.corpus_dir)
    X_synth = encode_with_cache(
        synth_texts, cache_key=f"synth-only-{ch}",
        refresh=False, batch_size=64,
    )
    log.info("synth corpus: %d rows, %d distinct codes (after singleton filter)",
             len(synth_rows), len(set(synth_labels)))

    # ── 2. Load + encode targeted reference rows ─────────────────────
    if args.mix_policy == "synth-only" or not targeted_codes:
        # No reference at all
        X_ref = np.zeros((0, X_synth.shape[1]), dtype=X_synth.dtype)
        ref_labels: list[str] = []
        ref_weights: np.ndarray = np.zeros(0, dtype=np.float32)
        log.info("reference: SKIPPED (mix-policy=%s, %d targeted codes)",
                 args.mix_policy, len(targeted_codes))
    else:
        all_ref_rows, all_ref_labels, all_ref_weights, _, _ = (
            load_reference_rows_with_weights(weights_path=args.weights_path)
        )
        # Filter to targeted codes only
        mask = [i for i, l in enumerate(all_ref_labels) if l in targeted_codes]
        if not mask:
            log.warning("no reference rows for targeted codes %s — "
                        "training will be synth-only", sorted(targeted_codes))
            X_ref = np.zeros((0, X_synth.shape[1]), dtype=X_synth.dtype)
            ref_labels = []
            ref_weights = np.zeros(0, dtype=np.float32)
        else:
            ref_rows_targeted = [all_ref_rows[i] for i in mask]
            ref_labels = [all_ref_labels[i] for i in mask]
            ref_weights = all_ref_weights[mask]
            ref_texts_targeted, _ = build_texts_and_labels(ref_rows_targeted)
            X_ref = encode_with_cache(
                ref_texts_targeted,
                cache_key=f"ref-targeted-{hashlib.sha1(','.join(sorted(targeted_codes)).encode()).hexdigest()[:8]}",
                refresh=False, batch_size=64,
            )
            log.info("reference (targeted): %d rows across %d codes",
                     len(ref_labels), len(set(ref_labels)))

    # ── 3. Combine + stratified split for calibration ─────────────────
    X_full = np.concatenate([X_synth, X_ref], axis=0) if len(X_ref) else X_synth
    y_full = list(synth_labels) + list(ref_labels)
    # Sample weights: all 1.0 in synth-primary (NOT down-weight synth as
    # the legacy reference-primary path did via training_weights.json)
    w_full = np.ones(len(y_full), dtype=np.float32)
    if len(ref_weights):
        # Optional: respect per-row audit weight on the reference rows
        # (don't penalize low-quality reference rows we explicitly kept)
        w_full[len(synth_labels):] = ref_weights.astype(np.float32)

    log.info("training corpus: %d rows total (%d synth + %d reference)",
             len(y_full), len(synth_labels), len(ref_labels))

    if args.calibration_holdout > 0:
        train_idx, holdout_idx = _stratified_split_by_code(
            y_full, args.calibration_holdout, seed=args.fold_seed,
        )
        log.info("split: %d train + %d holdout (target %.0f%%)",
                 len(train_idx), len(holdout_idx),
                 100 * args.calibration_holdout)
    else:
        train_idx = list(range(len(y_full)))
        holdout_idx = []
        log.info("calibration disabled (--calibration-holdout 0)")

    X_train = X_full[train_idx]
    y_train = [y_full[i] for i in train_idx]
    w_train = w_full[train_idx]
    X_holdout = X_full[holdout_idx] if holdout_idx else None
    y_holdout = [y_full[i] for i in holdout_idx] if holdout_idx else []

    # ── 4. Fit head on training partition ────────────────────────────
    cat_set = build_category_set()
    log.info("fitting NHSVMHead with BEST_KNOB: %s", BEST_KNOB)
    t_fit = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, y_train, cat_set,
        sample_weights=w_train,
        **BEST_KNOB, verbose=True, eval_every=50,
    )
    log.info("trained: fit_acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, time.time() - t_fit)

    # ── 5. Calibration: fit T on stratified holdout ──────────────────
    cal_meta: dict = {}
    fitted_T = 1.0
    if holdout_idx and len(holdout_idx) >= 20:
        log.info("fitting softmax_temperature on %d-row holdout", len(holdout_idx))
        # Compute raw logits
        head_logits = head_logits_batched(head, X_holdout, batch_size=512)
        # Label index in head.codes order
        code_to_idx = {c: i for i, c in enumerate(head.codes)}
        y_holdout_idx = np.array([code_to_idx.get(l, -1) for l in y_holdout])
        valid = y_holdout_idx >= 0
        if valid.sum() < len(y_holdout):
            log.warning("%d holdout labels not in head.codes — skipped from calibration",
                        (~valid).sum())
        cal_result = fit_softmax_temperature(
            head_logits[valid], y_holdout_idx[valid],
        )
        fitted_T = cal_result["best_T"]
        log.info("calibration: T=%.4f  NLL: %.4f → %.4f (Δ=%.4f)  ECE: %.4f → %.4f",
                 fitted_T,
                 cal_result["nll_at_T1"], cal_result["nll_at_best"],
                 cal_result["improvement"],
                 cal_result["ece_at_T1"], cal_result["ece_at_best"])
        cal_meta = cal_result
        # Mass diagnostics at fitted T
        mass_T1 = mass_metrics_at_T(head_logits[valid], y_holdout_idx[valid], 1.0)
        mass_best = mass_metrics_at_T(head_logits[valid], y_holdout_idx[valid], fitted_T)
        log.info("top1 mass at T=1.0:    mean=%.4f max=%.4f acc=%.4f",
                 mass_T1["mean"], mass_T1["max"], mass_T1["accuracy"])
        log.info("top1 mass at T=%.4f:  mean=%.4f max=%.4f acc=%.4f  ← calibrated",
                 fitted_T, mass_best["mean"], mass_best["max"], mass_best["accuracy"])
        cal_meta["mass_at_T1"] = mass_T1
        cal_meta["mass_at_best_T"] = mass_best
    else:
        log.warning("holdout too small (%d) for stable calibration — "
                    "leaving softmax_temperature=1.0", len(holdout_idx))

    # ── 6. Wrap in NHSVMHeadAdapter ──────────────────────────────────
    adapter = NHSVMHeadAdapter(
        head, encoder_id=args.encoder, embed_dim=int(X_train.shape[1]),
        training_metadata={
            "training_mode": "synth-primary",
            "mix_policy": args.mix_policy,
            "targeted_reference_codes": sorted(targeted_codes),
            "fit_acc": float(train_result.final_train_acc),
            "n_train": len(y_train),
            "n_synth": len(synth_labels),
            "n_reference": len(ref_labels),
            "n_calibration_holdout": len(holdout_idx),
            "softmax_temperature": float(fitted_T),
            "calibration": cal_meta,
            "fold_seed": args.fold_seed,
            "best_knob": BEST_KNOB,
            "description": (
                f"Synth-primary head with calibrated softmax_temperature={fitted_T:.4f}. "
                f"Trained on {len(y_train)} rows ({len(synth_labels)} synth + "
                f"{len(ref_labels)} targeted reference for codes "
                f"{sorted(targeted_codes)}). Calibrated against "
                f"{len(holdout_idx)}-row stratified holdout via NLL minimization. "
                f"Compatible with mass_calibration.svm_alpha=1.0 (the Phase 1 "
                f"post-hoc α=15 multiplier is NO LONGER NEEDED with this head)."
            ),
        },
    )

    vocab_sig = hashlib.sha1(
        ",".join(sorted(set(synth_labels) | set(ref_labels))).encode()
    ).hexdigest()[:16]
    ref_hash = reference_hash(args.weights_path) if len(ref_labels) else "synth-only"

    # ── 7. Persist + register (registry write deferred if no DB) ──────
    metrics = {
        "fit_acc": float(train_result.final_train_acc),
        "training_phase": "T1_synth_primary",
        "n_train_total": len(y_train),
        "n_synth": len(synth_labels),
        "n_reference": len(ref_labels),
        "n_calibration_holdout": len(holdout_idx),
        "softmax_temperature_fitted": float(fitted_T),
    }
    if cal_meta:
        metrics["calibration_improvement_nll"] = cal_meta.get("improvement")
        metrics["calibration_ece_at_best"] = cal_meta.get("ece_at_best")
        metrics["holdout_top1_mass_mean"] = cal_meta.get("mass_at_best_T", {}).get("mean")
        metrics["holdout_top1_mass_max"] = cal_meta.get("mass_at_best_T", {}).get("max")

    summary = (
        f"synth-primary head: {len(y_train)} train rows "
        f"({len(synth_labels)} synth + {len(ref_labels)} ref), "
        f"fit_acc={train_result.final_train_acc:.4f}, "
        f"softmax_T={fitted_T:.4f}"
    )

    if args.no_registry:
        # Save adapter + manifest to disk only; skip the registry call.
        # This is the path when PGlite is unreachable.  Operator can
        # register later via:
        #   from atelier.registry.nhsvm_head import register_building
        #   register_building(...)
        head_sig_seed = (
            f"{vocab_sig}|{ref_hash}|{ch}|{args.encoder}|{X_train.shape[1]}|"
            f"synth-primary|{args.fold_seed}|T={fitted_T:.6f}|"
            f"|target={','.join(sorted(targeted_codes))}"
        )
        head_sig = hashlib.sha1(head_sig_seed.encode()).hexdigest()[:16]
        artifact_dir = Path("build/cache/nhsvm") / head_sig
        artifact_dir.mkdir(parents=True, exist_ok=True)
        adapter.save(artifact_dir)
        manifest_path = artifact_dir / "manifest.json"
        manifest_path.write_text(json.dumps({
            "head_sig": head_sig,
            "taxonomy_id": args.taxonomy_id,
            "vocab_sig": vocab_sig,
            "reference_hash": ref_hash,
            "corpus_hash": ch,
            "encoder": args.encoder,
            "embedding_dim": int(X_train.shape[1]),
            "training_mode": "synth-primary",
            "fold_seed": args.fold_seed,
            "metrics": metrics,
            "summary": summary,
            "status": "building",
            "registry_pending": True,
        }, indent=2))
        log.info("(no-registry mode) saved adapter + manifest to %s", artifact_dir)
        log.info("    head_sig: %s", head_sig)
        log.info("    register later via: python -c 'from atelier.registry.nhsvm_head "
                 "import register_building; register_building(...)'")
        print()
        print(f"head_sig:    {head_sig}")
        print(f"artifact:    {artifact_dir}")
        print(f"status:      building (registry pending)")
        print(f"softmax_T:   {fitted_T:.4f}")
        print(f"fit_acc:     {train_result.final_train_acc:.4f}")
        return 0

    # Normal path: register in nhsvm_head_registry
    from atelier.registry.nhsvm_head import promote_to_current
    row = promote_head(
        adapter,
        taxonomy_id=args.taxonomy_id,
        vocab_sig=vocab_sig,
        training_mode="synth-primary",
        reference_hash=ref_hash,
        corpus_hash=ch,
        fold_seed=args.fold_seed,
        augment_floor=0,  # not used in synth-primary
        metrics=metrics,
        summary=summary,
    )
    log.info("promote_head: id=%s head_sig=%s artifact=%s",
             row["id"], row["head_sig"], row["artifact_path"])

    if args.promote_current and not args.no_promote_current:
        promote_to_current(row["id"])
        log.info("promoted to current ✓ (auto-promote)")
    else:
        log.info("registered in status='building' — manual promotion via "
                 "promote_to_current(id=%s) after gate review", row["id"])

    print()
    print(f"head_id:     {row['id']}")
    print(f"head_sig:    {row['head_sig']}")
    print(f"artifact:    {row['artifact_path']}")
    print(f"status:      {'current' if args.promote_current else 'building'}")
    print(f"softmax_T:   {fitted_T:.4f}")
    print(f"fit_acc:     {train_result.final_train_acc:.4f}")
    print(f"\ntotal wall: {time.time() - t_start:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
