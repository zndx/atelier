#!/usr/bin/env python3
"""scripts/reflect_nhsvm_eval_shap_v2.py — Phase D under train/validate/test framing.

Trains the factorized NHSVM head on the synthetic corpus alone (no
reference data in training); validates against the full agent-mediated
reference set.  The 271-example stratified slice (seed=42) is reported
as a named subset of validate for continuity with the historical 0.6125
baseline, but the load-bearing number is full-validate top-1.

The reference is NOT held out for "test" — the test surface is hive-poc
at large, evaluated separately by ``scripts/svm_target_health.py`` with
the best-pass model produced by the refinement loop.

Outputs (per pass when --pass-idx N is supplied):
  build/reflect_nhsvm_eval_shap_v2/report.md
  build/reflect_nhsvm_eval_shap_v2/results_pass{N}.json
  build/reflect_nhsvm_eval_shap_v2/per_category_accuracy_pass{N}.json
  build/reflect_nhsvm_eval_shap_v2/run.log
  build/reflect_nhsvm_eval_shap_v2/embeddings_cache.npz
  build/reflect_nhsvm_eval_shap_v2/results.json  (symlink to latest pass)

Usage:
  python scripts/reflect_nhsvm_eval_shap_v2.py
  python scripts/reflect_nhsvm_eval_shap_v2.py --pass-idx 3
  python scripts/reflect_nhsvm_eval_shap_v2.py --refresh-embeddings
  python scripts/reflect_nhsvm_eval_shap_v2.py --corpus-dir build/data/svm_training/corpus_v3
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from reflect_nhsvm import (
    Row,
    load_rows,
    build_texts_and_labels,
    build_category_set,
    characterize_failures,
    AGENT_MEDIATED,
)
from reflect_nhsvm_modernbert import (
    encode_modernbert,
    MODEL_ID,
    EMB_DIM,
)
from reflect_nhsvm_eval_shap import (
    stratified_split,
    stratified_k_fold,
    BEST_KNOB,
)
from atelier.classify.factorized_nhsvm import fit_factorized_nhsvm

log = logging.getLogger("reflect_nhsvm_eval_shap_v2")

# Data-loading + encoding helpers migrated to atelier core.  Re-exported
# here so existing consumers (svm_cosine_uplift_gate, svm_target_health,
# corpus_metrology) continue to import from this module unchanged.
from atelier.optimize.svm.reference import (  # noqa: E402
    DEFAULT_CORPUS_DIR,
    EMBED_CACHE,
    REPORT_DIR,
    corpus_hash,
    encode_with_cache,
    load_reference_rows_with_weights,
    load_synth_rows,
    reference_hash,
)

REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = REPORT_DIR / "results.json"
REPORT_MD = REPORT_DIR / "report.md"
PER_CAT_JSON = REPORT_DIR / "per_category_accuracy.json"
RUN_LOG = REPORT_DIR / "run.log"


# ──────────────────────────────────────────────────────────────────────
# Per-category eval breakdown
# ──────────────────────────────────────────────────────────────────────

def per_category_accuracy(
    pred: list[str], true: list[str],
) -> dict[str, dict]:
    """Per-class accuracy + n_test."""
    by_class: dict[str, list[bool]] = defaultdict(list)
    for p, t in zip(pred, true):
        by_class[t].append(p == t)
    result = {}
    for cls, hits in by_class.items():
        result[cls] = {
            "n_test": len(hits),
            "n_correct": sum(hits),
            "accuracy": round(sum(hits) / len(hits), 4) if hits else 0.0,
        }
    return result


# ──────────────────────────────────────────────────────────────────────
# Phase D core
# ──────────────────────────────────────────────────────────────────────

def _run_eval_synth_only(
    *,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    refresh_embeddings: bool = False,
    batch_size: int = 64,
    knob: dict | None = None,
    pass_idx: int | None = None,
) -> dict:
    """Phase D under the train/validate/test framing — synth-only branch.

    Trains on synth-only; evaluates on the full agent-mediated reference
    set (validate).  Reports two top-1 numbers: full-validate (primary
    refinement signal) and the 271-example seed=42 stratified slice
    (continuity with the historical 0.6125 baseline).

    Backwards-compatibility: this is the verbatim original body of
    ``run_eval``.  Its result schema (no ``training_mode`` key) is the
    back-compat marker — downstream render/print code treats missing
    ``training_mode`` as synth-only.
    """
    knob = knob or BEST_KNOB
    pass_label = f"pass {pass_idx}" if pass_idx is not None else "single run"
    log.info("=== Phase D (%s): synth-only train → full-reference validate ===",
             pass_label)

    log.info("Loading reference rows (full validate dataset)...")
    real_rows = load_rows(refresh_cache=False, database="reference_corpus")
    _, real_labels = build_texts_and_labels(real_rows)

    # Drop singletons — needed for the continuity-slice stratified split
    # to be well-defined.  Has no effect on synth training.
    code_counts = Counter(real_labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    val_mask = [l not in singletons for l in real_labels]
    validate_rows = [r for r, m in zip(real_rows, val_mask) if m]
    validate_labels = [l for l, m in zip(real_labels, val_mask) if m]
    log.info("  validate (full reference): %d rows  (%d singletons dropped)",
             len(validate_rows), len(singletons))

    # Compute the historical 271-example stratified slice (seed=42) within
    # validate — used only as a named subset for continuity reporting.
    # Train/test naming reflects the historical seed-42 80/20 partition; we
    # take the held-out 20% as the continuity slice.
    _, continuity_idx = stratified_split(
        validate_labels, test_size=0.2, seed=42,
    )
    continuity_set = set(continuity_idx)
    log.info("  continuity slice (seed=42, top-1 historical anchor): %d rows",
             len(continuity_idx))

    # Load synth corpus (training set)
    log.info("Loading synth corpus from %s...", corpus_dir)
    synth_rows = load_synth_rows(corpus_dir)
    synth_labels = [r.code for r in synth_rows]

    # Drop synth classes with < 2 examples (NHSVM constraint)
    synth_counts = Counter(synth_labels)
    synth_keep = [i for i, l in enumerate(synth_labels) if synth_counts[l] >= 2]
    train_rows = [synth_rows[i] for i in synth_keep]
    train_labels = [synth_labels[i] for i in synth_keep]
    dropped = len(synth_rows) - len(train_rows)
    if dropped:
        log.info("  dropped %d synth rows with singleton classes", dropped)
    log.info("  train (synth-only): %d rows  %d distinct codes",
             len(train_rows), len(set(train_labels)))

    # Build texts (lean svm-text shape)
    train_texts, _ = build_texts_and_labels(train_rows)
    validate_texts, _ = build_texts_and_labels(validate_rows)

    # Encode (cached)
    log.info("Encoding synth-only train texts with ModernBERT...")
    ch = corpus_hash(corpus_dir)
    X_train = encode_with_cache(
        train_texts, cache_key=f"synth-only-{ch}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("Encoding full-reference validate texts with ModernBERT...")
    X_validate = encode_with_cache(
        validate_texts, cache_key="full-reference",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  X_train shape=%s  X_validate shape=%s",
             X_train.shape, X_validate.shape)

    # Train factorized head
    log.info("Training factorized NHSVM head...")
    cat_set = build_category_set()
    train_t0 = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, train_labels, cat_set,
        **knob, verbose=True, eval_every=50,
    )
    train_elapsed = time.time() - train_t0
    log.info("  train fit-acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, train_elapsed)

    # Predict on full validate
    log.info("Predicting on full-reference validate...")
    from sklearn.preprocessing import normalize as sk_normalize
    X_val_norm = sk_normalize(X_validate, norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    pred_codes = head.predict_codes(
        torch.tensor(X_val_norm, device=device),
    )

    # Two reported numbers: full-validate + continuity-271
    full_correct = sum(1 for p, t in zip(pred_codes, validate_labels) if p == t)
    full_top1 = full_correct / len(validate_labels) if validate_labels else 0.0

    cont_correct = 0
    cont_total = 0
    for i, (p, t) in enumerate(zip(pred_codes, validate_labels)):
        if i in continuity_set:
            cont_total += 1
            if p == t:
                cont_correct += 1
    cont_top1 = cont_correct / cont_total if cont_total else 0.0

    log.info("  === VALIDATE NUMBERS (%s) ===", pass_label)
    log.info("    full-reference top-1: %.4f (%d/%d)",
             full_top1, full_correct, len(validate_labels))
    log.info("    continuity-271 slice top-1: %.4f (%d/%d)",
             cont_top1, cont_correct, cont_total)

    # Per-category accuracy on FULL validate (more stable than 271-only)
    per_cat = per_category_accuracy(pred_codes, validate_labels)

    # Failure characterization
    failures = characterize_failures(
        validate_rows, pred_codes, validate_labels, cat_set,
    )

    return {
        "pass_idx": pass_idx,
        "config": dict(knob),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_train.shape[1]),
        "corpus_hash": ch,
        "corpus_dir": str(corpus_dir),
        "split": {
            "n_synth_train": len(train_rows),
            "n_synth_train_classes": len(set(train_labels)),
            "n_validate": len(validate_labels),
            "n_validate_classes": len(set(validate_labels)),
            "n_continuity_slice": cont_total,
            "continuity_seed": 42,
        },
        "train": {
            "fit_acc": round(train_result.final_train_acc, 4),
            "final_loss": round(train_result.final_train_loss, 4),
            "elapsed_sec": round(train_elapsed, 1),
            "epochs": train_result.epochs_run,
        },
        "validate": {
            "full_top1": round(full_top1, 4),
            "full_n_correct": full_correct,
            "full_n": len(validate_labels),
            "continuity_top1": round(cont_top1, 4),
            "continuity_n_correct": cont_correct,
            "continuity_n": cont_total,
        },
        "per_category_accuracy": per_cat,
        "failures": {
            "kinds": failures["failure_kinds"],
            "examples_top": failures["examples_top"][:30],
        },
        "predictions": [
            {"key": f"{r.table}.{r.column}", "true": t, "pred": p,
             "correct": p == t,
             "in_continuity_slice": i in continuity_set}
            for i, (r, t, p) in enumerate(zip(validate_rows, validate_labels, pred_codes))
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Reference-primary k-fold training (Phase D protocol switch)
# ──────────────────────────────────────────────────────────────────────

def _train_synth_only_diagnostic(
    X_synth: np.ndarray,
    synth_labels: list[str],
    X_ref: np.ndarray,
    ref_labels: list[str],
    *,
    cat_set,
    knob: dict,
) -> dict:
    """Train a head on synth-only, predict on full reference.

    Returns a diagnostic block usable as ``validate.synth_only_diagnostic``.
    The diagnostic answers: 'how would synth-only training have scored
    today on this same reference?' — generalization signal that runs in
    parallel to k-fold so we never lose sight of the synth-only baseline
    while training reference-primary.
    """
    from sklearn.preprocessing import normalize as sk_normalize
    t0 = time.time()
    head_diag, result_diag = fit_factorized_nhsvm(
        X_synth, synth_labels, cat_set,
        **knob, verbose=False, eval_every=50,
    )
    X_ref_norm = sk_normalize(X_ref, norm="l2", axis=1).astype(np.float32)
    device = next(head_diag.parameters()).device
    pred = head_diag.predict_codes(torch.tensor(X_ref_norm, device=device))
    correct = sum(1 for p, t in zip(pred, ref_labels) if p == t)
    elapsed = time.time() - t0
    n = len(ref_labels)
    top1 = correct / n if n else 0.0
    log.info("  synth-only diagnostic: top-1=%.4f (%d/%d) fit=%.4f elapsed=%.1fs",
             top1, correct, n, result_diag.final_train_acc, elapsed)
    return {
        "full_top1": round(top1, 4),
        "n_correct": correct,
        "n": n,
        "fit_acc": round(result_diag.final_train_acc, 4),
        "elapsed_sec": round(elapsed, 1),
    }


def _select_synth_augmentation(
    *,
    train_counts: Counter,
    synth_by_code: dict[str, list[int]],
    augment_floor: int,
    rng: np.random.RandomState,
) -> tuple[list[int], list[str]]:
    """Pick synth row indices to top up under-represented codes in fold-train.

    For each code whose fold-train count is below ``augment_floor``, draws
    ``floor - count`` synth rows from that code's synth pool (without
    replacement if enough are available, otherwise uses all).  Codes with
    no synth coverage are silently skipped — no fix available at this
    layer; surface them in the generator-coverage audit instead.
    """
    aug_idx: list[int] = []
    aug_labels: list[str] = []
    for code in sorted(train_counts):
        if train_counts[code] >= augment_floor:
            continue
        need = augment_floor - train_counts[code]
        pool = synth_by_code.get(code, [])
        if not pool:
            continue
        if len(pool) >= need:
            chosen = list(rng.choice(pool, size=need, replace=False))
        else:
            chosen = list(pool)
        aug_idx.extend(chosen)
        aug_labels.extend(code for _ in chosen)
    return aug_idx, aug_labels


def _run_eval_reference_primary(
    *,
    corpus_dir: Path,
    refresh_embeddings: bool,
    batch_size: int,
    knob: dict,
    pass_idx: int | None,
    n_folds: int,
    fold_seed: int,
    weights_path: Path,
    augment_floor: int,
    also_report_synth_only: bool,
) -> dict:
    """K-fold reference-primary training: each fold trains on 80% of the
    reference (+ synth-augmentation for under-represented codes) and
    predicts the held-out 20%.

    The union of fold-vals is the full reference (every non-singleton row
    appears in exactly one fold's val partition), so ``validate.full_top1``
    is a true held-out number computed across the entire reference.
    Per-fold mean ± std quantifies the variance of that number.
    """
    pass_label = f"pass {pass_idx}" if pass_idx is not None else "single run"
    log.info("=== Phase D (%s, mode=reference-primary): %d-fold "
             "ref-primary train → held-out ref validate ===",
             pass_label, n_folds)

    ref_rows, ref_labels, ref_weights, coverage_table, audit_summary = (
        load_reference_rows_with_weights(weights_path=weights_path)
    )
    ref_texts, _ = build_texts_and_labels(ref_rows)
    log.info("Encoding full reference texts with ModernBERT...")
    X_ref = encode_with_cache(
        ref_texts, cache_key="full-reference",
        refresh=refresh_embeddings, batch_size=batch_size,
    )

    log.info("Loading synth corpus from %s (for fold augmentation)...", corpus_dir)
    synth_rows = load_synth_rows(corpus_dir)
    synth_labels_all = [r.code for r in synth_rows]
    synth_counts = Counter(synth_labels_all)
    synth_keep = [i for i, l in enumerate(synth_labels_all) if synth_counts[l] >= 2]
    synth_rows = [synth_rows[i] for i in synth_keep]
    synth_labels = [synth_labels_all[i] for i in synth_keep]
    if len(synth_keep) < len(synth_labels_all):
        log.info("  dropped %d singleton-class synth rows",
                 len(synth_labels_all) - len(synth_keep))
    synth_texts, _ = build_texts_and_labels(synth_rows)
    log.info("Encoding synth corpus with ModernBERT...")
    ch = corpus_hash(corpus_dir)
    X_synth = encode_with_cache(
        synth_texts, cache_key=f"synth-only-{ch}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    synth_by_code: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(synth_labels):
        synth_by_code[l].append(i)

    log.info("Building %d-fold stratified split (seed=%d)...", n_folds, fold_seed)
    folds = stratified_k_fold(ref_labels, n_folds=n_folds, seed=fold_seed)

    cat_set = build_category_set()
    rng = np.random.RandomState(fold_seed)
    from sklearn.preprocessing import normalize as sk_normalize

    per_fold_results: list[dict] = []
    union_predictions: dict[int, tuple[str, str, int]] = {}
    n_train_real_per_fold: list[int] = []
    n_train_synth_aug_per_fold: list[int] = []
    n_codes_aug_per_fold: list[int] = []
    fit_acc_per_fold: list[float] = []
    fold_train_elapsed_total = 0.0

    for fold_idx, (fold_train_idx, fold_val_idx) in enumerate(folds):
        log.info("--- Fold %d/%d: %d train, %d val ---",
                 fold_idx + 1, n_folds, len(fold_train_idx), len(fold_val_idx))

        X_fold_real = X_ref[fold_train_idx]
        y_fold_real = [ref_labels[i] for i in fold_train_idx]
        w_fold_real = ref_weights[fold_train_idx]

        train_counts = Counter(y_fold_real)
        aug_idx, aug_labels = _select_synth_augmentation(
            train_counts=train_counts,
            synth_by_code=synth_by_code,
            augment_floor=augment_floor,
            rng=rng,
        )
        n_codes_aug = len({c for c in aug_labels})

        if aug_idx:
            X_aug = X_synth[aug_idx]
            X_fold = np.concatenate([X_fold_real, X_aug], axis=0)
            y_fold = list(y_fold_real) + aug_labels
            w_fold = np.concatenate(
                [w_fold_real, np.ones(len(aug_idx), dtype=np.float32)]
            )
        else:
            X_fold = X_fold_real
            y_fold = list(y_fold_real)
            w_fold = w_fold_real

        n_train_real_per_fold.append(len(fold_train_idx))
        n_train_synth_aug_per_fold.append(len(aug_idx))
        n_codes_aug_per_fold.append(n_codes_aug)

        train_t0 = time.time()
        head_k, train_result_k = fit_factorized_nhsvm(
            X_fold, y_fold, cat_set,
            sample_weights=w_fold,
            **knob, verbose=False, eval_every=50,
        )
        fold_elapsed = time.time() - train_t0
        fold_train_elapsed_total += fold_elapsed
        fit_acc_per_fold.append(train_result_k.final_train_acc)

        X_val_norm = sk_normalize(
            X_ref[fold_val_idx], norm="l2", axis=1
        ).astype(np.float32)
        device = next(head_k.parameters()).device
        pred_codes_k = head_k.predict_codes(torch.tensor(X_val_norm, device=device))
        fold_true = [ref_labels[i] for i in fold_val_idx]
        fold_correct = sum(1 for p, t in zip(pred_codes_k, fold_true) if p == t)
        fold_top1 = fold_correct / len(fold_val_idx) if fold_val_idx else 0.0

        per_fold_results.append({
            "fold": fold_idx + 1,
            "n_train": len(y_fold),
            "n_train_real": len(fold_train_idx),
            "n_train_synth_aug": len(aug_idx),
            "n_codes_aug": n_codes_aug,
            "n_val": len(fold_val_idx),
            "fit_acc": round(train_result_k.final_train_acc, 4),
            "val_top1": round(fold_top1, 4),
            "val_correct": fold_correct,
            "elapsed_sec": round(fold_elapsed, 1),
        })
        log.info(
            "  fold %d: train n=%d (real=%d, synth_aug=%d over %d codes)  "
            "fit=%.4f  val_top1=%.4f (%d/%d)  elapsed=%.1fs",
            fold_idx + 1, len(y_fold), len(fold_train_idx), len(aug_idx),
            n_codes_aug, train_result_k.final_train_acc,
            fold_top1, fold_correct, len(fold_val_idx), fold_elapsed,
        )

        for ref_i, pred_code in zip(fold_val_idx, pred_codes_k):
            union_predictions[ref_i] = (ref_labels[ref_i], pred_code, fold_idx + 1)

    fit_acc_arr = np.asarray(fit_acc_per_fold)
    val_top1_arr = np.asarray([r["val_top1"] for r in per_fold_results])
    union_correct = sum(1 for true, pred, _ in union_predictions.values() if true == pred)
    union_n = len(union_predictions)
    full_top1 = union_correct / union_n if union_n else 0.0

    log.info("=== K-fold aggregate ===")
    log.info("  union top-1 (held-out, full reference): %.4f (%d/%d)",
             full_top1, union_correct, union_n)
    log.info("  per-fold val_top1: mean=%.4f std=%.4f  fit=%.4f±%.4f",
             val_top1_arr.mean(), val_top1_arr.std(),
             fit_acc_arr.mean(), fit_acc_arr.std())

    sorted_ref_idx = sorted(union_predictions)
    union_true = [union_predictions[i][0] for i in sorted_ref_idx]
    union_pred = [union_predictions[i][1] for i in sorted_ref_idx]
    union_rows = [ref_rows[i] for i in sorted_ref_idx]
    union_folds = [union_predictions[i][2] for i in sorted_ref_idx]
    per_cat = per_category_accuracy(union_pred, union_true)

    diagnostic_block = None
    if also_report_synth_only:
        log.info("Running synth-only diagnostic alongside reference-primary...")
        diagnostic_block = _train_synth_only_diagnostic(
            X_synth, synth_labels, X_ref, ref_labels,
            cat_set=cat_set, knob=knob,
        )

    failures = characterize_failures(union_rows, union_pred, union_true, cat_set)

    return {
        "pass_idx": pass_idx,
        "training_mode": "reference-primary",
        "n_folds": n_folds,
        "fold_seed": fold_seed,
        "augment_floor": augment_floor,
        "weights_path": str(weights_path),
        "audit_summary": audit_summary,
        "config": dict(knob),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_ref.shape[1]),
        "corpus_hash": ch,
        "corpus_dir": str(corpus_dir),
        "reference_hash": reference_hash(weights_path),
        "split": {
            "n_reference_total": len(ref_rows),
            "n_reference_classes": len(set(ref_labels)),
            "n_synth_pool": len(synth_rows),
            "n_synth_pool_classes": len(set(synth_labels)),
            "n_folds": n_folds,
            "n_train_real_per_fold": n_train_real_per_fold,
            "n_train_synth_aug_per_fold": n_train_synth_aug_per_fold,
            "n_codes_aug_per_fold": n_codes_aug_per_fold,
        },
        "train": {
            "fit_acc_mean": round(float(fit_acc_arr.mean()), 4),
            "fit_acc_std": round(float(fit_acc_arr.std()), 4),
            "fit_acc_per_fold": [round(x, 4) for x in fit_acc_per_fold],
            "elapsed_sec_total": round(fold_train_elapsed_total, 1),
        },
        "validate": {
            "full_top1": round(full_top1, 4),
            "full_n_correct": union_correct,
            "full_n": union_n,
            "full_top1_mean": round(float(val_top1_arr.mean()), 4),
            "full_top1_std": round(float(val_top1_arr.std()), 4),
            "full_top1_per_fold": [round(x, 4) for x in val_top1_arr.tolist()],
            "synth_only_diagnostic": diagnostic_block,
        },
        "per_fold": per_fold_results,
        "per_category_accuracy": per_cat,
        "failures": {
            "kinds": failures["failure_kinds"],
            "examples_top": failures["examples_top"][:30],
        },
        "predictions": [
            {
                "key": f"{r.table}.{r.column}",
                "true": t,
                "pred": p,
                "correct": p == t,
                "fold": f,
            }
            for r, t, p, f in zip(union_rows, union_true, union_pred, union_folds)
        ],
    }


def _run_eval_reference_plus_synth(
    *,
    corpus_dir: Path,
    refresh_embeddings: bool,
    batch_size: int,
    knob: dict,
    pass_idx: int | None,
    fold_seed: int,
    weights_path: Path,
    augment_floor: int,
    also_report_synth_only: bool,
) -> dict:
    """Single 80/20 stratified split of the reference + full synth corpus
    appended to train.  Faster than k-fold (one head trained), no variance
    estimate.  Useful for quick experimentation between full k-fold runs.

    ``augment_floor`` is accepted but unused under this mode — full synth
    is unconditionally appended; per-code top-up is meaningful only in the
    k-fold path where train counts vary across folds.
    """
    del augment_floor  # explicitly unused in this mode
    pass_label = f"pass {pass_idx}" if pass_idx is not None else "single run"
    log.info("=== Phase D (%s, mode=reference+synth): single 80/20 split + full synth ===",
             pass_label)

    ref_rows, ref_labels, ref_weights, coverage_table, audit_summary = (
        load_reference_rows_with_weights(weights_path=weights_path)
    )
    ref_texts, _ = build_texts_and_labels(ref_rows)
    log.info("Encoding full reference texts with ModernBERT...")
    X_ref = encode_with_cache(
        ref_texts, cache_key="full-reference",
        refresh=refresh_embeddings, batch_size=batch_size,
    )

    log.info("Loading synth corpus from %s...", corpus_dir)
    synth_rows = load_synth_rows(corpus_dir)
    synth_labels_all = [r.code for r in synth_rows]
    synth_counts = Counter(synth_labels_all)
    synth_keep = [i for i, l in enumerate(synth_labels_all) if synth_counts[l] >= 2]
    synth_rows = [synth_rows[i] for i in synth_keep]
    synth_labels = [synth_labels_all[i] for i in synth_keep]
    synth_texts, _ = build_texts_and_labels(synth_rows)
    log.info("Encoding synth corpus with ModernBERT...")
    ch = corpus_hash(corpus_dir)
    X_synth = encode_with_cache(
        synth_texts, cache_key=f"synth-only-{ch}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )

    train_idx, val_idx = stratified_split(ref_labels, test_size=0.2, seed=fold_seed)
    log.info("  split: %d train + %d synth = %d, %d val",
             len(train_idx), len(synth_rows), len(train_idx) + len(synth_rows),
             len(val_idx))

    X_train = np.concatenate([X_ref[train_idx], X_synth], axis=0)
    y_train = [ref_labels[i] for i in train_idx] + synth_labels
    w_train = np.concatenate([
        ref_weights[train_idx],
        np.ones(len(synth_labels), dtype=np.float32),
    ])

    cat_set = build_category_set()
    train_t0 = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, y_train, cat_set,
        sample_weights=w_train,
        **knob, verbose=True, eval_every=50,
    )
    train_elapsed = time.time() - train_t0
    log.info("  train fit-acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, train_elapsed)

    from sklearn.preprocessing import normalize as sk_normalize
    X_val_norm = sk_normalize(
        X_ref[val_idx], norm="l2", axis=1
    ).astype(np.float32)
    device = next(head.parameters()).device
    pred = head.predict_codes(torch.tensor(X_val_norm, device=device))
    true = [ref_labels[i] for i in val_idx]
    val_rows = [ref_rows[i] for i in val_idx]
    correct = sum(1 for p, t in zip(pred, true) if p == t)
    top1 = correct / len(true) if true else 0.0
    log.info("  validate held-out 20%%: top-1=%.4f (%d/%d)",
             top1, correct, len(true))

    per_cat = per_category_accuracy(pred, true)

    diagnostic_block = None
    if also_report_synth_only:
        log.info("Running synth-only diagnostic alongside reference+synth...")
        diagnostic_block = _train_synth_only_diagnostic(
            X_synth, synth_labels, X_ref, ref_labels,
            cat_set=cat_set, knob=knob,
        )

    failures = characterize_failures(val_rows, pred, true, cat_set)

    return {
        "pass_idx": pass_idx,
        "training_mode": "reference+synth",
        "fold_seed": fold_seed,
        "weights_path": str(weights_path),
        "audit_summary": audit_summary,
        "config": dict(knob),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_ref.shape[1]),
        "corpus_hash": ch,
        "corpus_dir": str(corpus_dir),
        "reference_hash": reference_hash(weights_path),
        "split": {
            "n_reference_total": len(ref_rows),
            "n_reference_classes": len(set(ref_labels)),
            "n_synth_pool": len(synth_rows),
            "n_train_real": len(train_idx),
            "n_train_synth": len(synth_labels),
            "n_train_total": len(y_train),
            "n_val": len(val_idx),
            "split_seed": fold_seed,
        },
        "train": {
            "fit_acc": round(train_result.final_train_acc, 4),
            "final_loss": round(train_result.final_train_loss, 4),
            "elapsed_sec": round(train_elapsed, 1),
            "epochs": train_result.epochs_run,
        },
        "validate": {
            "full_top1": round(top1, 4),
            "full_n_correct": correct,
            "full_n": len(true),
            "synth_only_diagnostic": diagnostic_block,
        },
        "per_category_accuracy": per_cat,
        "failures": {
            "kinds": failures["failure_kinds"],
            "examples_top": failures["examples_top"][:30],
        },
        "predictions": [
            {"key": f"{r.table}.{r.column}", "true": t, "pred": p,
             "correct": p == t}
            for r, t, p in zip(val_rows, true, pred)
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Phase D dispatcher
# ──────────────────────────────────────────────────────────────────────

VALID_TRAINING_MODES = ("synth-only", "reference-primary", "reference+synth")


def run_eval(
    *,
    training_mode: str = "synth-only",
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    refresh_embeddings: bool = False,
    batch_size: int = 64,
    knob: dict | None = None,
    pass_idx: int | None = None,
    n_folds: int = 5,
    fold_seed: int = 42,
    weights_path: Path = Path("build/data/agent_mediated/training_weights.json"),
    augment_floor: int = 5,
    also_report_synth_only: bool = False,
) -> dict:
    """Phase D dispatcher — routes to the per-training-mode evaluator.

    Modes:
      ``synth-only``       — historical default; train on synth corpus,
                             validate on full reference.  Back-compat:
                             result schema has no ``training_mode`` key.
      ``reference-primary`` — k-fold over reference; each fold trains 80%
                             of reference + synth-augmentation, validates
                             the held-out 20%.  Union of fold-vals = full
                             reference, headline ``validate.full_top1`` is
                             true held-out.
      ``reference+synth``   — single 80/20 split of reference + full synth
                             corpus appended to train.  Faster, no fold
                             variance.

    ``weights_path``, ``augment_floor``, ``n_folds``, ``fold_seed``,
    ``also_report_synth_only`` are only consulted by the non-synth-only
    modes.  Keeping them as defaultable kwargs preserves the existing
    synth-only call site signature.
    """
    knob = knob or BEST_KNOB
    if training_mode == "synth-only":
        return _run_eval_synth_only(
            corpus_dir=corpus_dir,
            refresh_embeddings=refresh_embeddings,
            batch_size=batch_size,
            knob=knob,
            pass_idx=pass_idx,
        )
    if training_mode == "reference-primary":
        return _run_eval_reference_primary(
            corpus_dir=corpus_dir,
            refresh_embeddings=refresh_embeddings,
            batch_size=batch_size,
            knob=knob,
            pass_idx=pass_idx,
            n_folds=n_folds,
            fold_seed=fold_seed,
            weights_path=weights_path,
            augment_floor=augment_floor,
            also_report_synth_only=also_report_synth_only,
        )
    if training_mode == "reference+synth":
        return _run_eval_reference_plus_synth(
            corpus_dir=corpus_dir,
            refresh_embeddings=refresh_embeddings,
            batch_size=batch_size,
            knob=knob,
            pass_idx=pass_idx,
            fold_seed=fold_seed,
            weights_path=weights_path,
            augment_floor=augment_floor,
            also_report_synth_only=also_report_synth_only,
        )
    raise ValueError(
        f"unknown training_mode={training_mode!r}; valid: {VALID_TRAINING_MODES}"
    )


# ──────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────

BASELINE_ANCHORS = {
    "tfidf_kronecker_fit_on_train_top1_kept": 0.9893,
    "modernbert_kronecker_fit_on_train_top1_kept": 0.0426,
    "modernbert_factorized_fit_on_train_top1_kept": 0.9938,
    # Different training regime (855 reference-train + 0 synth, evaluated
    # on 271 reference-test).  Preserved as historical context only.
    "historical_modernbert_factorized_held_out_271_real_only_train": 0.6125,
}


def _baseline_per_category() -> dict[str, float]:
    """Load the baseline (pre-corpus_v2) per-category accuracy from
    build/reflect_nhsvm_eval_shap/per_row_attribution.json if present, so
    the report can show per-category lift.  Falls back to empty dict."""
    p = Path("build/reflect_nhsvm_eval_shap/per_row_attribution.json")
    if not p.exists():
        return {}
    try:
        records = json.loads(p.read_text())
        by_class: dict[str, list[bool]] = defaultdict(list)
        for rec in records:
            by_class[rec["true"]].append(rec["correct"])
        return {cls: sum(hits) / len(hits) for cls, hits in by_class.items()}
    except Exception:
        return {}


def render_report(result: dict) -> str:
    """Dispatch report rendering by training_mode.

    Back-compat: a result without ``training_mode`` is treated as
    synth-only (since the old run_eval body never set that key).
    """
    mode = result.get("training_mode", "synth-only")
    if mode == "reference-primary":
        return _render_report_reference_primary(result)
    if mode == "reference+synth":
        return _render_report_reference_plus_synth(result)
    return _render_report_synth_only(result)


def _render_report_synth_only(result: dict) -> str:
    lines: list[str] = []
    pass_label = (f" — pass {result['pass_idx']}"
                  if result.get("pass_idx") is not None else "")
    lines.append(f"# Factorized NHSVM eval{pass_label} — synth-only train, full-reference validate")
    lines.append("")
    s = result["split"]
    t = result["train"]
    v = result["validate"]
    lines.append(
        f"Training set: **{s['n_synth_train']} synthetic examples** across "
        f"{s['n_synth_train_classes']} codes (synth corpus from {result['corpus_dir']}).  "
        f"Validate: **{s['n_validate']} reference examples** across "
        f"{s['n_validate_classes']} codes (full agent-mediated reference, "
        f"no held-out test slice — test is hive-poc-at-large via svm_target_health.py).  "
        f"Encoder: {result['encoder']}."
    )
    lines.append("")

    lines.append("## Headline (two validate numbers)")
    lines.append("")
    lines.append(
        f"- **Full-reference VALIDATE top-1: {v['full_top1']:.4f} "
        f"({v['full_n_correct']}/{v['full_n']})** — primary refinement signal"
    )
    lines.append(
        f"- **Continuity-271 slice VALIDATE top-1: {v['continuity_top1']:.4f} "
        f"({v['continuity_n_correct']}/{v['continuity_n']})** — seed=42 "
        f"stratified, comparable to the historical 0.6125 anchor in regime"
    )
    lines.append(f"- Train fit-acc: {t['fit_acc']:.4f}")
    lines.append(f"- Train time: {t['elapsed_sec']:.1f}s "
                 f"({t['epochs']} epochs, loss={t['final_loss']:.4f})")
    lines.append("")
    lines.append(
        "Note: the historical 0.6125 baseline used a DIFFERENT training "
        "regime (855 reference-train + 0 synth, evaluated on 271 reference-"
        "test).  The continuity-271 number here is on the SAME 271 examples "
        "but with synth-only training, so it is regime-comparable in "
        "evaluation surface but not in training data.  Both numbers should "
        "rise together as the corpus improves; if continuity-271 lags full-"
        "validate substantially, that signals the 271 slice over-represents "
        "structurally-hard codes."
    )
    lines.append("")

    lines.append("## Comparison anchors")
    lines.append("")
    lines.append("| Source | regime | top-1 |")
    lines.append("|---|---|---:|")
    lines.append("| reflect_nhsvm.py | TF-IDF + Kronecker, fit-on-train | 0.9893 on kept |")
    lines.append("| reflect_nhsvm_modernbert.py | ModernBERT + Kronecker, fit-on-train | 0.0426 (collapsed) |")
    lines.append("| reflect_nhsvm_factorized.py | ModernBERT + Factorized, fit-on-train | 0.9938 on kept |")
    lines.append("| historical Phase A | ModernBERT + Factorized, 271-test, real-only train | 0.6125 |")
    lines.append(f"| **this report (full-validate)** | ModernBERT + Factorized, full-reference validate, synth-only train | **{v['full_top1']:.4f}** |")
    lines.append(f"| **this report (continuity-271)** | ModernBERT + Factorized, 271-slice of validate, synth-only train | **{v['continuity_top1']:.4f}** |")
    lines.append("")

    # Per-category lift
    baseline_per_cat = _baseline_per_category()
    if baseline_per_cat:
        lines.append("## Per-category lift vs Phase A baseline (top 20 / top 10)")
        lines.append("")
        deltas = []
        for cls, info in result["per_category_accuracy"].items():
            base_acc = baseline_per_cat.get(cls, 0.0)
            new_acc = info["accuracy"]
            deltas.append((cls, base_acc, new_acc, new_acc - base_acc, info["n_test"]))
        deltas_imp = sorted([d for d in deltas if d[3] > 0],
                             key=lambda x: -x[3])[:20]
        deltas_reg = sorted([d for d in deltas if d[3] < 0],
                             key=lambda x: x[3])[:10]
        lines.append("### Improvements")
        lines.append("")
        lines.append("| code | baseline | this pass | Δ | n_validate |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, b, n, d, nt in deltas_imp:
            lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | +{d:.3f} | {nt} |")
        lines.append("")
        if deltas_reg:
            lines.append("### Regressions")
            lines.append("")
            lines.append("| code | baseline | this pass | Δ | n_validate |")
            lines.append("|---|---:|---:|---:|---:|")
            for cls, b, n, d, nt in deltas_reg:
                lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | {d:.3f} | {nt} |")
            lines.append("")

    # Failure breakdown
    lines.append("## Failure-mode breakdown (full validate)")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, val in sorted(result["failures"]["kinds"].items(),
                          key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {val} |")
    lines.append("")

    lines.append("## Success criterion check (against plan thresholds)")
    lines.append("")
    full_top1 = v["full_top1"]
    if full_top1 >= 0.65:
        lines.append(
            f"✓ **Stretch target MET**: full-validate top-1 = {full_top1:.4f} ≥ 0.65. "
            f"Metrology-driven agent feedback produced net-positive iteration."
        )
    elif full_top1 >= 0.62:
        lines.append(
            f"~ **Primary criterion MET**: full-validate top-1 = {full_top1:.4f} ≥ 0.62. "
            f"Volume-cap + metrology demonstrate that volume wasn't the constraint; "
            f"refinement loop has room to push toward 0.65 stretch."
        )
    elif full_top1 >= 0.60:
        lines.append(
            f"~ **Approaching target**: full-validate top-1 = {full_top1:.4f} in 0.60-0.62 zone. "
            f"Refinement loop should improve from here; if it plateaus, the next lever is "
            f"structural (pulling A_HD/A_ID/ENOS from SVM target, merging BILL/SHIP at parent)."
        )
    else:
        lines.append(
            f"✗ **Below 0.60**: full-validate top-1 = {full_top1:.4f}. "
            f"Likely indicates the corpus alone can't lift past structural limits. "
            f"Inspect per-code metrology trajectories for `abandon_structural` flags; "
            f"those codes need DST channel restructuring, not more synth."
        )
    lines.append("")
    return "\n".join(lines)


def _render_report_reference_primary(result: dict) -> str:
    lines: list[str] = []
    pass_label = (f" — pass {result['pass_idx']}"
                  if result.get("pass_idx") is not None else "")
    lines.append(
        f"# Factorized NHSVM eval{pass_label} — reference-primary "
        f"({result['n_folds']}-fold), synth-augmented"
    )
    lines.append("")
    s = result["split"]
    t = result["train"]
    v = result["validate"]
    lines.append(
        f"Reference: **{s['n_reference_total']} rows** across "
        f"{s['n_reference_classes']} codes (full agent-mediated reference, "
        f"weights from `{result['weights_path']}`).  Synth pool: "
        f"**{s['n_synth_pool']} rows** across {s['n_synth_pool_classes']} "
        f"codes from {result['corpus_dir']}.  K-fold: {result['n_folds']} "
        f"(seed={result['fold_seed']}).  Encoder: {result['encoder']}."
    )
    lines.append("")

    lines.append("## Headline (held-out across all folds)")
    lines.append("")
    lines.append(
        f"- **Full-reference held-out top-1: {v['full_top1']:.4f} "
        f"({v['full_n_correct']}/{v['full_n']})** — union of all fold-vals "
        f"(every row predicted by a model that did not see it during training)"
    )
    lines.append(
        f"- Per-fold val_top1: mean={v['full_top1_mean']:.4f}  "
        f"std={v['full_top1_std']:.4f}  "
        f"folds={v['full_top1_per_fold']}"
    )
    lines.append(
        f"- Train fit-acc: mean={t['fit_acc_mean']:.4f}  "
        f"std={t['fit_acc_std']:.4f}  folds={t['fit_acc_per_fold']}"
    )
    lines.append(f"- Total train time: {t['elapsed_sec_total']:.1f}s "
                 f"across {result['n_folds']} folds")
    diag = v.get("synth_only_diagnostic")
    if diag is not None:
        lines.append(
            f"- Synth-only diagnostic (parallel): {diag['full_top1']:.4f} "
            f"({diag['n_correct']}/{diag['n']}) — "
            f"generalization signal under the prior synth-only protocol; "
            f"large gap to held-out top-1 confirms the protocol switch "
            f"was load-bearing"
        )
    lines.append("")

    lines.append("## Per-fold detail")
    lines.append("")
    lines.append("| fold | train (real+aug) | aug codes | val | fit-acc | val top-1 | elapsed |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in result["per_fold"]:
        lines.append(
            f"| {r['fold']} | {r['n_train']} ({r['n_train_real']}+"
            f"{r['n_train_synth_aug']}) | {r['n_codes_aug']} | "
            f"{r['n_val']} | {r['fit_acc']:.4f} | {r['val_top1']:.4f} "
            f"({r['val_correct']}/{r['n_val']}) | {r['elapsed_sec']:.1f}s |"
        )
    lines.append("")

    # Per-category lift vs Phase A baseline (if available)
    baseline_per_cat = _baseline_per_category()
    if baseline_per_cat:
        lines.append("## Per-category lift vs Phase A baseline (top 20 / top 10)")
        lines.append("")
        deltas = []
        for cls, info in result["per_category_accuracy"].items():
            base_acc = baseline_per_cat.get(cls, 0.0)
            new_acc = info["accuracy"]
            deltas.append((cls, base_acc, new_acc, new_acc - base_acc, info["n_test"]))
        deltas_imp = sorted([d for d in deltas if d[3] > 0],
                            key=lambda x: -x[3])[:20]
        deltas_reg = sorted([d for d in deltas if d[3] < 0],
                            key=lambda x: x[3])[:10]
        lines.append("### Improvements")
        lines.append("")
        lines.append("| code | baseline | this pass | Δ | n_validate |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, b, n, d, nt in deltas_imp:
            lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | +{d:.3f} | {nt} |")
        lines.append("")
        if deltas_reg:
            lines.append("### Regressions")
            lines.append("")
            lines.append("| code | baseline | this pass | Δ | n_validate |")
            lines.append("|---|---:|---:|---:|---:|")
            for cls, b, n, d, nt in deltas_reg:
                lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | {d:.3f} | {nt} |")
            lines.append("")

    lines.append("## Failure-mode breakdown (held-out union)")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, val in sorted(result["failures"]["kinds"].items(),
                         key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {val} |")
    lines.append("")

    lines.append("## Gate A interpretation")
    lines.append("")
    full_top1 = v["full_top1"]
    std = v["full_top1_std"]
    if full_top1 >= 0.95 and std <= 0.03:
        lines.append(
            f"✓ **Gate A MET (k-fold)**: held-out top-1={full_top1:.4f} ≥ 0.95 "
            f"AND std={std:.4f} ≤ 0.03.  Channel deployment-ready on accuracy; "
            f"check Gate B (mutual affirmation with cosine) next."
        )
    elif full_top1 >= 0.85:
        lines.append(
            f"~ **Approaching Gate A**: held-out top-1={full_top1:.4f} "
            f"in [0.85, 0.95).  Std={std:.4f}.  Refinement loop should "
            f"push toward 0.95 by targeting per-code gaps (per-category "
            f"table above)."
        )
    elif full_top1 >= 0.65:
        lines.append(
            f"~ **Mid-band**: held-out top-1={full_top1:.4f} in [0.65, 0.85).  "
            f"Std={std:.4f}.  Likely indicates corpus-induced ceiling or "
            f"structural failures (heterogeneous codes like 0.1 INOS).  "
            f"Consider CCO subtype work + targeted refinement."
        )
    else:
        lines.append(
            f"✗ **Below 0.65**: held-out top-1={full_top1:.4f}.  Reference-"
            f"primary protocol should have lifted substantially from synth-"
            f"only baseline; if not, inspect per-fold variance and audit "
            f"weight distribution before refining."
        )
    if diag is not None:
        gap = full_top1 - diag["full_top1"]
        lines.append("")
        lines.append(
            f"**Protocol-switch lift**: reference-primary {full_top1:.4f} "
            f"− synth-only diagnostic {diag['full_top1']:.4f} = "
            f"**+{gap:.4f}**.  This is the headline 'switching to reference-"
            f"primary was worth doing' number; refinement loop continues to "
            f"steer the synth corpus, which retains direct value via fold "
            f"augmentation."
        )
    lines.append("")
    return "\n".join(lines)


def _render_report_reference_plus_synth(result: dict) -> str:
    lines: list[str] = []
    pass_label = (f" — pass {result['pass_idx']}"
                  if result.get("pass_idx") is not None else "")
    lines.append(
        f"# Factorized NHSVM eval{pass_label} — reference+synth (single 80/20)"
    )
    lines.append("")
    s = result["split"]
    t = result["train"]
    v = result["validate"]
    lines.append(
        f"Train: **{s['n_train_total']} examples** "
        f"({s['n_train_real']} real + {s['n_train_synth']} synth, "
        f"weights from `{result['weights_path']}`).  "
        f"Validate: **{s['n_val']} reference examples** (held-out 20% "
        f"single split, seed={s['split_seed']}).  Encoder: {result['encoder']}."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(
        f"- **Held-out 20% top-1: {v['full_top1']:.4f} "
        f"({v['full_n_correct']}/{v['full_n']})**"
    )
    lines.append(f"- Train fit-acc: {t['fit_acc']:.4f}")
    lines.append(f"- Train time: {t['elapsed_sec']:.1f}s "
                 f"({t['epochs']} epochs, loss={t['final_loss']:.4f})")
    diag = v.get("synth_only_diagnostic")
    if diag is not None:
        lines.append(
            f"- Synth-only diagnostic (parallel): {diag['full_top1']:.4f} "
            f"({diag['n_correct']}/{diag['n']})"
        )
    lines.append("")
    lines.append(
        "**Caveat**: single split — no fold variance.  Use "
        "`--training-mode reference-primary --n-folds 5` for a deployment-"
        "grade number with std."
    )
    lines.append("")

    lines.append("## Failure-mode breakdown (held-out 20%)")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, val in sorted(result["failures"]["kinds"].items(),
                         key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {val} |")
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--pass-idx", type=int, default=None,
                    help="Refinement pass index; pass-numbered outputs "
                         "written when set.  Without, single-run outputs.")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Corpus directory containing synth_rows.jsonl")
    ap.add_argument("--training-mode",
                    choices=VALID_TRAINING_MODES,
                    default="synth-only",
                    help="Training protocol.  synth-only (default, "
                         "back-compat) trains on synth corpus and "
                         "validates on full reference.  reference-primary "
                         "runs k-fold on the reference with synth "
                         "augmentation.  reference+synth uses a single "
                         "80/20 split + full synth.")
    ap.add_argument("--n-folds", type=int, default=5,
                    help="Number of folds for reference-primary mode")
    ap.add_argument("--fold-seed", type=int, default=42,
                    help="Seed for stratified-k-fold / stratified-split")
    ap.add_argument("--weights-path", type=Path,
                    default=Path("build/data/agent_mediated/training_weights.json"),
                    help="Path to audit-derived training weights")
    ap.add_argument("--augment-floor", type=int, default=5,
                    help="In reference-primary, top up codes whose fold-"
                         "train count falls below this floor by drawing "
                         "from synth corpus")
    ap.add_argument("--also-report-synth-only", action="store_true",
                    help="In non-synth-only modes, additionally train a "
                         "synth-only head and report its full-reference "
                         "top-1 as a parallel diagnostic")
    args = ap.parse_args()

    # Logging both to console and to RUN_LOG (append mode so pass-by-pass
    # runs accumulate in one log).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RUN_LOG, mode="a"),
        ],
    )

    result = run_eval(
        training_mode=args.training_mode,
        corpus_dir=args.corpus_dir,
        refresh_embeddings=args.refresh_embeddings,
        batch_size=args.batch_size,
        pass_idx=args.pass_idx,
        n_folds=args.n_folds,
        fold_seed=args.fold_seed,
        weights_path=args.weights_path,
        augment_floor=args.augment_floor,
        also_report_synth_only=args.also_report_synth_only,
    )

    # Pass-numbered output paths
    if args.pass_idx is not None:
        results_json = REPORT_DIR / f"results_pass{args.pass_idx}.json"
        per_cat_json = REPORT_DIR / f"per_category_accuracy_pass{args.pass_idx}.json"
    else:
        results_json = RESULTS_JSON
        per_cat_json = PER_CAT_JSON

    # Strip the verbose `predictions` from the main JSON; persist separately
    predictions = result.pop("predictions")
    results_json.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", results_json)

    per_cat_json.write_text(json.dumps({
        "per_category": result["per_category_accuracy"],
        "predictions": predictions,
    }, indent=2))
    log.info("Wrote %s", per_cat_json)

    # Symlink results.json + per_category_accuracy.json → latest pass so
    # downstream consumers that don't pass --pass-idx keep working.
    if args.pass_idx is not None:
        for live, target in [
            (RESULTS_JSON, results_json),
            (PER_CAT_JSON, per_cat_json),
        ]:
            if live.exists() or live.is_symlink():
                live.unlink()
            live.symlink_to(target.name)
            log.info("Updated symlink %s → %s", live, target.name)

    report = render_report(result)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    pass_label = (f"pass {args.pass_idx}" if args.pass_idx is not None
                  else "single run")
    mode = result.get("training_mode", "synth-only")
    print(f"=== Phase D headline ({pass_label}, mode={mode}) ===")
    v = result["validate"]
    if mode == "synth-only":
        print(f"  full-validate top-1:     {v['full_top1']:.4f} "
              f"({v['full_n_correct']}/{v['full_n']})")
        print(f"  continuity-271 top-1:    {v['continuity_top1']:.4f} "
              f"({v['continuity_n_correct']}/{v['continuity_n']})")
        print(f"  train fit-acc:           {result['train']['fit_acc']:.4f}")
        print(f"  historical 0.6125 anchor (different regime): "
              f"reference-only train + 271-test")
    elif mode == "reference-primary":
        print(f"  held-out full-ref top-1: {v['full_top1']:.4f} "
              f"({v['full_n_correct']}/{v['full_n']})")
        print(f"  per-fold val mean±std:   {v['full_top1_mean']:.4f} "
              f"± {v['full_top1_std']:.4f}")
        print(f"  per-fold val_top1:       {v['full_top1_per_fold']}")
        print(f"  train fit-acc mean±std:  {result['train']['fit_acc_mean']:.4f} "
              f"± {result['train']['fit_acc_std']:.4f}")
        diag = v.get("synth_only_diagnostic")
        if diag is not None:
            print(f"  synth-only diagnostic:   {diag['full_top1']:.4f} "
                  f"({diag['n_correct']}/{diag['n']}) — parallel baseline")
            print(f"  protocol-switch lift:    "
                  f"+{v['full_top1'] - diag['full_top1']:.4f}")
    elif mode == "reference+synth":
        print(f"  held-out 20% top-1:      {v['full_top1']:.4f} "
              f"({v['full_n_correct']}/{v['full_n']})")
        print(f"  train fit-acc:           {result['train']['fit_acc']:.4f}")
        diag = v.get("synth_only_diagnostic")
        if diag is not None:
            print(f"  synth-only diagnostic:   {diag['full_top1']:.4f} "
                  f"({diag['n_correct']}/{diag['n']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
