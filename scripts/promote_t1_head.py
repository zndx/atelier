#!/usr/bin/env python
"""scripts/promote_t1_head.py — train + promote the T1 baseline NHSVM head.

Trains a factorized NHSVM head on the FULL agent-mediated reference +
synth augmentation (no fold held out), wraps it in NHSVMHeadAdapter,
and promotes it to 'current' in nhsvm_head_registry.

This is the head that pipeline.py consumes as the T1-stage SVM channel
under ``classify.svm.source ∈ {registered, auto}``.  It differs from
Phase D's k-fold heads (which are validation models — each trained on
80% per fold); the T1 head trains on EVERYTHING the operator has
audited as reference + augmentation, since the deployment model gets
to see all available labels.

Promotion is two-step (status='building' → 'current') so the variance
threshold gate can intervene if metrics are loaded between the steps.
This script does the full sequence in one shot since it's invoked
manually after a satisfactory Phase D measurement.

Usage:
  python scripts/promote_t1_head.py
  python scripts/promote_t1_head.py --corpus-dir build/data/svm_training/corpus_v3
  python scripts/promote_t1_head.py --augment-floor 5 --no-promote-current
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path("build/data/svm_training/corpus_v3"),
                    help="Synth corpus directory (default: corpus_v3)")
    ap.add_argument("--weights-path", type=Path,
                    default=Path("build/data/agent_mediated/training_weights.json"),
                    help="Audit-derived per-row weights")
    ap.add_argument("--augment-floor", type=int, default=5,
                    help="Top up codes whose reference count < N from synth (default 5)")
    ap.add_argument("--taxonomy-id", default="default",
                    help="Logical taxonomy identifier (default: 'default')")
    ap.add_argument("--encoder", default="answerdotai/ModernBERT-base",
                    help="Embedding encoder identity")
    ap.add_argument("--fold-seed", type=int, default=42)
    ap.add_argument("--no-promote-current", action="store_true",
                    help="Register in status='building' but skip the "
                         "current-flip (operator promotes manually after "
                         "reviewing metrics)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("promote_t1_head")

    # Imports deferred so the help text doesn't trigger torch/numpy load.
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
    from atelier.registry.nhsvm_head import promote_to_current

    # 1. Load reference + per-row weights
    ref_rows, ref_labels, ref_weights, _, _ = load_reference_rows_with_weights(
        weights_path=args.weights_path,
    )
    log.info("reference: %d rows  %d codes",
             len(ref_rows), len(set(ref_labels)))

    # 2. Encode reference (cache hit if already populated)
    ref_texts, _ = build_texts_and_labels(ref_rows)
    X_ref = encode_with_cache(
        ref_texts, cache_key="full-reference",
        refresh=False, batch_size=64,
    )

    # 3. Load synth corpus
    synth_rows = load_synth_rows(args.corpus_dir)
    synth_labels_all = [r.code for r in synth_rows]
    synth_counts = Counter(synth_labels_all)
    keep = [i for i, l in enumerate(synth_labels_all) if synth_counts[l] >= 2]
    synth_rows = [synth_rows[i] for i in keep]
    synth_labels = [synth_labels_all[i] for i in keep]
    synth_texts, _ = build_texts_and_labels(synth_rows)
    ch = corpus_hash(args.corpus_dir)
    X_synth = encode_with_cache(
        synth_texts, cache_key=f"synth-only-{ch}",
        refresh=False, batch_size=64,
    )

    # 4. Augment under-represented reference codes
    synth_by_code: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(synth_labels):
        synth_by_code[l].append(i)
    ref_counts = Counter(ref_labels)
    rng = np.random.RandomState(args.fold_seed)
    aug_idx: list[int] = []
    aug_labels: list[str] = []
    for code in sorted(ref_counts):
        if ref_counts[code] >= args.augment_floor:
            continue
        need = args.augment_floor - ref_counts[code]
        pool = synth_by_code.get(code, [])
        if not pool:
            continue
        chosen = list(rng.choice(pool, size=min(need, len(pool)), replace=False))
        aug_idx.extend(chosen)
        aug_labels.extend(code for _ in chosen)
    log.info("augmentation: %d synth rows across %d codes",
             len(aug_idx), len({c for c in aug_labels}))

    if aug_idx:
        X_train = np.concatenate([X_ref, X_synth[aug_idx]], axis=0)
        y_train = list(ref_labels) + list(aug_labels)
        w_train = np.concatenate(
            [ref_weights, np.ones(len(aug_idx), dtype=np.float32)],
        )
    else:
        X_train, y_train, w_train = X_ref, list(ref_labels), ref_weights

    # 5. Train
    cat_set = build_category_set()
    head, train_result = fit_factorized_nhsvm(
        X_train, y_train, cat_set,
        sample_weights=w_train,
        **BEST_KNOB, verbose=True, eval_every=50,
    )
    log.info("trained: fit_acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, train_result.elapsed_sec)

    # 6. Wrap + promote
    adapter = NHSVMHeadAdapter(
        head, encoder_id=args.encoder, embed_dim=int(X_train.shape[1]),
        training_metadata={
            "training_mode": "reference-primary",
            "fit_acc": float(train_result.final_train_acc),
            "n_train": len(y_train),
            "n_ref": len(ref_labels),
            "n_synth_aug": len(aug_idx),
            "augment_floor": args.augment_floor,
            "fold_seed": args.fold_seed,
            "description": (
                "T1 baseline head: full reference + augmentation, no fold "
                "held-out"
            ),
        },
    )

    vocab_sig = hashlib.sha1(
        ",".join(sorted(set(ref_labels) | set(synth_labels))).encode()
    ).hexdigest()[:16]
    ref_hash = reference_hash(args.weights_path)

    row = promote_head(
        adapter,
        taxonomy_id=args.taxonomy_id,
        vocab_sig=vocab_sig,
        training_mode="reference-primary",
        reference_hash=ref_hash,
        corpus_hash=ch,
        fold_seed=args.fold_seed,
        augment_floor=args.augment_floor,
        metrics={
            "fit_acc": float(train_result.final_train_acc),
            "training_phase": "T1_baseline",
            "n_train_total": len(y_train),
            "n_ref": len(ref_labels),
            "n_synth_aug": len(aug_idx),
        },
        summary=(
            f"T1 baseline (no fold held out): {len(y_train)} train rows, "
            f"fit_acc={train_result.final_train_acc:.4f}"
        ),
    )
    log.info("promote_head: id=%s head_sig=%s artifact=%s",
             row["id"], row["head_sig"], row["artifact_path"])

    if not args.no_promote_current:
        promote_to_current(row["id"])
        log.info("promoted to current ✓")
    else:
        log.info(
            "registered in status='building' (per --no-promote-current); "
            "operator should call "
            "atelier.registry.nhsvm_head.promote_to_current(id=%s) after "
            "review", row["id"],
        )

    print()
    print(f"head_id:   {row['id']}")
    print(f"head_sig:  {row['head_sig']}")
    print(f"artifact:  {row['artifact_path']}")
    print(f"status:    {'current' if not args.no_promote_current else 'building'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
