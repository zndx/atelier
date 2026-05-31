"""Stratified train/val partitioning helpers + canonical training knob.

Used by the Phase D dispatcher (``atelier.optimize.svm.train``), the
Gate B head trainer (``atelier.optimize.svm.gate``), the
post-integration diagnostic, and the refinement-loop.

Public API:
    - ``stratified_split(labels, test_size, seed)`` → single 80/20 split
    - ``stratified_k_fold(labels, n_folds, seed)`` → list of (train_idx, val_idx)
    - ``BEST_KNOB`` — validated training-config dict from the 2026-05-25
      factorized grid; consumed by every fit_factorized_nhsvm caller.
"""
from __future__ import annotations

import numpy as np


# Validated best-knob config from the 2026-05-25 factorized grid.
# Persisted here so the SVM channel's optimize cycle, Gate B, and the
# runtime promotion utility all share an authoritative default.
BEST_KNOB = {
    "epochs": 300,
    "batch_size": 64,
    "lr": 5e-3,
    "weight_decay": 1e-4,
}


def stratified_split(
    labels: list[str], test_size: float = 0.2, seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified 80/20 split that handles 2-example classes by putting
    one in train and one in test.  Classes with a single example are
    excluded (caller should have already dropped singletons).

    Returns (train_indices, test_indices) as lists of original row
    positions.
    """
    rng = np.random.RandomState(seed)
    by_class: dict[str, list[int]] = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for cls, indices in by_class.items():
        if len(indices) < 2:
            train_idx.extend(indices)  # safety net — should be 0 after singleton drop
            continue
        shuf = list(indices)
        rng.shuffle(shuf)
        n_test = max(1, int(round(len(shuf) * test_size)))
        # Guarantee at least 1 in train and 1 in test
        n_test = min(n_test, len(shuf) - 1)
        test_idx.extend(shuf[:n_test])
        train_idx.extend(shuf[n_test:])
    return sorted(train_idx), sorted(test_idx)


def stratified_k_fold(
    labels: list[str], n_folds: int = 5, seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Per-class round-robin k-fold.  Returns a list of (train_idx,
    val_idx) tuples, one per fold.

    Classes with n >= n_folds are shuffled and round-robin-assigned to
    folds (each fold's val set gets ~n/n_folds examples from this class).
    Classes with 2 <= n < n_folds cycle through folds (each example
    contributes to exactly one fold's val partition; class-level coverage
    not guaranteed in every fold).  Singletons go entirely to train every
    fold (caller should have dropped them; we keep them safe).

    Determinism: same seed → identical fold assignments.

    Every example appears in exactly one fold's val partition (or is a
    singleton in train for all folds).  The union of fold-vals is
    therefore the full non-singleton index set.
    """
    rng = np.random.RandomState(seed)
    by_class: dict[str, list[int]] = {}
    for i, l in enumerate(labels):
        by_class.setdefault(l, []).append(i)

    # Assign each non-singleton index to a single fold-val partition.
    val_by_fold: list[list[int]] = [[] for _ in range(n_folds)]
    singletons: list[int] = []
    for cls, indices in by_class.items():
        if len(indices) < 2:
            singletons.extend(indices)
            continue
        shuf = list(indices)
        rng.shuffle(shuf)
        # Round-robin assignment: index k goes to fold (k mod n_folds).
        # When n < n_folds, some folds get nothing from this class — fine.
        # When n >= n_folds, distribution is balanced to within 1.
        for k, idx in enumerate(shuf):
            val_by_fold[k % n_folds].append(idx)

    folds: list[tuple[list[int], list[int]]] = []
    all_non_singleton = sorted(
        i for cls, indices in by_class.items() if len(indices) >= 2 for i in indices
    )
    non_singleton_set = set(all_non_singleton)
    for k in range(n_folds):
        val_idx = sorted(val_by_fold[k])
        val_set = set(val_idx)
        train_idx = sorted(
            list((non_singleton_set - val_set)) + singletons
        )
        folds.append((train_idx, val_idx))
    return folds
