# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Tests for the Crammer-Singer training-time NHSVM path.

Covers the production hierarchical NHSVM in its current form (Choi
et al. (2015) Eq. 5 via sklearn's ``multi_class="crammer_singer"``
LinearSVC + per-class expansion at inference):

  - Hierarchical fit produces a usable classifier (round-trip).
  - Inference uses per-class expansion (not universal): confirmed by
    counting the number of times ``expand_with_labels`` is invoked per
    prediction.
  - Probabilities returned by ``predict_proba`` are non-negative and
    sum to ~1 across the classes returned.
  - Save + load round-trip preserves classes and predictions.
  - Loading a bundle without the ``nhsvm_variant`` tag raises (the
    roll-forward enforcement).
  - Loading a bundle with a stale ``nhsvm_variant`` tag raises.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _build_test_taxonomy():
    """Minimal hierarchy with two subtrees + a parent/child split."""
    from atelier.classify.taxonomy import (
        HierarchicalCategorySet,
        ReferenceCategory,
    )

    cats = [
        ReferenceCategory(code="R", label="Root", embedding_text="root", parent_code=None),
        ReferenceCategory(code="R.A", label="A", embedding_text="a", parent_code="R"),
        ReferenceCategory(code="R.A.X", label="X", embedding_text="ax", parent_code="R.A"),
        ReferenceCategory(code="R.A.Y", label="Y", embedding_text="ay", parent_code="R.A"),
        ReferenceCategory(code="R.B", label="B", embedding_text="b", parent_code="R"),
        ReferenceCategory(code="R.B.Z", label="Z", embedding_text="bz", parent_code="R.B"),
    ]
    return HierarchicalCategorySet(name="test", categories=cats)


def _build_training_data():
    """Distinct patterns per leaf — enough signal for the SVM to fit cleanly."""
    samples = [
        ("alpha apple absent", "R.A.X"),
        ("alpha able ardent",  "R.A.X"),
        ("alpha across arrow", "R.A.X"),
        ("yes yellow yonder",  "R.A.Y"),
        ("yes yearly yummy",   "R.A.Y"),
        ("yes yawn yacht",     "R.A.Y"),
        ("zero zenith zero",   "R.B.Z"),
        ("zero zebra zone",    "R.B.Z"),
        ("zero zip zipper",    "R.B.Z"),
    ]
    return [t for t, _ in samples], [c for _, c in samples]


def _fit_hierarchical_classifier():
    from atelier.classify.svm_classifier import SVMClassifier

    cs = _build_test_taxonomy()
    texts, labels = _build_training_data()
    model = SVMClassifier(category_set=cs, hierarchical=True)
    model.fit(texts, labels)
    return model, cs


# ── Training + inference contract ───────────────────────────────────

def test_hierarchical_fit_returns_fitted_classifier():
    model, _ = _fit_hierarchical_classifier()
    assert model.is_fitted
    # Crammer-Singer is hierarchical-only; the path should have stored
    # the expander and feature_union (the dense-input contract).
    assert model._expander is not None
    assert model._feature_union is not None
    assert model._svd is not None


def test_predict_proba_uses_per_class_expansion():
    """Inference must invoke ``expand_with_labels`` once per candidate
    class per sample — that's the per-class Choi Eq. 5 inference rule.
    Universal expansion would invoke ``expand_universal`` instead.
    """
    model, _ = _fit_hierarchical_classifier()
    n_classes = len(model._classes)
    real_expand_with_labels = model._expander.expand_with_labels
    real_expand_universal = model._expander.expand_universal

    call_counts = {"with_labels": 0, "universal": 0}

    def counting_with_labels(X, labels):
        call_counts["with_labels"] += 1
        return real_expand_with_labels(X, labels)

    def counting_universal(X):
        call_counts["universal"] += 1
        return real_expand_universal(X)

    model._expander.expand_with_labels = counting_with_labels  # type: ignore[assignment]
    model._expander.expand_universal = counting_universal  # type: ignore[assignment]
    try:
        _ = model.predict_proba_single("alpha apple")
    finally:
        model._expander.expand_with_labels = real_expand_with_labels  # type: ignore[assignment]
        model._expander.expand_universal = real_expand_universal  # type: ignore[assignment]

    assert call_counts["with_labels"] == n_classes, (
        f"Per-class inference should call expand_with_labels once per "
        f"class; saw {call_counts['with_labels']} for {n_classes} classes"
    )
    assert call_counts["universal"] == 0, (
        "Crammer-Singer inference must NOT use universal expansion — "
        "the per-class path is the only inference geometry under the "
        "roll-forward posture"
    )


def test_predict_proba_returns_normalized_distribution():
    model, _ = _fit_hierarchical_classifier()
    proba = model.predict_proba_single("alpha apple")
    assert proba, "Expected at least one class with positive probability"
    for code, p in proba.items():
        assert 0.0 <= p <= 1.0, f"Probability for {code} out of range: {p}"
    total = sum(proba.values())
    # The filter ``p > 1e-6`` strips dust; remainder should sum near 1.
    assert 0.9 < total <= 1.0 + 1e-6, f"Probabilities sum to {total}, expected ~1.0"


# ── Save / load round-trip ──────────────────────────────────────────

def test_save_load_roundtrip_preserves_predictions(tmp_path: Path):
    from atelier.classify.svm_classifier import SVMClassifier

    model, _ = _fit_hierarchical_classifier()
    bundle = tmp_path / "nhsvm.pkl"
    model.save(bundle)

    loaded = SVMClassifier.load(bundle)
    assert loaded.is_fitted
    assert loaded._classes == model._classes
    proba_before = model.predict_proba_single("alpha apple")
    proba_after = loaded.predict_proba_single("alpha apple")
    assert set(proba_before) == set(proba_after)
    for code in proba_before:
        assert proba_before[code] == pytest.approx(proba_after[code], abs=1e-9)


def test_save_writes_classes_sidecar(tmp_path: Path):
    import json
    model, _ = _fit_hierarchical_classifier()
    bundle = tmp_path / "nhsvm.pkl"
    model.save(bundle)
    sidecar = bundle.with_suffix(".classes.json")
    assert sidecar.exists()
    persisted = json.loads(sidecar.read_text())
    assert persisted == model._classes


# ── Roll-forward enforcement ────────────────────────────────────────

def test_load_rejects_bundle_missing_variant_tag(tmp_path: Path):
    """Pre-Crammer-Singer bundles (no ``nhsvm_variant`` key) must fail
    loudly under the roll-forward posture.  Silent loading would let
    OvR-trained weights drive the per-class inference path."""
    import joblib

    from atelier.classify.svm_classifier import SVMClassifier

    model, _ = _fit_hierarchical_classifier()
    bundle_path = tmp_path / "legacy.pkl"
    # Construct a bundle in the pre-Crammer-Singer shape (no variant tag).
    legacy_bundle = {
        "hierarchical": True,
        # nhsvm_variant intentionally omitted
        "feature_union": model._feature_union,
        "svd": model._svd,
        "expander": model._expander,
        "classifier": model._pipeline,
        "classes": list(model._classes),
    }
    joblib.dump(legacy_bundle, str(bundle_path))

    with pytest.raises(RuntimeError, match="nhsvm_variant"):
        SVMClassifier.load(bundle_path)


def test_load_rejects_bundle_with_stale_variant_tag(tmp_path: Path):
    """Any bundle whose variant tag doesn't match the current expected
    value must fail loudly — protects against future shape changes
    accidentally loading older bundles."""
    import joblib

    from atelier.classify.svm_classifier import SVMClassifier

    model, _ = _fit_hierarchical_classifier()
    bundle_path = tmp_path / "stale.pkl"
    stale_bundle = {
        "hierarchical": True,
        "nhsvm_variant": "crammer_singer.v0-deprecated",
        "feature_union": model._feature_union,
        "svd": model._svd,
        "expander": model._expander,
        "classifier": model._pipeline,
        "classes": list(model._classes),
    }
    joblib.dump(stale_bundle, str(bundle_path))

    with pytest.raises(RuntimeError, match="nhsvm_variant"):
        SVMClassifier.load(bundle_path)


# ── Pipeline integration: cache suffix uses _nhsvm_cs ───────────────

def test_pipeline_cache_suffix_is_nhsvm_cs():
    """The pipeline must namespace Crammer-Singer-trained models under
    ``_nhsvm_cs`` so legacy ``_nhsvm.pkl`` files from the pre-roll-
    forward era are not silently picked up.
    """
    pipeline_src = Path("src/atelier/classify/pipeline.py").read_text()
    # The suffix construction lives near the cache-key derivation.
    assert '"_nhsvm_cs"' in pipeline_src, (
        "Pipeline cache suffix for hierarchical NHSVM should be "
        "'_nhsvm_cs' after the roll-forward to Crammer-Singer."
    )
    assert 'suffix = "_nhsvm" if svm_hierarchical else ""' not in pipeline_src, (
        "Pipeline still uses the pre-roll-forward '_nhsvm' suffix; "
        "Crammer-Singer bundles should be namespaced under '_nhsvm_cs'."
    )
