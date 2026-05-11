# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Lazy-loading inference wrappers for CatBoost and SVM classifiers.

Loads model files from configured paths on first use. When model files
are not present, predict functions return None (graceful degradation).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_catboost = None
_catboost_loaded = False
_svm = None
_svm_loaded = False
_catboost_path: Path | None = None
_svm_path: Path | None = None
_lock = threading.Lock()


def configure_paths(
    catboost_path: str | Path | None = None,
    svm_path: str | Path | None = None,
) -> None:
    """Override default model file locations.

    Call ``reset()`` first if models are already loaded.
    """
    global _catboost_path, _svm_path
    _catboost_path = Path(catboost_path) if catboost_path else None
    _svm_path = Path(svm_path) if svm_path else None


def get_catboost(model_path: str | Path | None = None):
    """Lazy-load CatBoost model (thread-safe). Returns None if not available."""
    global _catboost, _catboost_loaded
    if _catboost_loaded:
        return _catboost

    with _lock:
        if _catboost_loaded:
            return _catboost

        _catboost_loaded = True
        path = Path(model_path) if model_path else (_catboost_path or Path("build/models/catboost.cbm"))
        if not path.exists():
            logger.debug("CatBoost model not found at %s, skipping", path)
            return None

        try:
            from atelier.classify.catboost_classifier import CatBoostColumnClassifier
            _catboost = CatBoostColumnClassifier.load(path)
            return _catboost
        except Exception as e:
            logger.warning("Failed to load CatBoost model: %s", e)
            return None


def get_svm(model_path: str | Path | None = None):
    """Lazy-load SVM model (thread-safe). Returns None if not available."""
    global _svm, _svm_loaded
    if _svm_loaded:
        return _svm

    with _lock:
        if _svm_loaded:
            return _svm

        _svm_loaded = True
        path = Path(model_path) if model_path else (_svm_path or Path("build/models/svm.pkl"))
        if not path.exists():
            logger.debug("SVM model not found at %s, skipping", path)
            return None

        try:
            from atelier.classify.svm_classifier import SVMClassifier
            _svm = SVMClassifier.load(path)
            return _svm
        except Exception as e:
            logger.warning("Failed to load SVM model: %s", e)
            return None


def predict_catboost(features, category_set) -> tuple[dict[str, float], dict[str, float]] | None:
    """Get CatBoost probabilities + variance for a single column.

    The classifier encodes the structured per-feature input internally
    (one SentenceTransformer slice per text feature; native scalar
    columns for numerics).  See
    :class:`atelier.classify.catboost_classifier.CatBoostColumnClassifier`.

    Args:
        features: ColumnFeatures from extract_features().
        category_set: Category set for embedding context (unused at
            the inference layer; kept for backward-compat callers).

    Returns:
        (proba, variance) dicts, or None if model not loaded.
    """
    model = get_catboost()
    if model is None:
        return None

    proba = model.predict_proba_single(features)

    try:
        variance_list = model.virtual_ensemble_variance([features])
        variance = variance_list[0] if variance_list else {}
    except Exception:
        variance = {}

    return proba, variance


def predict_svm(features) -> dict[str, float] | None:
    """Get SVM probabilities for a single column.

    Args:
        features: ColumnFeatures from extract_features().

    Returns:
        {code: probability} dict, or None if model not loaded.
    """
    model = get_svm()
    if model is None:
        return None

    from atelier.classify.svm_classifier import build_svm_text

    text = build_svm_text(
        features.column_name_humanized,
        features.column_type,
        features.sample_values_text.split(", ") if features.sample_values_text else None,
    )

    return model.predict_proba_single(text)


def reset():
    """Reset cached models and configured paths (useful for testing)."""
    global _catboost, _catboost_loaded, _svm, _svm_loaded, _catboost_path, _svm_path
    with _lock:
        _catboost = None
        _catboost_loaded = False
        _svm = None
        _svm_loaded = False
        _catboost_path = None
        _svm_path = None


def install_catboost(model) -> None:
    """Install an in-memory CatBoost model, bypassing disk-load.

    Used by the fit-to-LLM mode in the classification pipeline:
    after the LLM sweep produces labels, CatBoost is trained in-memory
    on (embedding_text, llm_code) pairs and installed here so the rest
    of the evidence-fusion path uses the fresh model without ever
    touching the pre-trained ``classify.catboost_model_path`` file.

    Calling :func:`reset` clears this install too; prefer
    :func:`reset_catboost` when you want to swap only one model.
    """
    global _catboost, _catboost_loaded
    with _lock:
        _catboost = model
        _catboost_loaded = True
        logger.info("CatBoost model installed in-memory (fit-to-LLM mode)")


def install_svm(model) -> None:
    """Install an in-memory SVM model, bypassing disk-load.

    Parallel to :func:`install_catboost`.  Used by the incremental SVM
    hot-swap during bootstrap: after re-training on accumulated LLM
    labels, the new model is installed here so subsequent ML validation
    and final classification see it without touching the serialized
    ``svm_frontier.pkl`` on disk.

    Crucially this does NOT touch ``_catboost`` state — previously the
    hot-swap called :func:`reset` which silently wiped the fit-to-LLM
    CatBoost install.
    """
    global _svm, _svm_loaded
    with _lock:
        _svm = model
        _svm_loaded = True
        logger.info("SVM model installed in-memory")


def reset_catboost() -> None:
    """Surgical reset of CatBoost state only.  SVM state preserved."""
    global _catboost, _catboost_loaded, _catboost_path
    with _lock:
        _catboost = None
        _catboost_loaded = False
        _catboost_path = None


def reset_svm() -> None:
    """Surgical reset of SVM state only.  CatBoost state preserved."""
    global _svm, _svm_loaded, _svm_path
    with _lock:
        _svm = None
        _svm_loaded = False
        _svm_path = None
