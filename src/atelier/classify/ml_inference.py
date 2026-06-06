"""Lazy-loading inference wrappers for CatBoost and SVM classifiers.

CatBoost is loaded from a configured path on first use (pre-trained or
fit-to-LLM).  SVM is installed in-memory by ``_ensure_per_vocab_svm``
during the pipeline — there is no disk-load fallback.  When the
per-vocab SVM build fails, SVM evidence is absent (loud, not silent).
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
_lock = threading.Lock()


def configure_paths(
    catboost_path: str | Path | None = None,
) -> None:
    """Override default CatBoost model file location.

    SVM is not configured here — it is installed in-memory by
    ``_ensure_per_vocab_svm`` during the pipeline run.  There is no
    disk-load fallback for SVM; if the per-vocab build fails, SVM
    evidence is absent (deployment-degraded, logged loudly).

    Call ``reset()`` first if models are already loaded.
    """
    global _catboost_path
    _catboost_path = Path(catboost_path) if catboost_path else None


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
    """Return the in-memory SVM model, or None.

    The per-vocab SVM is installed by ``_ensure_per_vocab_svm`` during
    the pipeline via ``install_svm``.  There is no disk-load fallback —
    if install_svm was never called, this returns None and the SVM
    evidence source is absent for the run.

    The ``model_path`` parameter is retained for explicit callers
    (tests, ad-hoc scripts) that want to load a specific model file.
    """
    global _svm, _svm_loaded
    if _svm_loaded:
        return _svm

    with _lock:
        if _svm_loaded:
            return _svm

        if model_path:
            _svm_loaded = True
            path = Path(model_path)
            if not path.exists():
                logger.debug("SVM model not found at %s", path)
                return None
            try:
                from atelier.classify.svm_classifier import SVMClassifier
                _svm = SVMClassifier.load(path)
                return _svm
            except Exception as e:
                logger.warning("Failed to load SVM model: %s", e)
                return None

        # No explicit path and no in-memory install → SVM absent.
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

    Dispatch:
      - If the installed model exposes ``predict_proba_features``
        (current contract: ``NHSVMHeadAdapter``), pass the full
        ``ColumnFeatures`` so the model can include sibling enrichment
        in its text + match the training-time input shape.
      - Otherwise, fall back to the legacy ``predict_proba_single``
        text interface (current contract: ``SVMClassifier``).
    """
    model = get_svm()
    if model is None:
        return None

    # Richer features-aware path (NHSVMHeadAdapter exposes this).
    if hasattr(model, "predict_proba_features"):
        return model.predict_proba_features(features)

    # Legacy SVMClassifier interface — plain text in, dict out.
    from atelier.classify.svm_classifier import build_svm_text

    text = build_svm_text(
        features.column_name_humanized,
        features.column_type,
        features.sample_values_text.split(", ") if features.sample_values_text else None,
    )

    return model.predict_proba_single(text)


def reset():
    """Reset cached models and configured paths (useful for testing)."""
    global _catboost, _catboost_loaded, _svm, _svm_loaded, _catboost_path
    with _lock:
        _catboost = None
        _catboost_loaded = False
        _svm = None
        _svm_loaded = False
        _catboost_path = None


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
    """Install the per-vocab SVM trained during the pipeline.

    Called by ``_ensure_per_vocab_svm`` after training (or loading
    from cache) the enrichment-derived per-vocabulary SVM.  This is
    the ONLY path that populates the SVM for a pipeline run — there
    is no disk-load fallback.

    Does NOT touch ``_catboost`` state.
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
    global _svm, _svm_loaded
    with _lock:
        _svm = None
        _svm_loaded = False
