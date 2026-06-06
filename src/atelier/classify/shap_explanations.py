"""Item-wise SHAP feature importance analysis.

Provides per-item (per-column) explanations of classification decisions,
complementing the global SAGE analysis.  Two methods are available;
both attribute to the 12 named features defined in
``features.FEATURE_NAMES``.

1. **CatBoost TreeSHAP** — exact O(TLD) algorithm on the CatBoost
   model's structured input.  CatBoost is trained with each source
   feature occupying its own input slice (7 SentenceTransformer-
   embedded text features + 5 scalar features; see
   :func:`atelier.classify.embedding.feature_input_groups`).
   ``_build_feature_groups`` slices the raw SHAP output by source
   feature, so ``shap_top1_name``/``shap_top2_name``/``shap_top3_name``
   are always one of the 12 ``FEATURE_NAMES``.  Fast (~2 seconds for
   50 items).

2. **Embedding PermutationSHAP** — ``shap.PermutationExplainer`` (or
   the GPU-accelerated kernel in ``gpu_importance.gpu_permutation_shap``)
   on the same 12 features.  Reaches per-feature attribution by
   ablating each feature's contribution to the embedding text and
   re-encoding.  Slower (~60s on CPU; fast on GPU); reuses
   ``FeatureMaskModel`` from sage.py.

Both methods produce the same conceptual output (per-feature
attribution); they differ in speed-vs-fidelity tradeoffs.  TreeSHAP is
exact for the CatBoost model's tree decisions; PermutationSHAP is
sampling-based.  The auto-selection priority is:

    GPU available + no CatBoost  →  Embedding PermutationSHAP (GPU)
    method="auto", default        →  Try TreeSHAP first (now that
                                     it produces per-feature output);
                                     fall back to PermutationSHAP
                                     when CatBoost isn't loaded
    method=opt-in                 →  Honored as requested

Per-feature TreeSHAP became viable in a 2026-04 refactor that moved
CatBoost from a single 384-dim concatenated-embedding input to a
named-feature structured input.  Prior to that, the 384 dims were a
learned compression of all 12 features and TreeSHAP could only
report aggregate "embedding" contribution.

Ported from signals/src/sigint/shap_analysis.py, adapted for atelier's
embedding.py and 12-feature ColumnFeatures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from atelier.classify.features import ColumnFeatures
    from atelier.classify.taxonomy import HierarchicalCategorySet

logger = logging.getLogger(__name__)


# ── Feature group definitions for CatBoost feature space ──────────


def _build_feature_groups(
    model_groups: list[tuple[str, int, int]] | None = None,
) -> list[tuple[str, int, int]]:
    """Return ``(name, start, end)`` slices for SHAP per-feature grouping.

    Defers to the model's persisted ``_feature_groups`` (saved at fit
    time and loaded back from the sidecar JSON) when available, falling
    back to the canonical layout from
    :func:`atelier.classify.embedding.feature_input_groups`.

    The model-side persistence guarantees that even if the canonical
    layout evolves, a previously-trained model's SHAP output is still
    grouped against the layout it was trained with — no
    fit-time/inference-time drift.
    """
    if model_groups:
        return list(model_groups)
    from atelier.classify.embedding import feature_input_groups
    return list(feature_input_groups())


@dataclass(frozen=True)
class ShapResult:
    """Per-item SHAP feature attribution result."""

    feature_names: list[str]
    shap_values: np.ndarray  # (N, n_features)
    base_value: float
    method: str  # "catboost_treeshap" or "embedding_permutation"
    n_items: int
    elapsed_seconds: float

    def top_features(self, item_idx: int, k: int = 3) -> list[tuple[str, float]]:
        """Top-k features by absolute SHAP value for one item."""
        vals = self.shap_values[item_idx]
        indices = np.argsort(-np.abs(vals))[:k]
        return [(self.feature_names[i], float(vals[i])) for i in indices]

    def to_records(self, k: int = 3) -> list[dict]:
        """Per-item dicts with shap_top1_name, shap_top1_value, ..., shap_topK_*."""
        records = []
        for i in range(self.n_items):
            row: dict = {}
            top = self.top_features(i, k=k)
            for rank, (name, value) in enumerate(top, 1):
                row[f"shap_top{rank}_name"] = name
                row[f"shap_top{rank}_value"] = round(value, 6)
            # Pad if fewer than k features
            for rank in range(len(top) + 1, k + 1):
                row[f"shap_top{rank}_name"] = ""
                row[f"shap_top{rank}_value"] = 0.0
            records.append(row)
        return records

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "feature_names": self.feature_names,
            "base_value": self.base_value,
            "n_items": self.n_items,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


# ── CatBoost TreeSHAP ────────────────────────────────────────────


def run_catboost_shap(
    cb_classifier,
    features_list,
    predicted_indices: np.ndarray,
) -> ShapResult:
    """Compute per-item SHAP values via CatBoost TreeSHAP, grouped by source feature.

    The classifier was trained with a structured input where each of the
    12 ``ColumnFeatures`` occupies its own named slice (see
    :func:`atelier.classify.embedding.feature_input_groups`).  TreeSHAP
    runs at the per-dim level (raw SHAP shape ``(N, n_classes,
    n_dims+1)`` for multiclass), and we sum within each named slice to
    produce per-feature attribution.

    The grouping is read from the classifier's persisted
    ``_feature_groups`` so saved models stay aligned with the input
    shape they were trained on, even if the canonical layout evolves.

    Args:
        cb_classifier: Fitted ``CatBoostColumnClassifier`` (or any
            object exposing ``_model`` and ``_feature_groups``).
        features_list: List of ColumnFeatures for each item to explain.
        predicted_indices: (N,) array of predicted class indices.

    Returns:
        ShapResult with per-feature shap_values shaped ``(N, n_groups)``,
        feature_names = the source feature names from FEATURE_NAMES.
    """
    from catboost import Pool
    from atelier.classify.embedding import embed_features

    t0 = time.time()

    # Encode features → structured matrix matching what CatBoost was
    # trained on.  This is the same shape the classifier sees at
    # inference; SHAP on this matrix attributes per input dim, which
    # then groups cleanly by named slice.
    X_eval = embed_features(features_list)
    N, total_dim = X_eval.shape

    model = cb_classifier._model
    pool = Pool(X_eval)

    raw_shap = model.get_feature_importance(
        type="ShapValues",
        data=pool,
    )

    n_feat = total_dim
    if raw_shap.ndim == 3:
        # MultiClass: (N, n_classes, n_features+1).  Extract SHAP values
        # for each item's predicted class.
        item_shap = np.zeros((N, n_feat))
        base_values = np.zeros(N)
        for i in range(N):
            cls_idx = int(predicted_indices[i])
            item_shap[i] = raw_shap[i, cls_idx, :n_feat]
            base_values[i] = raw_shap[i, cls_idx, n_feat]
        base_value = float(np.mean(base_values))
    else:
        # Binary classification
        item_shap = raw_shap[:, :n_feat]
        base_value = float(np.mean(raw_shap[:, n_feat]))

    # Sum per-dim SHAP values within each named feature slice.
    groups = _build_feature_groups(getattr(cb_classifier, "_feature_groups", None))
    group_names = [g[0] for g in groups]
    grouped_shap = np.zeros((N, len(groups)))
    for g_idx, (_, start, end) in enumerate(groups):
        if end <= n_feat:
            grouped_shap[:, g_idx] = np.sum(item_shap[:, start:end], axis=1)

    elapsed = time.time() - t0
    logger.info(
        "CatBoost TreeSHAP: %d items, %d feature groups, %.1fs",
        N, len(groups), elapsed,
    )

    return ShapResult(
        feature_names=group_names,
        shap_values=grouped_shap,
        base_value=base_value,
        method="catboost_treeshap",
        n_items=N,
        elapsed_seconds=elapsed,
    )


# ── Embedding PermutationSHAP ────────────────────────────────────


def run_embedding_shap(
    all_features: list[ColumnFeatures],
    category_set: HierarchicalCategorySet,
    *,
    n_permutations: int = 64,
) -> ShapResult:
    """Compute item-wise SHAP values on the 12-feature embedding classifier.

    GPU path (CUDA available): vectorized kernel in
    :func:`atelier.classify.gpu_importance.gpu_permutation_shap` with
    fixed global donors — fast enough to run by default.  CPU path: the
    upstream ``shap.PermutationExplainer`` wrapped around
    :class:`FeatureMaskModel`, kept as a fallback.

    Args:
        all_features: List of ColumnFeatures for evaluated items.
        category_set: Category set with category definitions.
        n_permutations: Number of permutations for the explainer.

    Returns:
        ShapResult with per-item SHAP values on the 12 named features.
    """
    try:
        from atelier.classify.gpu import preflight_gpu
        if preflight_gpu().available:
            from atelier.classify.gpu_importance import gpu_permutation_shap
            return gpu_permutation_shap(
                all_features, category_set, n_permutations=n_permutations,
            )
    except Exception as exc:
        logger.warning(
            "GPU PermutationSHAP unavailable, falling back to CPU: %s", exc,
        )

    import shap

    from atelier.classify.features import FEATURE_NAMES
    from atelier.classify.sage import FeatureMaskModel

    t0 = time.time()
    N = len(all_features)
    n_feat = len(FEATURE_NAMES)

    # Build feature index matrix (same as SAGE)
    X = np.tile(np.arange(N).reshape(-1, 1), (1, n_feat))

    # Wrap classifier
    model_fn = FeatureMaskModel(all_features, category_set)

    explainer = shap.PermutationExplainer(model_fn, X)
    shap_values = explainer(X, npermutations=n_permutations)

    # shap_values.values: (N, n_feat, n_classes) or (N, n_feat)
    raw = shap_values.values
    if raw.ndim == 3:
        # Extract values for the predicted class per item
        # Use the class with highest model output
        model_output = model_fn(X)  # (N, n_classes)
        predicted = np.argmax(model_output, axis=1)
        item_shap = np.zeros((N, n_feat))
        for i in range(N):
            item_shap[i] = raw[i, :, predicted[i]]
    else:
        item_shap = raw

    base_value = float(np.mean(shap_values.base_values))

    elapsed = time.time() - t0

    # Report cache performance
    total_lookups = model_fn.cache_hits + model_fn.cache_misses
    if total_lookups > 0:
        hit_rate = model_fn.cache_hits / total_lookups * 100
        logger.info(
            "Embedding SHAP cache: %d/%d hits (%.0f%%), %d encodes",
            model_fn.cache_hits, total_lookups, hit_rate, model_fn.cache_misses,
        )

    logger.info(
        "Embedding PermutationSHAP: %d items, %d features, %.1fs",
        N, n_feat, elapsed,
    )

    return ShapResult(
        feature_names=list(FEATURE_NAMES),
        shap_values=item_shap,
        base_value=base_value,
        method="embedding_permutation",
        n_items=N,
        elapsed_seconds=elapsed,
    )


# ── High-level entry point ───────────────────────────────────────


def run_shap_analysis(
    all_features: list[ColumnFeatures],
    category_set: HierarchicalCategorySet,
    *,
    method: str = "auto",
) -> ShapResult | None:
    """High-level entry point: auto-selects method based on hardware + intent.

    Auto-selection priority (revised):
      1. ``method="auto"`` + GPU available → ``run_embedding_shap``
         (PermutationSHAP over the 12 named features) — fast on GPU
         and matches the per-feature interpretive intent.
      2. ``method="auto"`` + no GPU → skip with a log message; CPU
         PermutationSHAP is too slow for default (~60s/run) and
         TreeSHAP's single-"embedding" attribution doesn't carry the
         per-feature decomposition operators expect.  Operators who
         want SHAP on CPU can request it explicitly.
      3. ``method="permutation"`` → force PermutationSHAP regardless
         of hardware (CPU path is the upstream
         ``shap.PermutationExplainer``).
      4. ``method="treeshap"`` (or legacy ``"catboost_treeshap"``) →
         force CatBoost TreeSHAP.  Carries the aggregate-embedding
         interpretation; see module docstring + breadcrumbs in
         ``run_catboost_shap`` and ``_build_feature_groups`` for the
         deferred rework that would let TreeSHAP attribute per-feature.

    Args:
        all_features: ColumnFeatures for each column to explain.
        category_set: For reference embeddings and class lookup.
        method: ``"auto"``, ``"permutation"``, ``"treeshap"``, or the
            legacy aliases ``"embedding_permutation"`` /
            ``"catboost_treeshap"`` (kept for backward compatibility).

    Returns:
        ShapResult or None if no method is available.
    """
    # Normalize legacy aliases.
    if method == "embedding_permutation":
        method = "permutation"
    elif method == "catboost_treeshap":
        method = "treeshap"

    if method == "treeshap":
        # Explicit opt-in to TreeSHAP — honor as requested.
        return _run_treeshap(all_features, category_set)

    if method == "permutation":
        try:
            return run_embedding_shap(all_features, category_set)
        except Exception as e:
            logger.warning("PermutationSHAP failed: %s", e)
            return None

    # method == "auto"
    # TreeSHAP first when a CatBoost model is loaded — now that
    # CatBoost is trained on the structured per-feature input, TreeSHAP
    # produces per-feature attribution natively (no aggregate-
    # "embedding" bucket) and is faster than PermutationSHAP.
    treeshap_result = _run_treeshap(all_features, category_set)
    if treeshap_result is not None:
        return treeshap_result

    # No CatBoost model loaded — fall back to PermutationSHAP.  Use GPU
    # path when available (fast); skip on CPU rather than block.
    try:
        from atelier.classify.gpu import preflight_gpu
        gpu_available = preflight_gpu().available
    except Exception:
        gpu_available = False
    if gpu_available:
        try:
            return run_embedding_shap(all_features, category_set)
        except Exception as e:
            logger.warning("GPU PermutationSHAP failed under method=auto: %s", e)
            return None

    logger.info(
        "SHAP auto: no CatBoost model loaded and no GPU available — "
        "skipping per-item SHAP.  Set classify.shap.method='permutation' "
        "to run the slow CPU PermutationSHAP path explicitly."
    )
    return None


def _run_treeshap(
    all_features: list[ColumnFeatures],
    category_set: HierarchicalCategorySet,
) -> ShapResult | None:
    """Helper: try CatBoost TreeSHAP if a model is loaded; else None.

    Factored out of ``run_shap_analysis`` so the auto-path can fall
    back to it cleanly without duplicating the model-load + predict
    boilerplate.
    """
    try:
        from atelier.classify.ml_inference import get_catboost

        cb = get_catboost()
        if cb is None or cb._model is None:
            logger.debug("CatBoost TreeSHAP requested but no model loaded")
            return None

        # Predict classes via the classifier (which internally encodes
        # the features → structured matrix).
        proba_list = [
            cb.predict_proba_single(f) for f in all_features
        ]
        classes = cb._classes
        predicted_indices = np.array([
            classes.index(max(p, key=p.get)) if p else 0
            for p in proba_list
        ])

        # run_catboost_shap re-encodes internally so SHAP attribution
        # lands on the same structured matrix the classifier saw.
        return run_catboost_shap(cb, all_features, predicted_indices)
    except Exception as e:
        logger.debug("CatBoost TreeSHAP unavailable: %s", e)
        return None
