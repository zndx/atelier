"""Item-wise SHAP feature importance analysis.

Provides per-item (per-column) explanations of classification decisions,
complementing the global SAGE analysis.  Two methods are available:

1. **Embedding PermutationSHAP** (default when GPU available) —
   ``shap.PermutationExplainer`` (or the GPU-accelerated kernel in
   ``gpu_importance.gpu_permutation_shap``) on the 12 named features
   defined in ``features.FEATURE_NAMES`` (column_name, column_type,
   sample_values, cardinality, null_ratio, value_entropy,
   pattern_signals, avg_value_length, numeric_ratio, sibling_context,
   source_table, value_description).  Per-item attributions correspond
   directly to the source features the embedding text composes —
   matches the project's interpretability intent.  ~60s on CPU, fast
   on GPU; reuses ``FeatureMaskModel`` from sage.py.

2. **CatBoost TreeSHAP** (opt-in) — exact O(TLD) algorithm on the
   CatBoost feature space (the 384-dim sentence embedding of the
   concatenated ``embedding_text``, plus any discrete features).
   Fast (~2 seconds for 50 items), BUT the 384 raw embedding
   dimensions don't correspond to the 12 named source features —
   they're a learned compression.  ``_build_feature_groups`` sums
   the 384 SHAP values into a single "embedding" group, which means
   ``shap_top1_name = "embedding"`` for every row with rank-2/3
   empty.  Genuine per-feature TreeSHAP attribution requires
   retraining CatBoost on the 12 features as **native inputs**
   (rather than their concatenated embedding) — a model-architecture
   change deferred to a dedicated session.  Until that lands, this
   path is opt-in only via ``method="catboost_treeshap"``; the
   default-when-CatBoost-loaded behavior was that TreeSHAP wins, but
   that prioritized speed over the interpretive contract.

The auto-selection priority is:

    GPU available  →  Embedding PermutationSHAP (interpretable, fast on GPU)
    No GPU, no CB  →  Skip (CPU PermutationSHAP too slow for default)
    method=opt-in  →  Honored as requested

Operators who want the speed of TreeSHAP and accept the aggregate-
embedding interpretation can request it explicitly via
``classify.shap.method = "treeshap"``.

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
    emb_dim: int, n_discrete: int,
) -> list[tuple[str, int, int]]:
    """Build (name, start, end) slices mapping CatBoost dims to groups.

    CatBoost features layout: [embedding(emb_dim) | discrete(n_discrete)]
    Returns list of (group_name, start_idx, end_idx) where end is exclusive.

    .. note::
        The "embedding" group sums all ``emb_dim`` (384) raw SHAP values
        into one bucket because the current CatBoost classifier is
        trained on the **embedding of the concatenated 12-feature
        text**, not on the 12 features as native inputs.  Per-feature
        TreeSHAP attribution would require retraining CatBoost with
        the 12 features as separate inputs — see the module docstring
        for the deferred rework.  In the interim, callers that want
        per-feature attribution should use ``run_embedding_shap``
        (PermutationSHAP over the 12 features), which is the default
        when GPU is available.
    """  # TODO(deferred): per-feature TreeSHAP requires CatBoost retraining on individual features (see module docstring)
    groups: list[tuple[str, int, int]] = []
    offset = 0

    # Full embedding
    groups.append(("embedding", offset, offset + emb_dim))
    offset += emb_dim

    # Discrete features (individual)
    # These come from CatBoostColumnClassifier which uses 384-dim embedding only
    # Discrete features aren't included in the current CatBoost training,
    # but this is future-proof for when they are.
    if n_discrete > 0:
        discrete_names = [
            "cardinality", "null_ratio", "value_entropy",
            "pattern_email", "pattern_phone", "pattern_ssn",
            "pattern_ipv4", "pattern_uuid", "pattern_date_iso",
            "pattern_url", "pattern_credit_card",
        ]
        for i in range(min(n_discrete, len(discrete_names))):
            groups.append((discrete_names[i], offset + i, offset + i + 1))
        offset += n_discrete

    return groups


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
    model,
    X_eval: np.ndarray,
    predicted_indices: np.ndarray,
    emb_dim: int = 384,
) -> ShapResult:
    """Compute item-wise SHAP values using CatBoost's built-in TreeSHAP.

    .. note::
        This path attributes to the CatBoost model's native input space
        (the 384-dim sentence embedding of the concatenated
        ``embedding_text``, plus any discrete features).  The 384
        embedding dimensions are a **learned compression** of the 12
        named source features and do not correspond to them
        individually — ``_build_feature_groups`` sums them into one
        "embedding" group, which is why ``shap_top1_name = "embedding"``
        on every row with rank-2/3 empty.  Genuine per-feature
        TreeSHAP attribution requires retraining CatBoost on the 12
        features as **native inputs** rather than their concatenated
        embedding — a model-architecture change deferred to a
        dedicated session.  For per-feature attribution today, use
        ``run_embedding_shap`` (PermutationSHAP).

    Args:
        model: Fitted CatBoostClassifier (or CatBoostColumnClassifier._model).
        X_eval: (N, n_features) feature matrix used for prediction.
        predicted_indices: (N,) array of predicted class indices.
        emb_dim: Embedding dimension (default 384 for MiniLM-L6).

    Returns:
        ShapResult with grouped feature importance per item.
    """  # TODO(deferred): production-ready per-feature TreeSHAP requires CatBoost retraining on individual features (see module docstring)
    from catboost import Pool

    t0 = time.time()
    N, total_dim = X_eval.shape

    pool = Pool(X_eval)

    # CatBoost ShapValues: shape depends on classification type
    raw_shap = model.get_feature_importance(
        type="ShapValues",
        data=pool,
    )

    n_feat = total_dim

    # CatBoost ShapValues shape:
    #   MultiClass: (N, n_classes, n_features+1)
    #   Binary:     (N, n_features+1)
    if raw_shap.ndim == 3:
        # Extract SHAP values for each item's predicted class
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

    # Group raw dims into interpretable feature groups
    n_discrete = max(0, total_dim - emb_dim)
    groups = _build_feature_groups(emb_dim, n_discrete)

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
    try:
        from atelier.classify.gpu import preflight_gpu
        gpu_available = preflight_gpu().available
    except Exception:
        gpu_available = False

    if gpu_available:
        try:
            return run_embedding_shap(all_features, category_set)
        except Exception as e:
            logger.warning(
                "GPU PermutationSHAP failed under method=auto, falling back "
                "to TreeSHAP if a CatBoost model is loaded: %s", e,
            )
            return _run_treeshap(all_features, category_set)

    # No GPU under auto: skip rather than block on slow CPU PermutationSHAP
    # or land the TreeSHAP single-"embedding" attribution by default.
    logger.info(
        "SHAP auto: no GPU available — skipping per-item SHAP.  Set "
        "classify.shap.method='permutation' to run the slow CPU path "
        "explicitly, or 'treeshap' for the aggregate-embedding "
        "interpretation."
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
        from atelier.classify.embedding import embed_texts

        cb = get_catboost()
        if cb is None or cb._model is None:
            logger.debug("CatBoost TreeSHAP requested but no model loaded")
            return None

        # Build the evaluation matrix (same as predict_catboost)
        texts = [f.to_embedding_text() for f in all_features]
        X_eval = np.array(embed_texts(texts))

        # Get predicted class indices
        proba_list = [cb.predict_proba_single(X_eval[i]) for i in range(len(all_features))]
        classes = cb._classes
        predicted_indices = np.array([
            classes.index(max(p, key=p.get)) if p else 0
            for p in proba_list
        ])

        return run_catboost_shap(
            cb._model, X_eval, predicted_indices,
            emb_dim=X_eval.shape[1],
        )
    except Exception as e:
        logger.debug("CatBoost TreeSHAP unavailable: %s", e)
        return None
