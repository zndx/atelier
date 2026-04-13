"""Item-wise SHAP feature importance analysis.

Provides per-item (per-column) explanations of classification decisions,
complementing the global SAGE analysis.  Two methods are available:

1. **CatBoost TreeSHAP** — exact O(TLD) algorithm on the CatBoost feature
   space (384-dim embeddings + discrete features).  Fast (~2 seconds for
   50 items), but features are raw dimensions — we group them back to
   interpretable feature groups.

2. **Embedding PermutationSHAP** — ``shap.PermutationExplainer`` on the
   12-feature embedding classifier.  More interpretable but slower;
   reuses ``FeatureMaskModel`` from sage.py.

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
    """
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

    Args:
        model: Fitted CatBoostClassifier (or CatBoostColumnClassifier._model).
        X_eval: (N, n_features) feature matrix used for prediction.
        predicted_indices: (N,) array of predicted class indices.
        emb_dim: Embedding dimension (default 384 for MiniLM-L6).

    Returns:
        ShapResult with grouped feature importance per item.
    """
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

    Uses ``shap.PermutationExplainer`` which is model-agnostic but benefits
    from GPU-accelerated sentence-transformer encoding.  Reuses
    ``FeatureMaskModel`` from sage.py for efficient embedding caching.

    Args:
        all_features: List of ColumnFeatures for evaluated items.
        category_set: Category set with category definitions.
        n_permutations: Number of permutations for the explainer.

    Returns:
        ShapResult with per-item SHAP values on the 12 named features.
    """
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
    """High-level entry point: auto-selects method based on available models.

    "auto" prefers TreeSHAP when a CatBoost model is loaded (fast, ~2s),
    falls back to PermutationSHAP (slower, ~60s on CPU).

    Args:
        all_features: ColumnFeatures for each column to explain.
        category_set: For reference embeddings and class lookup.
        method: "auto", "catboost_treeshap", or "embedding_permutation".

    Returns:
        ShapResult or None if no method is available.
    """
    if method in ("auto", "catboost_treeshap"):
        try:
            from atelier.classify.ml_inference import get_catboost
            from atelier.classify.embedding import embed_texts

            cb = get_catboost()
            if cb is not None and cb._model is not None:
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
            logger.debug("CatBoost TreeSHAP not available: %s", e)
            if method == "catboost_treeshap":
                return None

    if method == "embedding_permutation":
        # Only run PermutationSHAP when explicitly requested — too slow for auto
        try:
            return run_embedding_shap(all_features, category_set)
        except Exception as e:
            logger.warning("Embedding PermutationSHAP failed: %s", e)
    elif method == "auto":
        logger.info("SHAP auto: no CatBoost model loaded, skipping (PermutationSHAP too slow for auto)")

    return None
