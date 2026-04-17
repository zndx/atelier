"""SAGE feature importance analysis for the classification pipeline.

SAGE (Shapley Additive Global importancE) quantifies how much each
evidence source contributes to classification decisions. This is a
critical pipeline component — not optional — gated by config only to
manage runtime cost during development and interactive UI testing.

Wraps the cosine similarity classifier as a SAGE model function and
computes marginal feature contributions via permutation unrolling.

Ported from signals/src/sigint/sage_analysis.py, adapted for atelier's
embedding.py and 12-feature ColumnFeatures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from atelier.classify.features import FEATURE_NAMES, ColumnFeatures

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SageResult:
    feature_names: list[str]
    importance_values: list[float]
    importance_std: list[float]
    n_samples: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "importance_values": self.importance_values,
            "importance_std": self.importance_std,
            "n_samples": self.n_samples,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }

    def ranked(self) -> list[tuple[str, float]]:
        """Features sorted by |importance| descending."""
        pairs = list(zip(self.feature_names, self.importance_values))
        pairs.sort(key=lambda p: -abs(p[1]))
        return pairs


class FeatureMaskModel:
    """Wraps atelier's cosine classifier as a SAGE model function.

    SAGE marginalizes features by swapping values between samples.
    Uses LRU cache for embedding re-encoding efficiency.

    X[i, j] = index into per-feature value lookup table.
    When SAGE removes feature j from sample i, it substitutes
    sample k's value — asking "what if this column had a different
    name but the same values?"
    """

    def __init__(
        self,
        all_features: list[ColumnFeatures],
        category_set,
        cache_size: int = 4096,
    ):
        self.all_features = all_features
        self.category_set = category_set
        self.n_features = len(FEATURE_NAMES)

        # Build per-feature value lookup tables
        self.feature_values: list[list[str]] = []
        for fname in FEATURE_NAMES:
            vals = [feat.feature_value(fname) for feat in all_features]
            self.feature_values.append(vals)

        # Precompute reference category embeddings
        from atelier.classify.embedding import embed_texts
        ref_texts = [cat.embedding_text for cat in category_set.categories]
        self._ref_embs = np.array(embed_texts(ref_texts))  # (n_classes, dim)
        self._ref_codes = [cat.code for cat in category_set.categories]

        # LRU embedding cache
        self._cache: dict[str, np.ndarray] = {}
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """(N, 12) feature index array → (N, n_classes) probabilities."""
        from atelier.classify.embedding import _get_model

        N = X.shape[0]
        dim = self._ref_embs.shape[1]
        model = _get_model()

        # Build embedding texts by mixing feature values per X indices
        texts: list[str] = []
        for i in range(N):
            parts = []
            for j, fname in enumerate(FEATURE_NAMES):
                idx = int(X[i, j])
                val = self.feature_values[j][idx]
                if val:
                    parts.append(val)
            texts.append(" | ".join(parts) if parts else "")

        # Smart caching: within-batch dedup + LRU
        vecs = np.zeros((N, dim))
        pending: dict[str, list[int]] = {}

        for i, text in enumerate(texts):
            if text in self._cache:
                vecs[i] = self._cache[text]
                self.cache_hits += 1
            elif text in pending:
                pending[text].append(i)
                self.cache_hits += 1
            else:
                pending[text] = [i]
                self.cache_misses += 1

        # Batch encode only unique uncached texts
        if pending:
            unique_texts = list(pending.keys())
            from atelier.classify.embedding import get_batch_size
            raw = model.encode(unique_texts, normalize_embeddings=True,
                               show_progress_bar=False, batch_size=get_batch_size())
            for j, text in enumerate(unique_texts):
                vec = raw[j]
                for idx in pending[text]:
                    vecs[idx] = vec
                if len(self._cache) < self._cache_size:
                    self._cache[text] = vec

        # Cosine similarities + softmax to probabilities
        sims = vecs @ self._ref_embs.T  # (N, n_classes)
        exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)

        return probs


def run_sage_analysis(
    all_features: list[ColumnFeatures],
    ground_truth_indices: np.ndarray,
    category_set,
    *,
    n_permutations: int = 512,
    detect_convergence: bool = True,
) -> SageResult:
    """Run SAGE feature importance analysis.

    Dispatches to the vectorized GPU kernel
    (:func:`atelier.classify.gpu_importance.gpu_sage`) when CUDA is
    available.  Falls back to the upstream ``sage-importance`` library
    on CPU — the CPU path is slow (many minutes on a large corpus) and
    is kept only for environments without a GPU.

    Args:
        all_features: ColumnFeatures for each sample.
        ground_truth_indices: (N,) integer class indices.
        category_set: For reference embeddings.
        n_permutations: Max permutations (may stop earlier if converged).
        detect_convergence: Auto-stop when SAGE values stabilize.

    Returns:
        SageResult with per-feature importance and std.
    """
    # GPU path — vectorized kernel + fixed global donors.
    try:
        from atelier.classify.gpu import preflight_gpu
        if preflight_gpu().available:
            from atelier.classify.gpu_importance import gpu_sage
            return gpu_sage(
                all_features, ground_truth_indices, category_set,
                n_permutations=n_permutations,
                detect_convergence=detect_convergence,
            )
    except Exception as exc:
        logger.warning("GPU SAGE unavailable, falling back to CPU: %s", exc)

    import sage

    t0 = time.time()
    N = len(all_features)
    n_feat = len(FEATURE_NAMES)

    # Build feature index matrix: X[i, j] = i (each sample starts with own values)
    X = np.tile(np.arange(N).reshape(-1, 1), (1, n_feat))
    Y = ground_truth_indices.astype(int)

    # Wrap classifier for SAGE
    model_fn = FeatureMaskModel(all_features, category_set)

    # MarginalImputer: marginalizes by substituting from data distribution
    imputer = sage.MarginalImputer(model_fn, X)

    # PermutationEstimator: SAGE values via permutation unrolling
    estimator = sage.PermutationEstimator(imputer, loss="cross entropy")

    logger.info("Running SAGE with %d permutations on %d samples", n_permutations, N)
    explanation = estimator(
        X, Y,
        n_permutations=n_permutations,
        detect_convergence=detect_convergence,
        bar=False,
    )

    elapsed = time.time() - t0
    logger.info("SAGE completed in %.1fs", elapsed)

    return SageResult(
        feature_names=list(FEATURE_NAMES),
        importance_values=[round(float(v), 6) for v in explanation.values],
        importance_std=[round(float(s), 6) for s in explanation.std],
        n_samples=N,
        elapsed_seconds=elapsed,
    )
