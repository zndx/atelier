"""GPU-accelerated SAGE and PermutationSHAP feature importance.

Replaces the ``sage-importance`` Python permutation loop (and its SHAP
PermutationExplainer cousin) with a chunk-batched, cache-backed GPU
kernel that targets the cosine classifier's ``FeatureMaskModel``.

The *only* component that touches the embedding model is a single batch
encode per chunk — every other step is vectorized numpy or torch matmul.

Algorithm (SAGE)::

  precompute per-feature text lookup T[f, i] = feature_value(f, sample i)
  precompute reference embeddings R[c, :] for every category c
  for each chunk of permutations σ[0..P], donor indices D[P, N, F]:
    for step k ∈ [0, F]:
      build row-contexts (" | " joined piece texts for present + donor)
      dedupe across (p, k, i), encode unique set on GPU in one call
    compute all losses L[p, k, i] via one torch matmul + log_softmax
    attribute Δ_k to feature σ[p, k-1]; accumulate mean + sq for variance
  stop when per-feature variance drops below convergence bar

Outputs are the same :class:`SageResult` / :class:`ShapResult` shapes as
the upstream libraries so callers stay source-compatible.  Graceful
fallback: when ``preflight_gpu()`` reports no CUDA, callers should not
route into this module — they keep using ``sage-importance``.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from atelier.classify.features import FEATURE_NAMES, ColumnFeatures

if TYPE_CHECKING:
    from atelier.classify.sage import SageResult
    from atelier.classify.shap_explanations import ShapResult

logger = logging.getLogger(__name__)

_F = len(FEATURE_NAMES)


# ── Shared context precomputation ────────────────────────────────


class _SharedContext:
    """Precomputed tables common to GPU SAGE and GPU PermutationSHAP.

    Building these once and sharing across estimators avoids redundant
    work when a single pipeline run computes both.
    """

    def __init__(self, all_features: list[ColumnFeatures], category_set):
        self.all_features = all_features
        self.category_set = category_set
        self.N = len(all_features)
        self.F = _F

        # T[f][i] = text contribution of feature f from sample i
        self.text_lookup: list[list[str]] = [
            [feat.feature_value(FEATURE_NAMES[f]) for feat in all_features]
            for f in range(_F)
        ]

        # Reference category embeddings (precomputed once, cached on GPU)
        from atelier.classify.embedding import _get_model
        encoder = _get_model()
        ref_texts = [cat.embedding_text for cat in category_set.categories]
        self._ref_codes = [cat.code for cat in category_set.categories]
        R = np.asarray(
            encoder.encode(
                ref_texts, normalize_embeddings=True, batch_size=256,
            ),
            dtype=np.float32,
        )
        self.n_classes = R.shape[0]
        self.dim = R.shape[1]
        self._R_np = R

        # Move reference embeddings onto GPU once.  Stored as fp16 when the
        # device supports it to halve VRAM for the big GEMM, but fp32 for
        # accuracy matters more than the savings at 384 dims × 316 classes.
        import torch
        self._torch = torch
        self._device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self._R = torch.from_numpy(R).to(self._device)

        # Persistent text → embedding cache (reused across chunks).  Memory-
        # bounded by default; evicts oldest entries when full.
        self._cache: dict[str, np.ndarray] = {}
        self._cache_capacity = 1_000_000  # plenty of headroom for synth

    # ── Text cache ────────────────────────────────────────────

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return (len(texts), dim) fp32 numpy matrix.  Caches results."""
        from atelier.classify.embedding import _get_model

        encoder = _get_model()

        # Partition: known vs novel
        out = [None] * len(texts)
        novel: list[str] = []
        novel_positions: dict[str, list[int]] = {}
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                out[i] = v
                continue
            if t in novel_positions:
                novel_positions[t].append(i)
            else:
                novel.append(t)
                novel_positions[t] = [i]

        if novel:
            batch = np.asarray(
                encoder.encode(
                    novel, normalize_embeddings=True, batch_size=256,
                ),
                dtype=np.float32,
            )
            for j, t in enumerate(novel):
                v = batch[j]
                if len(self._cache) < self._cache_capacity:
                    self._cache[t] = v
                for pos in novel_positions[t]:
                    out[pos] = v

        return np.stack(out, axis=0)

    # ── Loss helper: softmax cross-entropy against gt_indices ────

    def losses(
        self, embeddings: np.ndarray, gt_indices: np.ndarray,
    ) -> np.ndarray:
        """Compute cross-entropy loss per row on GPU.

        embeddings: (M, dim) np.float32
        gt_indices: (M,)      np.int64

        Returns: (M,) np.float32 of -log p(gt | embedding).
        """
        torch = self._torch
        emb_t = torch.from_numpy(embeddings).to(self._device)
        logits = emb_t @ self._R.T  # (M, n_classes)
        logp = torch.log_softmax(logits, dim=-1)
        gt_t = torch.from_numpy(gt_indices.astype(np.int64)).to(self._device)
        gathered = logp.gather(1, gt_t.unsqueeze(1)).squeeze(1)
        return (-gathered).cpu().numpy()


# ── Joined-text builder ─────────────────────────────────────────


def _build_context_texts(
    ctx: _SharedContext,
    sigma: np.ndarray,         # (chunk, F) permutations (feature-order arrays)
    donors: np.ndarray,        # (chunk, N, F) donor sample indices when marginalized
) -> tuple[list[str], np.ndarray]:
    """Build all (chunk * (F+1) * N) context texts; dedupe and index them.

    Matches the upstream sage-importance text contract: texts are joined
    in the FIXED ``FEATURE_NAMES`` order (not the permutation order).  The
    permutation σ only determines *which* feature draws from sample i's
    own value vs. a donor's value at each step.

    Returns:
        unique_texts: list of unique text strings (dedupe order)
        idx_map:      (chunk, F+1, N) int offsets into ``unique_texts``
    """
    chunk = sigma.shape[0]
    N = ctx.N
    F = _F

    # inv_sigma[p, f] = position (rank) of feature f in permutation σ[p].
    inv_sigma = np.argsort(sigma, axis=1).astype(np.int64)   # (chunk, F)

    # is_present[p, k, f] = True when feature f uses sample i's own value at
    # step k (i.e. f appears among σ[p, :k]).
    k_range = np.arange(F + 1, dtype=np.int64)
    is_present = inv_sigma[:, None, :] < k_range[None, :, None]   # (chunk, F+1, F)

    # Source sample index per (p, k, i, f):
    #   present → i
    #   absent  → donors[p, i, f]
    i_range = np.arange(N, dtype=np.int64)
    src = np.where(
        is_present[:, :, None, :],                               # (chunk, F+1, 1, F)
        i_range[None, None, :, None],                             # (1, 1, N, 1)
        donors[:, None, :, :].astype(np.int64),                   # (chunk, 1, N, F)
    )                                                             # (chunk, F+1, N, F)

    # Flatten and dedupe by fingerprint (src-vector).  ``np.unique`` on the
    # row axis gives us both the canonical index map (``inverse``) and the
    # compact list of unique fingerprints — no Python dedup loop needed.
    flat = src.reshape(-1, F)                                     # (M, F)
    unique_fps, inverse = np.unique(flat, axis=0, return_inverse=True)
    idx_map = inverse.reshape(chunk, F + 1, N).astype(np.int64)

    # Build texts for each unique fingerprint.  This is the only Python
    # hot loop remaining: ~|U| iterations, each joining ≤ F pieces.
    T = ctx.text_lookup                                           # list[list[str]]
    unique_texts: list[str] = []
    for row in unique_fps:
        parts: list[str] = []
        for f in range(F):
            v = T[f][int(row[f])]
            if v:
                parts.append(v)
        unique_texts.append(" | ".join(parts) if parts else "")

    return unique_texts, idx_map


# ── SAGE ────────────────────────────────────────────────────────


def gpu_sage(
    all_features: list[ColumnFeatures],
    ground_truth_indices: np.ndarray,
    category_set,
    *,
    n_permutations: int = 512,
    chunk_size: int = 16,
    detect_convergence: bool = True,
    convergence_threshold: float = 0.025,
    seed: int = 42,
) -> "SageResult":
    """GPU-accelerated SAGE.  Mirrors sage-importance math; fast path.

    Output is shape-compatible with :class:`atelier.classify.sage.SageResult`
    so callers in ``sage.run_sage_analysis`` can swap in the GPU kernel
    without changing their result handling.
    """
    from atelier.classify.sage import SageResult  # avoid circular import

    t0 = time.time()
    ctx = _SharedContext(all_features, category_set)

    rng = np.random.default_rng(seed)
    accum = np.zeros((_F,), dtype=np.float64)
    accum_sq = np.zeros((_F,), dtype=np.float64)
    n_done = 0

    # Fixed global donors (one per feature) — vastly improves cross-chunk
    # dedup and caches most contexts after the first ~100 permutations.
    # With fixed donors, unique contexts are bounded by (2^F × N); in
    # practice, permutation sampling visits only the masks that appear on
    # actual σ chains, which amortizes quickly.
    donor_per_f = rng.integers(0, ctx.N, size=(_F,))

    while n_done < n_permutations:
        chunk = min(chunk_size, n_permutations - n_done)
        sigma = np.stack([rng.permutation(_F) for _ in range(chunk)], axis=0)
        donors = np.broadcast_to(
            donor_per_f[None, None, :], (chunk, ctx.N, _F),
        ).copy()                                             # (chunk, N, F)

        unique_texts, idx_map = _build_context_texts(ctx, sigma, donors)

        # Single encode for this chunk's unique contexts.
        embeddings = ctx.embed(unique_texts)

        # Flat loss computation on GPU.
        flat_idx = idx_map.reshape(-1)                      # (chunk*(F+1)*N,)
        flat_emb = embeddings[flat_idx]                     # (M, dim)
        gt_rep = np.tile(ground_truth_indices, chunk * (_F + 1))
        L = ctx.losses(flat_emb, gt_rep)                     # (M,)
        L = L.reshape(chunk, _F + 1, ctx.N).astype(np.float64)

        # Per-permutation SAGE contributions: Δ_k = L_{k-1} − L_k,
        # attributed to the feature that became present at step k (σ[p, k-1]).
        # Mean over samples → scalar contribution for this permutation.
        for p in range(chunk):
            deltas = (L[p, :-1] - L[p, 1:]).mean(axis=1)     # (F,)
            # deltas[j] is the contribution at step j+1, attributed to σ[p, j]
            for j in range(_F):
                f = int(sigma[p, j])
                accum[f] += deltas[j]
                accum_sq[f] += deltas[j] ** 2

        n_done += chunk

        # Convergence check: coefficient of variation across permutations
        if detect_convergence and n_done >= 32:
            mean = accum / n_done
            var = accum_sq / n_done - mean ** 2
            var = np.maximum(var, 0.0)
            stderr = np.sqrt(var / n_done)
            max_cv = float(np.max(stderr / (np.abs(mean) + 1e-9)))
            if max_cv < convergence_threshold:
                logger.info(
                    "GPU SAGE converged after %d/%d permutations (max CoV %.3f < %.3f)",
                    n_done, n_permutations, max_cv, convergence_threshold,
                )
                break

    elapsed = time.time() - t0
    mean = accum / max(n_done, 1)
    var = accum_sq / max(n_done, 1) - mean ** 2
    stds = np.sqrt(np.maximum(var, 0.0) / max(n_done, 1))

    logger.info(
        "GPU SAGE: N=%d, F=%d, %d permutations, %d unique contexts, %.1fs",
        ctx.N, _F, n_done, len(ctx._cache), elapsed,
    )

    return SageResult(
        feature_names=list(FEATURE_NAMES),
        importance_values=[round(float(v), 6) for v in mean],
        importance_std=[round(float(s), 6) for s in stds],
        n_samples=ctx.N,
        elapsed_seconds=elapsed,
    )


# ── PermutationSHAP ─────────────────────────────────────────────


def gpu_permutation_shap(
    all_features: list[ColumnFeatures],
    category_set,
    *,
    n_permutations: int = 64,
    chunk_size: int = 16,
    seed: int = 42,
) -> "ShapResult":
    """GPU-accelerated per-item PermutationSHAP on the 12-feature classifier.

    Same permutation machinery as :func:`gpu_sage`, but attribution is
    *per-item* rather than marginalized across the sample axis.  Output
    is shape-compatible with :class:`atelier.classify.shap_explanations.ShapResult`.
    """
    from atelier.classify.shap_explanations import ShapResult

    t0 = time.time()
    ctx = _SharedContext(all_features, category_set)

    # For per-item SHAP, ground truth = the predicted class under the
    # full-context embedding (same convention as the upstream library).
    full_texts = [f.to_embedding_text() for f in all_features]
    full_emb = ctx.embed(full_texts)
    full_logits = full_emb @ ctx._R_np.T
    pred = np.argmax(full_logits, axis=-1)     # (N,)
    rng = np.random.default_rng(seed)

    # Accumulate (N, F) SHAP values; we attribute Δ_k per item.
    shap_sum = np.zeros((ctx.N, _F), dtype=np.float64)
    shap_count = np.zeros((_F,), dtype=np.int64)
    L = np.zeros((1, _F + 1, ctx.N), dtype=np.float64)  # init for type-checker

    n_done = 0
    donor_per_f = rng.integers(0, ctx.N, size=(_F,))

    while n_done < n_permutations:
        chunk = min(chunk_size, n_permutations - n_done)
        sigma = np.stack([rng.permutation(_F) for _ in range(chunk)], axis=0)
        donors = np.broadcast_to(
            donor_per_f[None, None, :], (chunk, ctx.N, _F),
        ).copy()

        unique_texts, idx_map = _build_context_texts(ctx, sigma, donors)
        embeddings = ctx.embed(unique_texts)

        flat_idx = idx_map.reshape(-1)
        flat_emb = embeddings[flat_idx]
        gt_rep = np.tile(pred, chunk * (_F + 1))
        L_flat = ctx.losses(flat_emb, gt_rep)
        L = L_flat.reshape(chunk, _F + 1, ctx.N).astype(np.float64)

        # Per-item attribution: deltas[p, k, i] = L[p, k-1, i] - L[p, k, i]
        # attributed to feature σ[p, k-1] for item i.
        deltas = L[:, :-1, :] - L[:, 1:, :]             # (chunk, F, N)
        for pi in range(chunk):
            for ji in range(_F):
                f = int(sigma[pi, ji])
                shap_sum[:, f] += deltas[pi, ji, :]
                shap_count[f] += 1

        n_done += chunk

    elapsed = time.time() - t0

    shap_values = shap_sum / np.maximum(shap_count, 1)[None, :]
    # Negate because our "loss" formulation measures unexplained mass —
    # SHAP convention is +ve = increases the predicted-class score.
    shap_values = -shap_values

    base_value = float(np.mean(L[:, 0, :]))              # baseline loss
    logger.info(
        "GPU PermutationSHAP: N=%d, F=%d, %d permutations, %.1fs",
        ctx.N, _F, n_done, elapsed,
    )

    return ShapResult(
        feature_names=list(FEATURE_NAMES),
        shap_values=shap_values.astype(np.float32),
        base_value=base_value,
        method="gpu_permutation",
        n_items=ctx.N,
        elapsed_seconds=elapsed,
    )
