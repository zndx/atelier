# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Factorized NHSVM head for dense pretrained embeddings.

The original ``HierarchicalFeatureExpander`` materializes the Kronecker
product ``phi(x, y) = sqrt(alpha_y) * (x ⊗ e_y)`` as an explicit
``(n_samples, d * n_nodes)`` matrix and fits a LinearSVC over it.  That
works for sparse text features (TF-IDF char n-grams: only a few hundred
nonzero dims per row) but **catastrophically fails on dense pretrained
embeddings** (ModernBERT mean-pool: every dim nonzero with magnitudes in
similar ranges).  Empirically on the 1149-row reference: TF-IDF best
fit-on-train = 98.93% top-1, ModernBERT best = 4.26% — a structural
mismatch, not a tuning gap.

This module implements the *factorized* form of NHSVM that Choi et al.
2015 used internally: one weight vector ``W_n in R^d`` per hierarchy
node + per-node alpha scalars, with the path score computed as

    gamma(x, y) = sum_{n in A_y} alpha_n * (W_n^T x)

where ``A_y`` is the root-to-y ancestor set (inclusive).  No Kronecker
product is ever materialized.

Training uses the paper's structured-SVM margin objective with
loss-augmented inference, optimized with AdamW (the paper used SGD;
both work).  Inference is a single batched matmul plus argmax over
nodes; non-leaf nodes are first-class prediction targets.

Designed for end-to-end backprop through a frozen or fine-tuned
encoder — for first runs the encoder is frozen and inputs are
precomputed embeddings.

Key tensors:
    W       (n_nodes, d)       learnable per-node weights
    alpha   (n_nodes,)         per-node normalization scalars (frozen)
    M_alpha (n_nodes, n_nodes) precomputed (path indicator) * diag(alpha)
    delta   (n_nodes, n_nodes) precomputed tree-distance loss matrix

The forward pass is:
    node_scores = X @ W.T              # (batch, n_nodes)
    path_scores = node_scores @ M_alpha.T   # (batch, n_nodes)

so the model is mathematically a single (n_nodes, d) linear layer with
the structural prior baked into M_alpha.  SHAP attribution via
LinearExplainer is therefore exact in milliseconds per row.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    epochs_run: int
    final_train_loss: float
    final_train_acc: float
    elapsed_sec: float


class FactorizedNHSVMHead(nn.Module):
    """Per-node-weight NHSVM head over a frozen taxonomy.

    Parameters
    ----------
    category_set : HierarchicalCategorySet
        Must expose ``all_categories``, ``path_from_root(code) -> list``,
        and ``compute_nhsvm_alphas() -> dict[code, float]``.
    embed_dim : int
        Dimensionality of the input embedding (e.g. 768 for ModernBERT-base).
    alphas : dict[str, float] | None
        Per-node normalization scalars.  If None, derived from category_set.
    """

    def __init__(
        self,
        category_set,
        embed_dim: int,
        alphas: dict[str, float] | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.categories = list(category_set.all_categories)
        self.codes: list[str] = [c.code for c in self.categories]
        self.code_to_idx: dict[str, int] = {c: i for i, c in enumerate(self.codes)}
        self.n_nodes = len(self.codes)

        if alphas is None:
            alphas = category_set.compute_nhsvm_alphas()

        # alpha vector, frozen buffer
        alpha_vec = torch.tensor(
            [alphas.get(c, 0.0) for c in self.codes],
            dtype=torch.float32,
        )
        self.register_buffer("alpha", alpha_vec)

        # Path indicator M[y, n] = 1 iff n is ancestor-or-self of y.
        # Then M_alpha = M * alpha (broadcast over rows), so
        # gamma(x, y) = sum_n M_alpha[y, n] * (W_n^T x) = (node_scores @ M_alpha.T)[y]
        M = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float32)
        for y_idx, code in enumerate(self.codes):
            path = list(category_set.path_from_root(code))
            for ancestor in path:
                a_idx = self.code_to_idx.get(ancestor)
                if a_idx is not None:
                    M[y_idx, a_idx] = 1.0
        M_alpha = M * alpha_vec.unsqueeze(0)
        self.register_buffer("M_alpha", M_alpha)

        # Tree-distance loss matrix delta[y, y'] = sqrt(sum alpha_n over
        # symmetric difference of ancestor sets).  Same formulation as
        # ``build_nhsvm_distance_matrix`` in svm_classifier.py.
        ancestor_sets: list[set[str]] = []
        for code in self.codes:
            anc = set(category_set.path_from_root(code))
            ancestor_sets.append(anc)
        D = torch.zeros(self.n_nodes, self.n_nodes, dtype=torch.float32)
        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                sym = ancestor_sets[i] ^ ancestor_sets[j]
                d_sq = sum(alphas.get(n, 0.0) for n in sym)
                d = math.sqrt(max(d_sq, 0.0))
                D[i, j] = d
                D[j, i] = d
        self.register_buffer("delta", D)

        # Per-node learnable weights — standard Linear-layer init
        self.W = nn.Parameter(torch.empty(self.n_nodes, embed_dim))
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Compute path scores for every node, batched.

        Parameters
        ----------
        X : (batch, embed_dim) float tensor

        Returns
        -------
        path_scores : (batch, n_nodes) float tensor
            gamma(x_i, y) for every node y in the taxonomy.
        """
        node_scores = X @ self.W.T            # (batch, n_nodes)
        path_scores = node_scores @ self.M_alpha.T  # (batch, n_nodes)
        return path_scores

    def structured_hinge_loss(
        self,
        X: torch.Tensor,
        y_idx: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Structured-SVM hinge loss with tree-distance margin.

        For each example i with true label y_i, compute

            max_{y'} [gamma(x_i, y') + delta(y', y_i)] - gamma(x_i, y_i)

        clamped at 0 (the augmented max always includes y' = y_i with
        delta = 0, so the difference is non-negative iff some wrong
        node violates the tree-distance margin).

        If ``sample_weight`` is provided (shape ``(batch,)``), per-row
        hinge values are multiplied by the weights before averaging.
        The result is a weighted mean: ``sum(w_i * m_i) / sum(w_i)``,
        so a uniform weight vector reproduces the unweighted mean.
        Default ``None`` preserves identical behavior to the unweighted
        form.
        """
        scores = self(X)                                       # (batch, n_nodes)
        true_scores = scores.gather(1, y_idx.unsqueeze(1)).squeeze(1)
        augmented = scores + self.delta[y_idx]                 # (batch, n_nodes)
        max_violators, _ = augmented.max(dim=1)
        margins = (max_violators - true_scores).clamp(min=0)
        if sample_weight is None:
            return margins.mean()
        # Weighted mean — guard against all-zero-weight batch (shouldn't
        # happen if caller filters weight==0 rows upstream, but be safe).
        w_sum = sample_weight.sum().clamp(min=1e-8)
        return (margins * sample_weight).sum() / w_sum

    def predict_codes(self, X: torch.Tensor) -> list[str]:
        """Argmax over nodes, return list of code strings."""
        with torch.no_grad():
            scores = self(X)
            idx = scores.argmax(dim=1).cpu().numpy().tolist()
        return [self.codes[i] for i in idx]

    def predict_proba(self, X: torch.Tensor, temperature: float = 1.0) -> np.ndarray:
        """Softmax over path scores for a probability-like output."""
        with torch.no_grad():
            scores = self(X) / max(temperature, 1e-6)
            proba = torch.softmax(scores, dim=1).cpu().numpy()
        return proba

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def fit_factorized_nhsvm(
    X: np.ndarray,
    labels: list[str],
    category_set,
    *,
    embed_dim: int | None = None,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str | None = None,
    l2_normalize_input: bool = True,
    eval_every: int = 20,
    verbose: bool = True,
    sample_weights: np.ndarray | None = None,
) -> tuple[FactorizedNHSVMHead, TrainResult]:
    """Train a FactorizedNHSVMHead on precomputed embeddings.

    Inputs:
        X        : (N, d) numpy array of dense embeddings (cached encoder output)
        labels   : list of N node-code strings
        category_set : provides nodes, paths, alphas
        sample_weights : optional (N,) numpy array of per-row weights for the
            structured-SVM hinge.  Weight semantics: w==1.0 means full hinge
            contribution; w==0.5 means half-contribution to the weighted mean;
            w==0.0 rows should be filtered out by the caller (zero weight
            still consumes batch slots without contributing).  ``None``
            (default) preserves identical behavior to the unweighted form.

    Returns the trained head + training metadata.
    """
    import time

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if embed_dim is None:
        embed_dim = int(X.shape[1])
    else:
        assert X.shape[1] == embed_dim, f"X dim {X.shape[1]} != embed_dim {embed_dim}"

    # L2-normalize input (same reasoning as the L2-regularized LinearSVC
    # case: keeps per-dim contributions comparable under weight decay).
    if l2_normalize_input:
        from sklearn.preprocessing import normalize as sk_normalize
        X = sk_normalize(X, norm="l2", axis=1)

    head = FactorizedNHSVMHead(category_set, embed_dim).to(device)
    if verbose:
        log.info(
            "FactorizedNHSVMHead: n_nodes=%d  embed_dim=%d  params=%d  device=%s",
            head.n_nodes, embed_dim, head.num_params(), device,
        )

    # Map label strings to node indices.  Any label whose code isn't in
    # the taxonomy is a data bug — fail loudly.
    missing = [l for l in labels if l not in head.code_to_idx]
    if missing:
        raise ValueError(
            f"{len(missing)} labels are not in category_set.all_categories "
            f"(first: {missing[:5]})"
        )

    X_t = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    y_t = torch.tensor(
        [head.code_to_idx[l] for l in labels],
        dtype=torch.long, device=device,
    )
    w_t: torch.Tensor | None = None
    if sample_weights is not None:
        if len(sample_weights) != len(labels):
            raise ValueError(
                f"sample_weights length {len(sample_weights)} != labels {len(labels)}"
            )
        w_t = torch.tensor(
            np.asarray(sample_weights, dtype=np.float32), device=device,
        )

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=lr, weight_decay=weight_decay,
    )

    n = X_t.shape[0]
    t0 = time.time()
    final_loss = float("nan")
    final_acc = float("nan")
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            w_batch = w_t[idx] if w_t is not None else None
            loss = head.structured_hinge_loss(X_t[idx], y_t[idx], sample_weight=w_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * int(idx.shape[0])
        avg_loss = total / n

        # Capture final-acc regardless of verbose; log per-epoch only if verbose.
        if epoch % eval_every == 0 or epoch == epochs - 1:
            head.eval()
            with torch.no_grad():
                preds = head(X_t).argmax(dim=1)
                acc = (preds == y_t).float().mean().item()
            if verbose:
                log.info("  epoch %3d  loss=%.4f  fit-acc=%.4f",
                         epoch, avg_loss, acc)
            final_loss, final_acc = avg_loss, acc

    elapsed = time.time() - t0
    return head, TrainResult(
        epochs_run=epochs,
        final_train_loss=final_loss,
        final_train_acc=final_acc,
        elapsed_sec=elapsed,
    )
