# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Factorized NHSVM head for dense pretrained embeddings + runtime adapter.

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

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

    # ──────────────────────────────────────────────────────────────────
    # Persistence: save/load for runtime promotion via the registry
    # ──────────────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Serialize the head to ``directory``.

        Writes:
          - ``state_dict.pt``  — torch state_dict (W parameter + alpha,
            M_alpha, delta buffers)
          - ``head_manifest.json`` — codes (in node order), embed_dim,
            n_nodes — enough to reconstruct an instance without a
            category_set at load time.

        The category_set is NOT required at load time because the head's
        operational state is fully captured by W + the derived buffers.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), d / "state_dict.pt")
        manifest = {
            "codes": list(self.codes),
            "embed_dim": int(self.embed_dim),
            "n_nodes": int(self.n_nodes),
            "_schema_version": 1,
        }
        (d / "head_manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str = "cpu",
    ) -> "FactorizedNHSVMHead":
        """Reconstruct a head from artifacts written by ``save()``.

        Does NOT call ``__init__`` (which would require a category_set).
        Instead reconstructs the module skeleton and loads state_dict
        directly.  The buffers M_alpha, delta, alpha that were derived
        from the category_set at training time are restored verbatim
        from disk.
        """
        d = Path(directory)
        manifest = json.loads((d / "head_manifest.json").read_text())
        if manifest.get("_schema_version") not in (None, 1):
            raise ValueError(
                f"unsupported FactorizedNHSVMHead manifest schema "
                f"{manifest.get('_schema_version')}"
            )

        # Instantiate without going through __init__ to avoid the
        # category_set dependency.  Set up the structural attributes
        # __init__ would normally set, then register the buffers as
        # placeholders that load_state_dict will populate.
        inst = cls.__new__(cls)
        nn.Module.__init__(inst)
        inst.embed_dim = manifest["embed_dim"]
        inst.codes = manifest["codes"]
        inst.code_to_idx = {c: i for i, c in enumerate(inst.codes)}
        inst.n_nodes = manifest["n_nodes"]
        # categories: not reconstructable without the original CategorySet;
        # callers that need category objects should load the category_set
        # separately.  None is the explicit signal.
        inst.categories = None

        inst.register_buffer(
            "alpha", torch.zeros(inst.n_nodes, dtype=torch.float32),
        )
        inst.register_buffer(
            "M_alpha",
            torch.zeros(inst.n_nodes, inst.n_nodes, dtype=torch.float32),
        )
        inst.register_buffer(
            "delta",
            torch.zeros(inst.n_nodes, inst.n_nodes, dtype=torch.float32),
        )
        inst.W = nn.Parameter(
            torch.zeros(inst.n_nodes, inst.embed_dim, dtype=torch.float32),
        )

        state = torch.load(
            d / "state_dict.pt", map_location=device, weights_only=True,
        )
        inst.load_state_dict(state)
        inst = inst.to(device)
        inst.eval()
        return inst


# ──────────────────────────────────────────────────────────────────────
# Runtime adapter — bridges ColumnFeatures input to {code: prob} output
# ──────────────────────────────────────────────────────────────────────


class NHSVMHeadAdapter:
    """Runtime wrapper around a trained ``FactorizedNHSVMHead``.

    Provides the ``predict_proba(features) -> dict[str, float]`` contract
    that ``ml_inference.predict_svm()`` consumers expect at pipeline
    runtime, polymorphic with the legacy ``SVMClassifier``:

      - ``adapter.classes_``     — list of code strings (sklearn-style)
      - ``adapter.predict_proba(features)``  — dict[code, prob] for one row

    Wraps the trained head + encoder identity + training metadata so the
    runtime can pick the right encoder + apply the same input pipeline
    the head was trained against (build_svm_text + sibling enrichment +
    L2-normalize).

    Save layout (under ``directory``):
      - ``state_dict.pt``        — head weights + buffers (via head.save)
      - ``head_manifest.json``   — head structural metadata
      - ``adapter_manifest.json``— encoder_id, training_metadata, etc.
    """

    def __init__(
        self,
        head: FactorizedNHSVMHead,
        encoder_id: str,
        embed_dim: int,
        training_metadata: dict[str, Any] | None = None,
    ):
        self.head = head
        self.encoder_id = encoder_id
        self.embed_dim = embed_dim
        self.training_metadata = training_metadata or {}
        # sklearn-style classes_ for install_svm() polymorphism
        self.classes_ = list(head.codes)

    # ──────────────────────────────────────────────────────────────────
    # Inference: ColumnFeatures → {code: prob}
    # ──────────────────────────────────────────────────────────────────

    def _build_text(self, features: Any) -> str:
        """Recreate the same text shape the head was trained against.

        Mirrors ``atelier.optimize.svm.reflect.build_texts_and_labels``:
        build_svm_text base + " | siblings: a, b, c" enrichment over
        the centered-window k=4 closest siblings.
        """
        from atelier.classify.svm_classifier import build_svm_text
        from atelier.classify.features import _closest_siblings

        column_name = getattr(features, "column_name_humanized", None)
        if column_name is None:
            # Fallback for unusual feature shapes (training side uses
            # Row.column, which is raw_name; here we accept either).
            column_name = getattr(features, "column_name", "")
        column_type = getattr(features, "column_type", None)

        # Sample values: training uses list[str]; ColumnFeatures stores
        # an already-joined string.  Convert back to a list for build_svm_text.
        sample_values_text = getattr(features, "sample_values_text", None)
        if sample_values_text:
            sample_values = [
                v.strip() for v in str(sample_values_text).split(",") if v.strip()
            ]
        else:
            sample_values = []

        base = build_svm_text(column_name, column_type, sample_values)

        # Centered-window siblings (k=4) — same as training-side
        # build_texts_and_labels
        siblings = getattr(features, "sibling_names", []) or []
        top_siblings = _closest_siblings(column_name, siblings, k=4)
        sibling_part = ", ".join(s.replace("_", " ") for s in top_siblings)
        if sibling_part:
            return f"{base} | siblings: {sibling_part}"
        return base

    def _encode(self, text: str) -> np.ndarray:
        """Encode a single text via the adapter's encoder_id.

        Currently dispatches to ``encode_modernbert`` for the only
        supported encoder.  Future encoder IDs can be dispatched here.
        Returns a (1, embed_dim) float32 array.
        """
        if self.encoder_id == "answerdotai/ModernBERT-base":
            # Canonical encoder; loaded-once + cached process-wide, so
            # per-column encoding doesn't reload the model each call.
            from atelier.optimize.svm.encoder import encode_modernbert

            emb = encode_modernbert([text], batch_size=1, pooling="mean")
            return np.asarray(emb, dtype=np.float32)
        raise ValueError(
            f"unsupported encoder_id={self.encoder_id!r}; the adapter "
            f"currently dispatches only 'answerdotai/ModernBERT-base'"
        )

    def _encode_and_predict(self, text: str) -> dict[str, float]:
        """Encode ``text``, L2-normalize, run head, return {code: prob}.

        Shared by both ``predict_proba_features`` and
        ``predict_proba_single`` — the only difference between them is
        whether siblings are folded into the text.
        """
        emb = self._encode(text)  # (1, embed_dim)
        # L2-normalize input — same as training-time normalization in
        # fit_factorized_nhsvm.  Without this, the head's path scores
        # are mis-scaled.
        from sklearn.preprocessing import normalize as sk_normalize
        emb_norm = sk_normalize(emb, norm="l2", axis=1).astype(np.float32)
        device = next(self.head.parameters()).device
        X = torch.as_tensor(emb_norm, device=device)
        # Fitted softmax temperature: a head trained with the (uncalibrated)
        # structured-SVM hinge loss produces cold logits that flatten under
        # the 287-way softmax.  PASS 2 of the synth-primary retrain fits T
        # against a held-out reference slice and writes it here; default
        # 1.0 preserves byte-identical behavior for adapters trained
        # before this calibration step existed (c3cf4fce).
        T = float(self.training_metadata.get("softmax_temperature", 1.0))
        with torch.no_grad():
            proba = self.head.predict_proba(X, temperature=T)  # (1, n_nodes)
        return {code: float(proba[0, i]) for i, code in enumerate(self.head.codes)}

    def predict_proba_features(self, features: Any) -> dict[str, float]:
        """Rich-input inference path.

        Builds the SAME text the head was trained against — including
        centered-window sibling enrichment — then encodes + predicts.
        Used by ``ml_inference.predict_svm`` when the installed model
        exposes this method.

        Returns ``{code: probability}`` for every node in the taxonomy.
        """
        text = self._build_text(features)
        return self._encode_and_predict(text)

    def predict_proba_single(self, text: str) -> dict[str, float]:
        """Back-compat shape: takes a raw text string (no sibling
        enrichment) and returns ``{code: prob}``.

        Mirrors the legacy ``SVMClassifier.predict_proba_single``
        signature so a polymorphic ``install_svm()`` consumer that
        only knows the legacy contract still works against this
        adapter.  Note that this path's text shape will differ from
        the training-time shape (no siblings); accuracy will be
        slightly lower than the features-aware path.  Prefer
        ``predict_proba_features`` when the caller has a
        ``ColumnFeatures`` object.
        """
        return self._encode_and_predict(text)

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> None:
        """Persist the adapter (head + encoder identity + metadata)."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        self.head.save(d)
        manifest = {
            "encoder_id": self.encoder_id,
            "embed_dim": int(self.embed_dim),
            "training_metadata": self.training_metadata,
            "_schema_version": 1,
        }
        (d / "adapter_manifest.json").write_text(json.dumps(manifest, indent=2))

    @classmethod
    def load(
        cls,
        directory: str | Path,
        device: str = "cpu",
    ) -> "NHSVMHeadAdapter":
        """Reconstruct an adapter from artifacts written by ``save()``."""
        d = Path(directory)
        head = FactorizedNHSVMHead.load(d, device=device)
        manifest = json.loads((d / "adapter_manifest.json").read_text())
        if manifest.get("_schema_version") not in (None, 1):
            raise ValueError(
                f"unsupported NHSVMHeadAdapter manifest schema "
                f"{manifest.get('_schema_version')}"
            )
        return cls(
            head=head,
            encoder_id=manifest["encoder_id"],
            embed_dim=manifest["embed_dim"],
            training_metadata=manifest.get("training_metadata", {}),
        )


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
