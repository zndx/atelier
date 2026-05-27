"""Temperature scaling + ECE + mass diagnostics for the NHSVM head.

Pure library — no GPU/encoder dependencies.  Imported by the training
script (``promote_t1_head_v2.py``) and the k-fold validator
(``reflect_nhsvm_eval_shap_v2.py``) to fit a calibrated
``softmax_temperature`` against a held-out reference slice and report
mass-magnitude diagnostics.

Why this exists
---------------

The factorized NHSVM head is trained with structured-SVM hinge loss
(``factorized_nhsvm.py:158-191``) — margin-driven, not probability-
calibrated.  Under the 287-way softmax, hinge-trained logits flatten
naturally: even a large margin between top-1 and second produces top1
probabilities in the 0.02-0.04 range.  Phase 1 of the calibration work
(2026-05-26) needed a post-hoc ``α_svm=15`` multiplier to expose the
ranking signal; temperature scaling solves the problem at the right
layer — pre-softmax — using validation data rather than an arbitrary
multiplier.

Reference: Guo et al. (2017), *On Calibration of Modern Neural
Networks* (https://arxiv.org/abs/1706.04599).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Default grid for temperature search.  Range justified by:
# - T = 1.0: hinge-default (no scaling)
# - T < 1.0: sharper softmax (recover hinge margin signal)
# - T > 1.0: smoother softmax (rarely useful for hinge-trained heads)
DEFAULT_T_GRID: tuple[float, ...] = (
    0.05, 0.07, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00, 1.50, 2.00,
)

# Minimum temperature — preserves numeric stability.  T=0.05 corresponds
# to ~20× sharper logits; below this, gradients collapse and the
# softmax pins to a delta-distribution that overfits the holdout.
T_FLOOR = 0.05


def _softmax_at_T(logits: np.ndarray, T: float) -> np.ndarray:
    """Numerically stable softmax with temperature scaling.

    ``logits`` shape: (N, n_classes).  Subtracts row-wise max for
    stability before exponentiation.
    """
    scaled = logits / max(float(T), T_FLOOR)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll_at_T(logits: np.ndarray, y_idx: np.ndarray, T: float) -> float:
    """Negative log-likelihood (mean over samples) at temperature T."""
    proba = _softmax_at_T(logits, T)
    # Clip to avoid log(0) on very confident wrong predictions.
    p_y = np.clip(proba[np.arange(len(y_idx)), y_idx], 1e-12, 1.0)
    return float(-np.log(p_y).mean())


def fit_softmax_temperature(
    logits: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    T_grid: tuple[float, ...] = DEFAULT_T_GRID,
    refine: bool = True,
    refine_tol: float = 1e-3,
) -> dict[str, Any]:
    """Fit softmax temperature T to minimize NLL on a held-out slice.

    Two-stage: grid search over ``T_grid`` for robustness, then
    optional scalar refinement via ``scipy.optimize.minimize_scalar``
    bracketing the grid winner.  Floors at ``T_FLOOR`` to prevent
    overfit-collapse on small holdouts.

    Args:
        logits: ``(N, n_classes)`` raw pre-softmax scores from the head.
            Use :func:`head_logits_batched` to compute these.
        y_true_idx: ``(N,)`` integer label indices into the head's
            ``codes`` list.  Must align with the logits' column order.
        T_grid: candidate temperatures for the coarse grid sweep.
        refine: when True, locally refine the grid winner via
            scipy.optimize for sub-grid resolution.
        refine_tol: convergence tolerance for the local refinement.

    Returns:
        Dict with keys:
            - ``best_T``: float, the fitted temperature
            - ``nll_at_best``: float, validation NLL at best_T
            - ``nll_at_T1``: float, baseline NLL at T=1.0 (no scaling)
            - ``ece_at_best``: float, ECE at best_T (15 bins)
            - ``ece_at_T1``: float, ECE at T=1.0
            - ``improvement``: float, ``nll_at_T1 - nll_at_best``
              (positive means calibration helped)
            - ``grid_results``: list[dict] with per-T NLL for diagnostic
            - ``n_calibration``: int, holdout sample count

    Raises:
        ValueError: if logits/y_true_idx shapes mismatch or grid is
            empty.
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D; got {logits.shape}")
    if y_true_idx.shape != (logits.shape[0],):
        raise ValueError(
            f"y_true_idx shape {y_true_idx.shape} must match "
            f"logits rows {logits.shape[0]}"
        )
    if not T_grid:
        raise ValueError("T_grid must be non-empty")

    grid_results = []
    best = {"T": T_grid[0], "nll": float("inf")}
    for T in T_grid:
        nll = _nll_at_T(logits, y_true_idx, T)
        grid_results.append({"T": float(T), "nll": nll})
        if nll < best["nll"]:
            best = {"T": float(T), "nll": nll}

    final_T = best["T"]
    final_nll = best["nll"]

    if refine:
        try:
            from scipy.optimize import minimize_scalar
            # Bracket around the grid winner.  Pick neighbors from
            # the sorted grid for a sound bracket.
            sorted_grid = sorted(T_grid)
            idx = sorted_grid.index(best["T"])
            lo = sorted_grid[max(0, idx - 1)]
            hi = sorted_grid[min(len(sorted_grid) - 1, idx + 1)]
            if lo == hi:
                lo, hi = max(T_FLOOR, best["T"] * 0.7), best["T"] * 1.3
            result = minimize_scalar(
                lambda T: _nll_at_T(logits, y_true_idx, max(T, T_FLOOR)),
                bracket=(lo, best["T"], hi),
                method="brent",
                options={"xtol": refine_tol},
            )
            if result.success and result.fun < final_nll:
                final_T = max(float(result.x), T_FLOOR)
                final_nll = float(result.fun)
        except Exception as e:
            logger.debug("scalar refinement failed (using grid winner): %s", e)

    nll_at_T1 = _nll_at_T(logits, y_true_idx, 1.0)
    proba_best = _softmax_at_T(logits, final_T)
    proba_T1 = _softmax_at_T(logits, 1.0)
    ece_at_best = expected_calibration_error(proba_best, y_true_idx)
    ece_at_T1 = expected_calibration_error(proba_T1, y_true_idx)

    improvement = nll_at_T1 - final_nll
    if improvement < 0.05:
        logger.warning(
            "softmax temperature calibration improvement = %.4f "
            "(< 0.05 threshold) — leaving T=1.0 may be safer. "
            "Caller should consider the per-head metadata 'softmax_temperature' decision.",
            improvement,
        )

    return {
        "best_T": float(final_T),
        "nll_at_best": float(final_nll),
        "nll_at_T1": float(nll_at_T1),
        "ece_at_best": float(ece_at_best),
        "ece_at_T1": float(ece_at_T1),
        "improvement": float(improvement),
        "grid_results": grid_results,
        "n_calibration": int(len(y_true_idx)),
    }


def expected_calibration_error(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Compute ECE: weighted average gap between confidence and accuracy per bin.

    Standard 15-bin ECE (Guo et al. 2017).  For each sample, top-1
    confidence = max(proba) and top-1 prediction = argmax(proba).
    Samples binned by confidence; per-bin gap = |mean_confidence -
    accuracy|; ECE = sum_b (n_b / N) * gap_b.

    Args:
        proba: ``(N, n_classes)`` probability distributions (rows sum
            to 1).
        y_true_idx: ``(N,)`` integer label indices.
        n_bins: number of confidence bins (default 15).

    Returns:
        ECE in [0, 1].  Lower is better; 0.10 is "well-calibrated"
        territory, > 0.20 indicates substantial miscalibration.
    """
    if proba.shape[0] != y_true_idx.shape[0]:
        raise ValueError("proba and y_true_idx row counts must match")
    confidences = proba.max(axis=1)
    predictions = proba.argmax(axis=1)
    accuracies = (predictions == y_true_idx).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin includes the upper boundary
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (n_b / n) * abs(bin_conf - bin_acc)
    return float(ece)


def head_logits_batched(
    head: "Any",
    X: np.ndarray,
    *,
    batch_size: int = 512,
    device: "Any" = None,
) -> np.ndarray:
    """Compute raw path scores (pre-softmax logits) for a batch of inputs.

    Uses ``head.forward`` rather than ``head.predict_proba`` so the
    softmax/temperature transformation can be controlled by the caller.
    Memory-efficient: chunks inputs into ``batch_size`` slices.

    Args:
        head: ``FactorizedNHSVMHead`` (must expose ``forward(X)``
            returning path scores of shape ``(N, n_nodes)``).
        X: ``(N, embed_dim)`` L2-normalized embeddings (same shape
            ``head.forward`` expects).
        batch_size: rows per chunk through the head.
        device: optional torch device.  If None, uses the head's
            parameter device.

    Returns:
        ``(N, n_nodes)`` numpy array of raw scores in the same column
        order as ``head.codes``.
    """
    import torch

    if device is None:
        device = next(head.parameters()).device

    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    out = []
    head.eval()
    with torch.no_grad():
        for start in range(0, X_t.shape[0], batch_size):
            chunk = X_t[start : start + batch_size]
            scores = head.forward(chunk)  # (chunk, n_nodes)
            out.append(scores.cpu().numpy())
    return np.concatenate(out, axis=0)


def mass_metrics_at_T(
    logits: np.ndarray,
    y_true_idx: np.ndarray,
    T: float,
) -> dict[str, float]:
    """Compute raw top1 mass statistics at temperature T.

    The "raw" stats are what comes directly off the softmax — the
    pre-discount, pre-NHSVM-reweight distribution that the caller
    will feed into ``nhsvm_to_mass``.  Use
    :func:`runtime_top1_mass_after_reweight` for the runtime-equivalent
    post-reweight stats.

    Returns a dict with: ``mean``, ``p50``, ``p95``, ``max``, ``std``
    of top1 probabilities, plus ``accuracy`` (top1 argmax vs label).
    """
    proba = _softmax_at_T(logits, T)
    top1_probs = proba.max(axis=1)
    top1_preds = proba.argmax(axis=1)
    return {
        "mean": float(top1_probs.mean()),
        "p50": float(np.percentile(top1_probs, 50)),
        "p95": float(np.percentile(top1_probs, 95)),
        "max": float(top1_probs.max()),
        "std": float(top1_probs.std()),
        "accuracy": float((top1_preds == y_true_idx).mean()),
    }


def runtime_top1_mass_after_reweight(
    proba: np.ndarray,
    codes: list[str],
    category_set: "Any",
    alphas: dict[str, float],
    *,
    discount: float = 0.20,
    T_reweight: float = 1.0,
) -> dict[str, float]:
    """Compute post-reweight, post-discount top1 mass — runtime proxy.

    Applies the same transformations ``pipeline.py`` does at inference:
    ``nhsvm_reweight`` (Choi 2015 Eq. 9 tree-distance penalty) then
    ``svm_to_mass`` (discount allocation).  This is THE metric to gate
    on — what fusion actually sees.

    Args:
        proba: ``(N, n_codes)`` softmax probabilities (already at the
            target temperature).
        codes: column-order list of hierarchical codes matching
            ``proba`` columns.
        category_set: HierarchicalCategorySet.
        alphas: NHSVM normalization alphas
            (``category_set.compute_nhsvm_alphas()``).
        discount: SVM discount (default 0.20 from base.conf).
        T_reweight: reweighting temperature (the nhsvm_reweight
            distance penalty's own T; not the softmax T).

    Returns:
        ``{mean, p50, p95, max, std}`` of top1 mass post-reweight +
        post-discount.
    """
    from atelier.classify.svm_classifier import nhsvm_reweight

    top1_masses: list[float] = []
    for i in range(proba.shape[0]):
        row = {codes[j]: float(proba[i, j]) for j in range(len(codes))}
        reweighted = nhsvm_reweight(
            row, alphas, category_set, T_reweight,
        )
        # svm_to_mass discount: each per-code mass becomes
        # prob * (1 - discount); Θ gets the discount.  Top1 mass after
        # discount = max(reweighted) * (1 - discount).
        if not reweighted:
            top1_masses.append(0.0)
            continue
        top_prob = max(reweighted.values())
        top1_masses.append(top_prob * (1.0 - discount))

    arr = np.array(top1_masses, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def path_product_probs(
    logits: np.ndarray,
    codes: list[str],
    parent_of: dict[str, "str | None"],
    *,
    T_global: float = 1.0,
    T_per_level: dict[int, float] | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Hierarchical path-product probability normalization (Variant 1).

    For each node n at depth d:
      ``cond_prob[n] = softmax(logits[siblings_of_n] / (T_global *
      T_per_level[d]))[n]``
    For each node n:
      ``p(n) = ∏ over ancestors of n (inclusive of n itself) of
      cond_prob[ancestor]``

    The result respects the tree's conditional-probability structure
    rather than treating all 287 nodes as flat-softmax competitors.
    Winning path's mass dominates by multiplicative concentration —
    expected 3-5× improvement in top1 mass mean over flat softmax at
    the same head logits (no retrain required).

    Args:
        logits: ``(N, n_nodes)`` raw path scores from
            ``head.forward(X)`` — pre-softmax.
        codes: column-order list of hierarchical codes matching
            ``logits`` columns.
        parent_of: ``{code: parent_code or None}``.  ``None`` marks
            forest roots (sibling group includes all forest roots).
            Must include an entry for every code in ``codes``.
        T_global: global temperature scalar applied to ALL logits
            before per-level softmax.
        T_per_level: optional ``{depth: T}`` for per-level temperature
            tuning.  Use when mass-leakage stays high at wide levels
            (e.g. the wide root subtree in sensitive taxonomies).
        normalize: when True (default), rescale each row so the sum
            of all per-node probs == 1.  Path-product naturally
            produces rows summing > 1 (internal nodes double-count
            their descendants), so renormalization makes the output
            compatible with ``nhsvm_to_mass``'s discount math.

    Returns:
        ``(N, n_nodes)`` path-normalized probabilities, same column
        order as ``codes``.
    """
    from collections import defaultdict

    code_to_idx = {c: i for i, c in enumerate(codes)}
    n_rows = logits.shape[0]

    # Group sibling indices by parent
    children_by_parent: dict["str | None", list[int]] = defaultdict(list)
    for code, parent in parent_of.items():
        if code in code_to_idx:
            children_by_parent[parent].append(code_to_idx[code])

    # Compute depth per code (root depth = 0, root's child depth = 1, ...)
    depth_by_code: dict[str, int] = {}

    def _depth(c: str) -> int:
        if c in depth_by_code:
            return depth_by_code[c]
        p = parent_of.get(c)
        d = 0 if p is None else _depth(p) + 1
        depth_by_code[c] = d
        return d

    for c in codes:
        _depth(c)

    # Walking path: ancestors (inclusive of self), root-first order doesn't matter
    ancestors_of: dict[str, list[str]] = {}

    def _ancestors(c: str) -> list[str]:
        if c in ancestors_of:
            return ancestors_of[c]
        p = parent_of.get(c)
        out = [c] if p is None else [c] + _ancestors(p)
        ancestors_of[c] = out
        return out

    for c in codes:
        _ancestors(c)

    # Per-sibling-group conditional softmax
    cond_prob = np.zeros_like(logits, dtype=np.float64)
    for parent, child_indices in children_by_parent.items():
        if not child_indices:
            continue
        # Determine temperature for this level (use depth of first child;
        # all siblings have the same depth by construction)
        any_child_code = codes[child_indices[0]]
        level = depth_by_code[any_child_code]
        T_lvl = T_per_level.get(level, 1.0) if T_per_level else 1.0
        T = max(float(T_global) * float(T_lvl), 1e-3)
        sib_logits = logits[:, child_indices].astype(np.float64) / T
        # Numerically stable softmax over siblings only
        sib_logits = sib_logits - sib_logits.max(axis=1, keepdims=True)
        sib_exp = np.exp(sib_logits)
        cond = sib_exp / sib_exp.sum(axis=1, keepdims=True)
        cond_prob[:, child_indices] = cond

    # Path product: each node's prob = ∏ over ancestors of cond_prob[ancestor]
    final_prob = np.zeros_like(logits, dtype=np.float64)
    for i, code in enumerate(codes):
        p = np.ones(n_rows, dtype=np.float64)
        for a in ancestors_of[code]:
            p *= cond_prob[:, code_to_idx[a]]
        final_prob[:, i] = p

    if normalize:
        row_sums = final_prob.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-15)
        final_prob = final_prob / row_sums

    return final_prob


def mass_leakage(
    proba: np.ndarray,
    codes: list[str],
    parent_of: dict[str, "str | None"],
) -> dict[str, float]:
    """Fraction of probability mass on nodes NOT on the argmax's path.

    Per the path-normalization diagnostic: with proper conditional
    softmax structure, the argmax leaf's ancestor chain should
    accumulate most of the mass; off-path nodes should drop near
    zero.  High leakage (> 8%) signals the per-level Ts are
    miscalibrated for unbalanced branching.

    For each row:
      argmax_idx = argmax(proba[row])
      on_path_codes = ancestors(codes[argmax_idx]) ∪ {codes[argmax_idx]}
      on_path_mass = sum(proba[row, on_path_indices])
      leakage[row] = 1 - on_path_mass / sum(proba[row])

    Returns ``{mean, p50, p95, max, std}`` of per-row leakage.
    """
    code_to_idx = {c: i for i, c in enumerate(codes)}
    ancestors_of: dict[str, list[str]] = {}

    def _ancestors(c: str) -> list[str]:
        if c in ancestors_of:
            return ancestors_of[c]
        p = parent_of.get(c)
        out = [c] if p is None else [c] + _ancestors(p)
        ancestors_of[c] = out
        return out

    for c in codes:
        _ancestors(c)

    leakage_per_row = []
    for row in range(proba.shape[0]):
        argmax_idx = int(np.argmax(proba[row]))
        argmax_code = codes[argmax_idx]
        on_path_idxs = [code_to_idx[a] for a in ancestors_of[argmax_code]
                        if a in code_to_idx]
        on_path_mass = proba[row, on_path_idxs].sum()
        total = proba[row].sum()
        if total <= 1e-15:
            continue
        leakage_per_row.append(1.0 - on_path_mass / total)

    if not leakage_per_row:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0}

    arr = np.array(leakage_per_row, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "std": float(arr.std()),
    }


def alpha_depth_reweight(
    proba: np.ndarray,
    codes: list[str],
    parent_of: dict[str, "str | None"],
    alpha: float = 0.7,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Variant 4 sanity baseline: post-hoc α^depth scalar reweight.

    For each node n at depth d:
      ``w_n = alpha ** d``
      ``p'(n) = p(n) * w_n``

    With α < 1, deeper nodes are down-weighted relative to shallower
    ones — concentrates mass toward shorter paths.  This is the
    cheap, zero-tree-math baseline to compare path_product against.
    Expected behavior: helps a bit, but path_product should dominate.
    """
    depth_by_code: dict[str, int] = {}

    def _depth(c: str) -> int:
        if c in depth_by_code:
            return depth_by_code[c]
        p = parent_of.get(c)
        d = 0 if p is None else _depth(p) + 1
        depth_by_code[c] = d
        return d

    for c in codes:
        _depth(c)

    weights = np.array(
        [alpha ** depth_by_code[c] for c in codes], dtype=np.float64,
    )
    reweighted = proba * weights[None, :]
    if normalize:
        row_sums = reweighted.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-15)
        reweighted = reweighted / row_sums
    return reweighted


def per_code_ece(
    proba: np.ndarray,
    y_true_idx: np.ndarray,
    codes: list[str],
    *,
    n_bins: int = 15,
    min_samples: int = 5,
) -> dict[str, float]:
    """Per-code ECE — surfaces which codes the head is miscalibrated on.

    Computes ECE separately for each code in ``codes``, restricting to
    samples where the true label is that code.  Codes with fewer than
    ``min_samples`` samples are skipped (ECE on tiny groups is noisy).

    Returns:
        ``{code: ece}`` for every code with sufficient samples.  Use
        ``sorted(result.items(), key=lambda x: -x[1])`` to surface
        worst-calibrated codes.
    """
    out: dict[str, float] = {}
    for code_idx, code in enumerate(codes):
        mask = y_true_idx == code_idx
        n = int(mask.sum())
        if n < min_samples:
            continue
        out[code] = expected_calibration_error(
            proba[mask], y_true_idx[mask], n_bins=n_bins,
        )
    return out
