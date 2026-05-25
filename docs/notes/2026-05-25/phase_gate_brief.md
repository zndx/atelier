# Phase gate brief — DST channel re-architecture (cosine + SVM)

**Date**: 2026-05-25
**Audience**: technical engineering review
**Status**: phase complete pending long-run validation
**Scope**: cosine channel reorganization + SVM channel ground-up re-architecture

---

## Executive summary

Two of the six DST evidence channels — late-interaction cosine and the
hierarchical SVM — have been re-architected over the past several
days.  The cosine reorganization (Qdrant multi-vector enrichment +
ColBERT MaxSim) is now delivering material held-out accuracy lift over
the prior single-vector approach.  The SVM channel was rebuilt from
first principles: the explicit Kronecker-expanded NHSVM head was
diagnosed as catastrophically incompatible with dense pretrained
encoders, replaced with the paper's efficient factorized form, and
extended for non-leaf prediction.  Initial held-out test accuracy is
61.25% on a 271-row stratified split (vs near-random for the broken
prior); a SHAP-priority-guided synthetic corpus expansion to ~57k
training rows is currently in flight to push that toward the ≥80%
target.

The factorized NHSVM implementation is the load-bearing technical
contribution.  It appears to be novel as an Apache-licensed open-source
artifact for hierarchical SVM classification over dense transformer
embeddings with first-class non-leaf prediction.

---

## Cosine channel — late-interaction reorganization

**What changed.**  Per-annotation Qdrant points now carry separate
multi-vectors per enrichment view (label, description, prototype
values, value patterns, name hints, parent path).  Column-side input
is symmetrically multi-vector.  Cosine signal is computed as
ColBERT-style MaxSim across views in-engine, with view-level mass
contributions composed into the DST fusion.

**Outcome.**  Significant held-out accuracy lift over the prior
single-vector cosine path.  Anti-example views were investigated and
ultimately excluded — the negative-MaxSim contribution introduced sign
instability in the mass function under Denoeux's combination rule for
the configurations tested.

**DST framing.**  Late interaction's per-view decomposition is also
the right abstraction for per-decision SHAP-style explanation; this is
captured in the architecture pivot doc as the long-term direction for
operator-facing per-prediction attribution.

---

## SVM channel — ground-up re-architecture

### Problem diagnosis

The existing `HierarchicalFeatureExpander` materializes Choi et al.
2015's NHSVM feature map `phi(x, y) = sqrt(α_y) · (x ⊗ e_y)` as an
explicit `(N, d × |nodes|)` sparse matrix and fits a LinearSVC over
it.  This works for sparse TF-IDF char-ngrams (where per-row only a
few hundred dims are nonzero) — fit-on-train top-1 = 98.93% on the
1149-row reference.  Substituting dense ModernBERT mean-pool
embeddings collapses the same configuration to **4.26%** — below
random for a 177-class problem.

The collapse is structural, not a tuning gap.  TF-IDF's sparsity makes
each class's Kronecker subspace discriminable from the others' all-
zero blocks at training time; dense embeddings populate every block
with similar-magnitude content at training and inference, leaving the
classifier no signal to learn class-specific block patterns.

### Resolution: factorized NHSVM head

Implemented `src/atelier/classify/factorized_nhsvm.py` per Choi et al.
2015's efficient form (Section 5.1 reduction):

- One learnable weight vector `W_n ∈ R^d` per hierarchy node
  (287 × 768 = 220k params for the deployed taxonomy)
- Frozen per-node normalization scalars `α_n` constrained to
  `Σ_n α_n = 1` along every root-to-leaf path
- Path score `γ(x, y) = Σ_{n in A_y} α_n · (W_n^T x)` computed via
  two batched matmuls (no Kronecker product ever materialized)
- Structured-SVM hinge loss with loss-augmented inference over all
  287 nodes; AdamW optimization

The effective model is mathematically a single `(|nodes|, d)` linear
layer with the structural prior baked into a precomputed
path-indicator × α matrix, so it admits exact Shapley attribution via
`shap.LinearExplainer` in milliseconds per row.

### Tree-distance loss design

Loss-augmented margin uses `Δ(y, y_i) = sqrt(Σ_{n ∈ A_y △ A_{y_i}} α_n)`
— the symmetric ancestor-set difference, α-weighted.  This naturally
handles three failure modes without special-casing:

- y ancestor of y_i (too general): `sqrt(Σ α_n on the path from y down to y_i)`
- y_i ancestor of y (too specific): symmetric to too-general
- Cross-subtree: `sqrt(Σ α_n on both halves of the LCA cut)`

Symmetric treatment of too-general and too-specific is deliberate.
12% of the production reference labels are at internal nodes (the
"parent-backoff pattern"); a too-specific prediction collapses
legitimate uncertainty into an arbitrary leaf, and the symmetric
penalty discourages that.

### Non-leaf prediction as a first-class output

The argmax over path scores ranges over all 287 nodes, leaves and
internals alike.  Both training (the structured loss accepts
internal-node `y_true`) and inference (the predicted argmax may land
on any node) support this without code branches.  Validated end-to-
end on synthetic and real data; the 134 internal-node reference
labels are no longer silently misclassifiable.

### Quantitative outcomes

| Configuration | Regime | Top-1 |
|---|---|---:|
| TF-IDF + Kronecker | fit-on-train, kept | 0.9893 |
| ModernBERT + Kronecker | fit-on-train, kept | 0.0426 (collapse) |
| ModernBERT + Factorized | fit-on-train, kept | 0.9938 |
| ModernBERT + Factorized | **held-out test, real-only train** | **0.6125** |

Held-out 61.25% on a 855-row training set across 177 classes (~5
examples/class) reflects data-scarcity overfitting, not architectural
ceiling.

### SHAP attribution — directional findings

Per-slice attribution on the held-out test (lean concatenated text):

- `sample_values`: |attr| = 0.85 (4× the next slice)
- `column_name`, `column_type`, `value_description`: ~0.21-0.23
- `sibling_context`, `source_table`: ~0.18-0.19
- `pattern_signals`: 0.14 (lowest)

These magnitudes are now the design priorities for the in-flight
synthetic corpus expansion: per-category value-distribution diversity
is the highest-leverage axis, per-category column-name vocabulary is
secondary, table/sibling context tertiary, explicit pattern signals
deprioritized.

---

## Novelty assessment

The factorized NHSVM implementation appears to be novel as an
Apache-licensed open-source artifact:

- **Choi 2015 NHSVM** (arXiv:1508.02479) is academic; no canonical
  open-source implementation under a permissive license that we are
  aware of.
- Most open-source hierarchical-SVM packages (sklearn-hierarchical-
  classification, hiclass, etc.) implement local-classifier-per-node
  or per-parent-node binary-decomposition strategies, not the
  globally-normalized structured-SVM Choi describes.
- The combination of (factorized form + frozen transformer encoder +
  loss-augmented inference + first-class non-leaf prediction + exact
  Shapley attribution via the head's linearity) does not appear in
  the open literature we have surveyed.
- The DST-fusion context (one of several analytically-independent
  evidence channels) is itself uncommon — most hierarchical SVM
  publications treat the classifier as a standalone end-to-end model
  rather than an evidence source.

This deserves dedicated technical write-up — the algorithm, the
non-leaf extension, the symmetric tree-distance Δ derivation, the
empirical comparison TF-IDF/Kronecker vs ModernBERT/Factorized — as a
standalone artifact independent of the Atelier-specific deployment
context.

---

## Open questions for phase gate

1. **Generalization confirmation.**  Held-out 61.25% is real but
   below the 80% target.  Corpus expansion to ~57k training rows is
   currently running on the App pod (Agent SDK + Bedrock Opus 4.6
   authoring per-node generators per a SHAP-priority spec).  Result
   expected in 6-12 hours.  Gate question: do we hold the phase open
   pending this result, or close on the architectural validation
   alone with corpus expansion as a continuation?

2. **TF-IDF generalization unknown.**  The 98.93% TF-IDF figure is
   fit-on-train; held-out generalization has not been measured.  Until
   it is, we cannot assert that ModernBERT-Factorized is *better* than
   TF-IDF-Kronecker on the metric that matters — only that it works
   correctly with dense encoders.  Recommended: run TF-IDF through the
   same held-out harness as a parity check before the standalone NHSVM
   write-up commits to comparative claims.

3. **DST discount recalibration.**  The factorized head is structurally
   different enough from the prior implementation that the previous
   discount calibration (0.22) should be re-derived.  Pending corpus
   expansion result.

4. **Standalone publication scope.**  Recommend separate technical
   note covering only the factorized NHSVM head + non-leaf extension +
   tree-distance Δ derivation + benchmark protocol, decoupled from
   Atelier-specific deployment context.  Audience: hierarchical-
   classification research community + open-source ML practitioners.

---

## Recommendation

**Conditional gate pass** pending the in-flight corpus expansion
result (D-day decision: held-out top-1 ≥ 0.70 keeps the architecture
on its current trajectory; <0.70 triggers structural simplification
work as parallel investigation — low-rank `W_n`, sibling weight
sharing).

Next phase scope:
- Corpus expansion convergence + held-out result interpretation
- TF-IDF parity baseline through the same harness
- DST discount recalibration against the new channel
- Standalone technical write-up draft for external publication
