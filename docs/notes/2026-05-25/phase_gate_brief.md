# Phase gate brief — DST channel re-architecture (cosine + SVM)

**Date**: 2026-05-25 00:25 MDT, amended 2026-05-25 12:09 MDT
**Audience**: technical engineering review
**Status**: phase complete pending long-run validation
**Scope**: cosine channel reorganization + SVM channel re-architecture

---

## Amendment — corrected framing

Two architectural corrections surfaced after the original brief was
written.  Both materially change the success criteria.  Earlier
sections are preserved as written for record; this addendum is the
authoritative interpretation.

### 1. Synth-only validate baseline replaces the 0.6125 figure

The original brief cited **0.6125** as the SVM baseline on a 271-row
held-out test split.  Under the corrected train/validate/test
framing — where the SVM trains on the synthetic corpus alone (no
reference data in training) and validates against the full
agent-mediated reference (1126 entities) — the comparable measurement
is **0.1856 full-validate top-1** with optional 0.1771 on the
historical 271-example slice (continuity).

The 0.6125 figure was carried almost entirely by the 855 real
reference-train rows mixed into training under the prior framing.
The 0.1856 figure is the direct measurement of synth-only
generalization to labeled reference, and is the starting point for
refinement under the metrology + agent-feedback loop.

### 2. DEPLOYMENT_READY redefinition

The original brief discussed a "primary criterion held-out test
top-1 ≥ 0.80" target.  That framing implicitly required the SVM
channel to be standalone-deployable at the operator bar.  The
correct framing for a DST evidence channel is **mutual affirmation
with the prior stage's output**:

- The SVM channel is DEPLOYMENT_READY when (cosine ⊕ SVM) materially
  uplifts cosine-alone on the full reference, with bounded per-code
  regressions and no overturns of confident-and-right cosine calls.
- A high-accuracy SVM that's redundant with cosine is NOT
  deployment-ready (adds compute without uplift).
- A low-accuracy SVM that's complementary to cosine IS
  deployment-ready (adds independent evidence the fusion can use).

The gate is implemented in `scripts/svm_cosine_uplift_gate.py`.
Four checks: global uplift ≥ ε, per-code regression band ≤ 5% of
cosine-correct rows, zero confident-cosine overturns, median
Dempster K ≤ 0.5 (diagnostic).

TARGET_ACCURACY (0.95) remains relevant — it's the operator's bar
for synth VALIDATION and the RUNTIME PIPELINE (full DST ensemble:
cosine + SVM + LLM + CatBoost + name + pattern fused by
`src/atelier/classify/pipeline.py`). The PIPELINE is a downstream
measurement made by running the Atelier classification pipeline
against hive-poc and counting `matches_reference == True` rate in
`classifications.json`.  It is NOT the same as the SVM-stage gate.

### 3. Cosine channel context

`just optimize cosine` is **enrichment evolution**, not model
training: a GEPA-shaped APO critic loop (`scripts/semantic_optimize.py`)
that iteratively edits ColBERT enrichment payloads in Qdrant via
LLM-mediated rewriting.  The "weights" of the cosine channel are the
multi-vector ColBERT representations stored in Qdrant; optimization
is in-place payload editing keyed on per-cluster rescue@1/3/5.  The
SVM channel must affirm what that stage produced — not replace or
override it.

### 4. Why standalone accuracy is the wrong gate

DST evidence fusion's load-bearing property is complementary independence
between channels, not individual channel strength. Two cases prove this
cuts both ways:

Case A — low-accuracy complementary channel is DEPLOYMENT_READY:
    - A_cos = 0.78, A_svm = 0.65, A_cos⊕svm = 0.86
    - The SVM is only 65% standalone but its errors are uncorrelated with cosine's errors
    - Fusion realizes +8pp uplift over cosine alone
    - Architecturally correct addition → ship it

Case B — high-accuracy redundant channel is NOT DEPLOYMENT_READY:
    - A_cos = 0.93, A_svm = 0.95, A_cos⊕svm = 0.93
    - The SVM is 95% standalone but its mass distribution mirrors cosine's
    - Adding it to the ensemble costs compute without uplift
    - Violates the source-independence assumption that makes Dempster's rule valid
    - Reject — even though it "passes" any standalone-accuracy bar

Case C — high SVM that regresses cosine's wins:
    - A_cos = 0.78, A_svm = 0.92, A_cos⊕svm = 0.81
    - Globally positive uplift, but SVM is fighting cosine where cosine was right
    - Silent failure mode hidden by aggregate accuracy
    - The per-code regression band is what catches this

#### The formalization
      
SVM is DEPLOYMENT_READY iff:
    (1) A_cos⊕svm − A_cos ≥ ε                    [global uplift, mandatory]
    (2) regression_rate ≤ δ                      [per-code regression band, mandatory]
    where regression_rate = |{rows : cos_top1=true AND fused_top1≠true}| / |{cos_top1=true}|
    (3) ∀ rows with m_cos({top1}) ≥ τ AND cos_top1=true: fused_top1 = true
                                                   [confidence-conditional do-no-harm, mandatory]
    (4) median(K) ≤ κ                            [conflict bound, diagnostic only]

Defaults: ε=0.01, δ=0.05, τ=0.7, κ=0.5

TARGET_ACCURACY is two separate concepts, found in multiple operational modalities.
      
The 0.95 operator bar lives at _both the runtime ensemble, _and_ the synthetic NHSVM channel:
    - Validate accuracy is the upper-bound proxy for test accuracy (reference is sampled from hive-poc;
    same distribution)
    - The full DST ensemble's matches_reference == True rate in classifications.json is the production
    gate
    - Each channel's pairwise-uplift gate is necessary; the full-ensemble gate is the ultimate decision

These are different decisions: an architecturally-correct ensemble can still fall short of 0.95 (keep iterating);
a 0.95-achieving ensemble can still contain a redundant channel that should be removed (audit independence).
     
The bidirectional NHSVM / Cosine channel logic in one sentence:
    ▎ A channel passes its deployment gate by demonstrating uplift over the prior stage's fusion, 
    ▎ regardless of its absolute accuracy; conversely, no level of standalone accuracy passes the gate if
    ▎ uplift is absent.

This is what scripts/svm_cosine_uplift_gate.py encodes — pure pairwise Dempster fusion of cosine ⊕
SVM mass on the full agent-mediated reference (1126 rows), compared against cosine-alone on the same
rows.

---

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
   below the true 95% target.  Corpus expansion to ~57k training
   rows is currently running with Agent SDK + Bedrock Opus 4.6
   authoring per-node generators per a SHAP-priority spec.  Result
   expected in 6-12 hours.  We will hold the phase open pending this
   result, and any remediation indicated by the result.

2. **DST discount recalibration.**  The factorized head is structurally
   different enough from the prior implementation that the previous
   discount calibration (0.22) should be re-derived.  Pending corpus
   expansion result.

3. **Standalone publication scope.**  Recommend separate technical
   note covering only the factorized NHSVM head + non-leaf extension +
   tree-distance Δ derivation + benchmark protocol, decoupled from
   Atelier-specific deployment context.  Audience: hierarchical-
   classification research community + open-source ML practitioners.

---

## Recommendation

**Conditional gate pass** pending the in-flight corpus expansion
result (D-day decision: held-out top-1 ≥ 0.90 keeps the architecture
on its current trajectory; <0.90 triggers structural simplification
work as parallel investigation — low-rank `W_n`, sibling weight
sharing).

Next phase scope:
- Corpus expansion convergence + held-out result interpretation
- DST discount recalibration against the new channel
- Standalone technical write-up draft for external publication

