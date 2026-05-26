# Phase gate brief — DST channel re-architecture (cosine + SVM)

**Date**: 2026-05-25 (consolidated rewrite from 00:25 MDT draft + 12:09 MDT amendment)
**Audience**: technical engineering review
**Status**: architecture complete, first end-to-end gate run executed, next refinement cycle defined
**Scope**: cosine channel reorganization, SVM channel re-architecture, `just optimize` framework boundaries, dual-gate exit contract

---

## Executive summary

Two of the six DST evidence channels — late-interaction cosine and the
hierarchical SVM — have been re-architected over the past several
days, along with the `just optimize` framework that prepares them for
the runtime Atelier classification pipeline.

- **Cosine channel** (`just optimize cosine` → `scripts/semantic_optimize.py`):
  per-annotation Qdrant points now carry separate multi-vectors per
  enrichment view; cosine signal is computed via ColBERT-style MaxSim
  in-engine; iterative GEPA-shaped LLM critic loop edits enrichment
  payloads in place keyed on per-cluster rescue@5.  Significant
  held-out accuracy lift over the prior single-vector path; the
  active production collection is at
  `hive-poc_hive-poc-default-annotations_019e5636-...`.

- **SVM channel** (`just optimize svm`): rebuilt from first principles.
  The legacy Kronecker-expanded NHSVM head collapses catastrophically
  on dense pretrained encoders (4.26% fit-on-train with ModernBERT
  vs 98.93% with TF-IDF).  The factorized form per Choi et al. 2015
  Section 5.1 — per-node weight vectors `W_n ∈ R^d` + frozen
  path-normalization scalars `α_n` — recovers full architectural
  capacity (99.38% fit-on-train) and admits exact Shapley attribution
  via `shap.LinearExplainer`.  Synth-only-trained, full-reference-
  validated baseline measurement is **in progress** (1126-row reference).

- **Closed-loop metrology + refinement** (`scripts/corpus_metrology.py`
  + `scripts/refine_generators_from_failures.py`): per-code fidelity,
  separability, spread percentiles + shape exemplars drive
  Bedrock Opus 4.6 generator authorship; per-pass cooldown,
  abandon_structural exclusion, and best-pass snapshotting prevent
  the orchestration pathologies the prior pipeline exhibited
  (silent peak overwrite, target-list churn, n_validate=1 noise
  feedback).

- **Dual-gate exit contract** (`scripts/svm_cosine_uplift_gate.py`):
  the SVM stage exits when (a) TARGET_ACCURACY = 0.95 is reached or
  demonstrably plateaued AND (b) DEPLOYMENT_READY passes via pairwise
  cosine⊕SVM mutual affirmation on the full reference set.
  Both gates required for shipping into the runtime ensemble.

- **First gate run against corpus_v2** (2026-05-25 evening):
  DEPLOYMENT_READY = FALSE.  A_cos = 0.5657, A_svm = 0.1918,
  A_fused = 0.5071 (uplift −5.86pp).  The gate works exactly as
  designed and produces actionable per-code data for the next
  refinement cycle.

The factorized NHSVM implementation is the load-bearing technical
contribution.  It appears to be novel as an Apache-licensed open-
source artifact for hierarchical SVM classification over dense
transformer encoders with first-class non-leaf prediction.

---

## The `just optimize` framework

Three sequential stages with a strict composability contract.  Each
stage's output is consumed (not replaced) by the next; the chain ends
at `svm` and the Atelier classification pipeline handles full DST
fusion at runtime.

```
just optimize agent     just optimize cosine      just optimize svm
─────────────────       ────────────────────      ─────────────────
Agent-mediated          Iterative LLM critic      Synth corpus + factorized
reference curation      editing ColBERT enrichment NHSVM training + closed-
of hive-poc columns →   payloads in Qdrant →      loop metrology refinement
                                                                            ↓
                              EXITS via dual gate (TARGET_ACCURACY ∧ DEPLOYMENT_READY)
                                                                            ↓
                              ┌─────────────────────────────────────────────┘
                              ↓
        Atelier classification pipeline (src/atelier/classify/pipeline.py)
        runtime DST fusion of six channels:
            cosine ⊕ SVM ⊕ LLM ⊕ CatBoost ⊕ name-match ⊕ pattern
        Production decision: matches_reference == True rate on
        classifications.json against hive-poc ≥ 0.95
```

CatBoost and LLM do not have `just optimize` stages — they join at
runtime through the existing fusion machinery.  The framework
boundary is what it is by design: each `just optimize` stage prepares
a channel; the runtime pipeline composes channels.

---

## Cosine channel — late-interaction reorganization

**Architecture**: `just optimize cosine` runs
`scripts/semantic_optimize.py`, a GEPA-shaped APO critic loop.  Per
target cluster:

1. Read failure rows from `classifications.json`.
2. Have an LLM propose **enrichment edits** to the Qdrant payload for
   the target code (description, name_hints, prototype_values,
   value_patterns) — NOT model retraining.
3. Apply via `upsert_point_in_place(point_id, vectors, payload)`,
   preserving point_ids.
4. Re-measure rescue@1/3/5.  Reference is frozen oracle; all rows
   stay in the denominator throughout.

The cosine channel's "weights" are the ColBERT multi-vector
representations stored in Qdrant; optimization is in-place payload
editing.

**Mass-function output**: `late_interaction_to_mass(scored_tags,
frame, discount=0.20)` in `src/atelier/classify/mass_functions.py`:

- Haenni-Hartmann reliability shaping: α bounded by
  `[reliability_floor=0.10, ceiling=1−discount=0.80]`; α is a
  function of top-1 absolute MaxSim and top-1/top-2 margin
- Margin-aware allocation: `m(top-1) = α · margin_w`; residual
  split between LCA internal-node focal element (when top-K share
  an ancestor) and per-leaf softmax
- `m(Θ) = 1 − α` carries epistemic uncertainty when MaxSim is
  diffuse

Anti-example views were investigated and ultimately excluded —
negative-MaxSim contributions introduced sign instability in the
mass function under the tested configurations.

**DST framing**: the per-view decomposition is also the right
abstraction for per-decision SHAP-style explanation; this is captured
in the architecture pivot doc as the long-term direction for
operator-facing per-prediction attribution.

---

## SVM channel — ground-up re-architecture

### Problem diagnosis

The legacy `HierarchicalFeatureExpander` materializes Choi et al.
2015's NHSVM feature map `phi(x, y) = sqrt(α_y) · (x ⊗ e_y)` as an
explicit `(N, d × |nodes|)` sparse matrix and fits a LinearSVC over
it.  Empirical results on the 1149-row reference:

| Configuration | Regime | Top-1 |
|---|---|---:|
| TF-IDF + Kronecker | fit-on-train, kept | 0.9893 |
| ModernBERT + Kronecker | fit-on-train, kept | **0.0426 (collapse)** |
| ModernBERT + Factorized | fit-on-train, kept | 0.9938 |

The collapse is structural, not a tuning gap.  TF-IDF's sparsity
makes each class's Kronecker subspace discriminable from the others'
all-zero blocks at training time; dense embeddings populate every
block with similar-magnitude content at training and inference,
leaving the classifier no signal to learn class-specific block
patterns.

### Resolution: factorized NHSVM head

`src/atelier/classify/factorized_nhsvm.py` implements Choi et al.
2015's efficient form (Section 5.1 reduction):

- One learnable weight vector `W_n ∈ R^d` per hierarchy node
  (287 nodes × 768 ModernBERT dim = ~220k params)
- Frozen per-node normalization scalars `α_n` constrained to
  `Σ_n α_n = 1` along every root-to-leaf path
- Path score `γ(x, y) = Σ_{n ∈ A_y} α_n · (W_n^T x)` computed via
  two batched matmuls (no Kronecker product ever materialized)
- Structured-SVM hinge loss with loss-augmented inference over all
  287 nodes; AdamW optimization
- The effective model is mathematically a single `(|nodes|, d)`
  linear layer with the structural prior baked into a precomputed
  path-indicator × α matrix, so it admits exact Shapley attribution
  via `shap.LinearExplainer` in milliseconds per row

### Tree-distance loss design

Loss-augmented margin uses
`Δ(y, y_i) = sqrt(Σ_{n ∈ A_y △ A_{y_i}} α_n)` — the symmetric
ancestor-set difference, α-weighted.  This handles three failure
modes without special-casing:

- y ancestor of y_i (too general):
  `sqrt(Σ α_n on the path from y down to y_i)`
- y_i ancestor of y (too specific): symmetric to too-general
- Cross-subtree:
  `sqrt(Σ α_n on both halves of the LCA cut)`

Symmetric treatment of too-general and too-specific is deliberate:
12% of the production reference labels are at internal nodes (the
"parent-backoff pattern"); a too-specific prediction collapses
legitimate uncertainty into an arbitrary leaf, and the symmetric
penalty discourages that.

### Non-leaf prediction as a first-class output

The argmax over path scores ranges over all 287 nodes — leaves and
internals alike.  Both training (the structured loss accepts
internal-node `y_true`) and inference (the predicted argmax may land
on any node) support this without code branches.  Validated
end-to-end on synthetic and real data; the 134 internal-node
reference labels are no longer silently misclassifiable.

### Mass-function output

`nhsvm_to_mass(proba, frame, category_set, alphas, discount=0.20)`
in `mass_functions.py`:

1. `nhsvm_reweight`: Choi-2015 path-distance reweighting applied to
   the head's softmax probabilities (penalizes cross-subtree
   probability flow proportionally to normalized tree distance,
   structurally discouraging shallow catch-all assignments)
2. `svm_to_mass`: standard projection to focal elements with
   `evidence_mass = 1 − discount` and `m(Θ) = discount + (residual)`

Confusable-pair redistribution applies the same way it does for the
legacy TF-IDF path.

### Synth-only validate baseline

Under the corrected train/validate/test framing — SVM trains on the
synthetic corpus alone (no reference data in training) and validates
against the full agent-mediated reference (1126 examples) — first
measurement is:

| Metric | Value |
|---|---:|
| Train fit-acc (synth-only, 57400 rows) | 0.9302 |
| **Full-reference validate top-1** | **0.1856** |
| Continuity-271 slice validate top-1 (seed=42) | 0.1771 |

The historical 0.6125 figure cited in prior drafts was carried almost
entirely by the 855 reference-train rows mixed into training under
the legacy framing.  The 0.1856 figure is the direct measurement of
synth-only generalization to labeled reference; it is the honest
starting point for the metrology + refinement loop.  The 0.6125
anchor is preserved in `BASELINE_ANCHORS` as historical context only.

---

## Closed-loop metrology + refinement

The refinement loop's purpose is **quantitative steering of corpus
quality toward TARGET_ACCURACY = 0.95**.  Three machinery pieces
make this concrete:

### Per-code metrology (`scripts/corpus_metrology.py`)

Computes per-code signals in the embedding space the factorized
NHSVM head actually uses:

| Signal | Computation | Interpretation |
|---|---|---|
| **Fidelity** | cos distance: synth-centroid ↔ reference-centroid | high distance → synth shape doesn't resemble target |
| **Separability** | cos distance: code's synth-centroid ↔ each top-K neighbor's synth-centroid (neighbors from validate-set confusions) | low distance → generators collapsed toward each other |
| **Spread** | mean pairwise cosine within code's synth | high spread → generator saturated coverage |
| **Shape exemplars** | up to 5 reference lean-text strings for that code | concrete ground-truth examples the agent can imitate |

Thresholds are **percentile-based**, not absolute — ModernBERT's
general-purpose encoder produces high-baseline similarity (most
text pairs cos > 0.85), so absolute centroid distances aren't
directly interpretable across codes.  A code is flagged for action
only when `validate_accuracy < 0.5` AND one of its signals lands in
the population tail (fidelity in top decile, separability in bottom
decile, spread in top decile).

### Refinement loop (`scripts/refine_generators_from_failures.py`)

Per-pass workflow:

1. Read latest Phase D pass-numbered results.
2. Run metrology → per-code recommended_action ∈
   {improve_fidelity, improve_separability, reduce_redundancy,
   abandon_structural, hold}.
3. Identify bottom-N targets, filter by `n_validate ≥ 3` (eliminates
   the singleton-noise feedback loop the prior pipeline exhibited),
   exclude codes in cooldown or permanently abandoned.
4. Join with metrology signals + shape exemplars.
5. Write `refinement_targets.json` for the agent.
6. Re-author generators via Agent SDK (Bedrock Opus 4.6) with rich
   diagnostic context.
7. Regenerate corpus; re-run Phase D with pass-numbered output.
8. Snapshot per-pass corpus state for best-pass restoration.
9. Update best_pass.json on improvement; apply cooldown=2 to
   non-improvers; promote persistently-abandon-flagged codes to a
   permanent exclude set.

Stop conditions:
- `full_validate_top1 ≥ TARGET_ACCURACY` (default 0.95), OR
- Two consecutive passes with lift < 1pp (plateau), OR
- `--max-passes` reached (default 5)

### Volume cap + diversity gating (`scripts/generate_corpus_v2.py`)

The encoder consumes each synthetic example as a single lean-text
string with at most ~5 sample values visible.  Beyond ~30-50
well-chosen distinct examples per code, additional volume produces
clustered redundant embeddings, not new signal.  Default
`--synth-examples-per-code` reduced 200 → 40; per-candidate
diversity gate rejects new examples with cosine sim > 0.95 to any
already-accepted same-code example; marginal-coverage stop halts
generation when convex-hull-radius growth falls below ε = 0.02 over
10 consecutive added examples.

---

## The dual-gate exit contract

The SVM stage has **two distinct gates that BOTH apply**.  This is
the central conceptual point of the redesign and the place where
prior framings hid an out.

### Gate A — TARGET_ACCURACY (quantitative steering)

**What it is**: the validate-stage stop signal for the refinement
loop.  Default 0.95.  The loop iterates until full-validate top-1
reaches this OR honest plateau OR max-passes.

**Why it matters as the primary signal**: the metrology + agent
feedback machinery exists to **drive corpus quality up**.  Per-round
recommended_actions, shape exemplars, percentile rankings — all of
it is quantitative pressure on the agent to author better
generators.  If TARGET_ACCURACY is diminished, the agent has no
ceiling-pressure and the metrology's purpose dissolves.

**Why 0.95**: the reference is sampled from hive-poc, so validate
accuracy is the upper-bound proxy for what the runtime pipeline's
matches_reference rate can reach.  The full DST ensemble's
matches_reference rate ≥ 0.95 against hive-poc is the operator's
production bar; each contributing channel must aim for that bar
because no fusion can systematically exceed the strongest
contributing signal on hard codes.

**Failure mode if absent**: agent stops at "good enough to mutually
affirm cosine" (e.g., 0.30 standalone) and the full ensemble can't
hit production bar because the SVM doesn't carry enough weight on
codes the other channels miss.

### Gate B — DEPLOYMENT_READY (architectural correctness)

**What it is**: the architectural-correctness check at the end of
`just optimize svm`.  Tests whether the trained SVM channel is a
verified mutually-affirming addition to the cosine channel that
`just optimize cosine` previously established.

**Why it matters separately**: an SVM channel can hit TARGET_ACCURACY
and still be wrong-shaped for the ensemble — if its mass
distribution mirrors cosine's, it's redundant; if its high-confidence
votes conflict with cosine's high-confidence votes on the same wrong
codes, it's source-dependent in the Denoeux-2008 sense, which
violates Dempster's rule's independence assumption.

**Failure mode if absent**: high-accuracy redundant channel ships
into the ensemble, adding inference cost without uplift; or
high-accuracy SVM with correlated errors silently undermines
fusion calibration in production.

### Formalization

```
SVM channel is shippable iff BOTH:

  Gate A (TARGET_ACCURACY, quantitative steering):
    Under synth-only training:
      full_validate_top1 ≥ τ_acc                            [default τ_acc = 0.95]
    Under reference-primary k-fold training (default since 2026-05-26):
      mean(fold-val_top1) ≥ τ_acc  AND  std(fold-val_top1) ≤ τ_std
                                                            [default τ_acc = 0.95, τ_std = 0.03]

  Gate B (DEPLOYMENT_READY, architectural correctness):
    (1) A_cos⊕svm − A_cos ≥ ε                               [global uplift]
    (2) regression_rate ≤ δ                                  [per-code regression band]
        where regression_rate = |{rows: cos_top1=true ∧ fused_top1≠true}|
                                / |{rows: cos_top1=true}|
    (3) ∀ rows with m_cos({top1}) ≥ τ_conf ∧ cos_top1=true:
            fused_top1 = true                                [do-no-harm on confident cosine]
    (4) median(K) ≤ κ                                        [conflict bound, diagnostic]

  Defaults: ε = 0.01, δ = 0.05, τ_conf = 0.7, κ = 0.5
```

### Training protocol (Gate A measurement surface)

Two training protocols measure Gate A.  They are CALIBRATIONS of the
same gate, not separate gates:

**Synth-only training (historical default until 2026-05-26)**.
Train on the synthetic corpus alone; validate on the full reference.
`full_validate_top1` is the headline number.  Generalization to
hive-poc-at-large is upper-bounded by validate-on-reference because
reference is sampled from hive-poc.  Pre-switch best: 0.55 on a 287-
node tree.

**Reference-primary k-fold training (default since 2026-05-26)**.
The agent-mediated reference is itself the training data; each fold
trains on 80% of the reference (audit-policy-weighted) augmented with
synth rows for under-represented codes, then predicts the held-out
20%.  The union of fold-vals is the full reference, so the headline
`validate.full_top1` is a true held-out number computed across the
entire reference.  Per-fold mean ± std quantifies the variance of
that number; both `mean ≥ τ_acc` and `std ≤ τ_std` must hold.

The synth-only diagnostic is preserved as a parallel signal
(`--also-report-synth-only`) so the operator continues to see how the
synth corpus alone would score — useful for spotting when the held-
out lift is coming from reference-data exposure (fold augmentation
working) versus structural synth improvement (the corpus genuinely
learning the deployment scope).

**Rationale for the switch**: the factorized NHSVM head has ~98%
proven fit capacity at the reference's scale; synth-only-train denied
the head access to the labels we already had.  Validation IS the
deployment target (the reference is a sample of the deployment
scope), so reference-primary aligns training with what the channel
must predict.  Per [`feedback_validation_target_over_generalization`](
../../../.claude/projects/-home-cdsw/memory/feedback_validation_target_over_generalization.md):
generalizability is secondary to validation-target accuracy when
validation IS the deployment target.

### Re-baseline 2026-05-26 (post-protocol-switch)

| metric | synth-only (corpus_v3) | reference-primary k=2 smoke | reference-primary k=5 (run-pending) |
|---|---:|---:|---:|
| held-out full-ref top-1 | 0.5488 | 0.7397 | TBD |
| per-fold mean ± std | n/a (single split) | 0.7403 ± 0.0092 | TBD |
| synth-only diagnostic | 0.5488 | 0.5519 | TBD |
| protocol-switch lift   | — | +0.1878 | TBD |
| Gate A pass            | ✗ | ✗ (need τ_acc=0.95) | TBD |

Smoke test was k=2 (a 50/50 train/val partition, the LEAST favorable
k-fold setting); k=5 (80/20 per fold) is expected to score
substantially higher and is what the AMP-pod convergence cycle will
measure.  Numbers above set the baseline; subsequent refinement
passes will populate the k=5 column.

The asymmetry that matters:

- A high-accuracy SVM that fails Gate B = **learned the wrong thing**
  (redundant with cosine, or correlated errors).  Don't ship.
- A DEPLOYMENT_READY SVM that hasn't hit Gate A = **correct but
  weak**.  Don't ship — keep iterating the corpus to push accuracy up.
- A DEPLOYMENT_READY SVM at Gate A = **shippable**.  Integration
  into `pipeline.py` follows; the runtime ensemble takes over.

The production gate — the FULL ensemble's matches_reference rate ≥
0.95 against hive-poc — is downstream and operator-driven.  It is a
distinct decision: an architecturally-correct ensemble can still
fall short of 0.95 (keep iterating); a 0.95-achieving ensemble can
still contain a redundant channel that should be removed (audit
independence).

### The bidirectional logic in one sentence

> A channel passes Gate B by demonstrating uplift over the prior
> stage's fusion, regardless of its absolute accuracy; conversely,
> no level of standalone accuracy passes Gate B if uplift is absent.
> Gate A is independent and enforces the quantitative ceiling.

---

## First gate result against corpus_v2 (2026-05-25 evening)

End-to-end pipeline executed: PGlite + Qdrant started, taxonomy_registry
populated, factorized NHSVM trained on synth-only (cache hits across
the board), 1126 reference rows scored through cosine + SVM + Dempster
fusion.  Full run wall-clock ~21 min; report at
`build/svm_uplift_gate/uplift_report.md`.

### Headline

| Metric | Value |
|---|---:|
| A_cos (cosine alone) | **0.5657** |
| A_svm (SVM alone) | 0.1918 |
| A_fused (cosine ⊕ SVM) | **0.5071** |
| **Uplift (A_fused − A_cos)** | **−0.0586** |

DEPLOYMENT_READY = **FALSE**.

### Per-check breakdown

| # | Check | Threshold | Observed | Pass |
|---|---|---:|---:|---|
| 1 | global_uplift | ≥ +0.01 | **−0.0586** | ✗ |
| 2 | regression_band | ≤ 0.05 | **0.1774** | ✗ |
| 3 | confident_cosine_preservation | 0 violations | 0 | ✓ |
| 4 | median_K | ≤ 0.5 | 0.4864 | ✓ |

### Interpretation

The gate caught exactly the failure mode it was designed for.
Patterns visible in the per-row data:

- **Check 3 passes**: zero overturns of confident cosine top-1
  predictions (rows where cosine emitted ≥ 0.7 belief on its
  top-1 and was right).  The SVM stays out of the way on cosine's
  strong predictions.
- **Check 2 fails**: 17.74% of rows cosine got right become wrong
  in fusion — the SVM is voting confidently on wrong codes in the
  mid-confidence cosine band.  These are rows where cosine had
  moderate belief on the right code and the SVM's diffuse-but-
  committed mass on the wrong code tipped fusion the wrong way.
- **Check 4 just under threshold**: median K = 0.4864.  Channels
  are conflicting frequently but not catastrophically.  Useful
  diagnostic noise floor.

The signature is consistent with what we predicted: at 0.19
standalone accuracy, the SVM is **confidently wrong on most codes**
rather than well-calibrated-and-uncertain.  Calibration improvement
— driving Θ-mass up on uncertain predictions — is part of the fix;
raw accuracy improvement (driving the corpus toward TARGET_ACCURACY)
is the other part.

### Per-code uplift patterns

**Top-20 improvements** (the SVM is already adding value here even
at 0.19 standalone):
- `1.1.1.2.8` (n=4): cosine 0%, SVM 100%, fused 100% — pure +1.0pp
- `1.1.1.9.6.1` (n=4): cosine 0%, SVM 100%, fused 100% — pure +1.0pp
- `1.1.1.3.4.2` (n=4): cosine 0%, SVM 100%, fused 75%
- `1.1.1.4.2.3.2` (BILL/SHIP family, n=4): cosine 25%, SVM 100%, fused 75%
- `1.1.1.6.3` (n=3): cosine 67%, SVM 100%, fused 100%
- ...20 codes net positive

**Top regressions** (cosine was strong, SVM dragged fusion wrong):
- `1.2.2` (MONEY, n=43): cosine 100%, SVM 0%, fused 72.1% — **−27.9pp**
- `1.3.2.1.1` (SYSURL, n=5): cosine 100%, SVM 0%, fused 60%
- `1.1.2.4.1` (n=4): cosine 100%, SVM 0%, fused 50%
- `1.2.6.2` (DAID neighbor, n=5): cosine 100%, SVM 0%, fused 60%
- `1.1.2.1.1.4` (n=3): cosine 100%, SVM 0%, fused 67%

The regression pattern is the actionable next-cycle target.  High-n
codes where cosine is already at 100% and the SVM is at 0% are the
ones where the SVM most needs to **abstain** (emit high Θ-mass)
rather than vote confidently wrong.  The metrology's
`abandon_structural` flag is the right escape hatch for codes that
genuinely cannot be discriminated from value shape alone; for
codes the SVM should learn, the refinement loop's targeted
re-authoring is the lever.

---

## What the next refinement cycle does with this

The gate's per-code regression table is **the agent's diagnostic
context for the next cycle**.  Specifically:

- High-n cosine-strong-SVM-zero codes (`1.2.2` MONEY especially —
  43 rows is the largest single-code population in the validate
  set) are the highest priority for either accurate re-authoring
  OR principled abandonment.
- The 20 improvement codes prove the architecture works when the
  SVM has signal.  Per-code metrology trajectory across the next
  passes should show abandonment of the structurally-hard codes
  and continued lift on the learnable ones.
- TARGET_ACCURACY (0.95) remains the steering signal; the agent
  authors generators with metrology percentiles + shape exemplars
  as quantitative context, NOT "is it deployment ready yet"
  (because that question has a much lower-bar answer).

Concrete operational next step:

```bash
# Drive next refinement cycle with the gate's per-code data as input
bash scripts/run_corpus_expansion_pipeline.sh \
    --corpus-dir build/data/svm_training/corpus_v3 \
    --synth-examples-per-code 40 \
    --target-accuracy 0.95
```

The pipeline will: regenerate corpus_v3 with volume cap + diversity
gating, run 5-pass refinement with metrology feedback aimed at 0.95,
restore best-pass corpus state, then re-run the uplift gate.

---

## Novelty assessment

The factorized NHSVM implementation appears to be novel as an
Apache-licensed open-source artifact:

- **Choi 2015 NHSVM** (arXiv:1508.02479) is academic; no canonical
  open-source implementation under a permissive license that we have
  found.
- Most open-source hierarchical-SVM packages
  (sklearn-hierarchical-classification, hiclass, etc.) implement
  local-classifier-per-node or per-parent-node binary-decomposition
  strategies, not the globally-normalized structured-SVM Choi
  describes.
- The combination of (factorized form + frozen transformer encoder +
  loss-augmented inference + first-class non-leaf prediction + exact
  Shapley attribution via the head's linearity) does not appear in
  the open literature we have surveyed.
- The DST-fusion context — channel as an evidence source bounded by
  pairwise mutual-affirmation with an enrichment-driven cosine
  channel — is itself uncommon; most hierarchical SVM publications
  treat the classifier as a standalone end-to-end model.

This deserves dedicated technical write-up — the algorithm, the
non-leaf extension, the symmetric tree-distance Δ derivation, the
empirical comparison TF-IDF/Kronecker vs ModernBERT/Factorized, the
uplift-gate framing as a deployment criterion for ensemble-member
channels — as a standalone artifact independent of the
Atelier-specific deployment context.

---

## Open questions for the phase gate

1. **TARGET_ACCURACY achievability under the metrology loop**.
   Synth-only validate is 0.1856; the loop targets 0.95.  Whether
   the metrology + Bedrock Opus authorship can close that 77pp gap
   over multiple refinement cycles is the central empirical
   question.  Per-cycle convergence rate is the data we need to
   make this concrete.

2. **DST discount recalibration**.  The factorized head's mass
   distribution is structurally different from the legacy TF-IDF
   path; the previous discount calibration (0.30 in production,
   0.20 in standalone signature) should be re-derived under the
   new shape.  Pending corpus convergence.

3. **Structural code-set decisions**.  Several codes are likely
   beyond what the SVM channel can learn from value shape alone
   (A_HD / A_ID / ENOS catch-alls, BILL/SHIP family value-identical
   pairs, opaque proprietary IDs).  Owner conversation needed on
   whether to:
   - Pull these from the SVM target set entirely (let cosine + LLM
     + name-match handle them)
   - Collapse BILL/SHIP at the parent for SVM training (let the
     name and table-context channels disambiguate leaves)
   The metrology produces `abandon_structural` flags as the data
   point for that conversation.

4. **Channel-independence audit at the full ensemble level**.  The
   current gate audits cosine ⊕ SVM pairwise.  Once the SVM ships,
   the natural extension is to audit each pairwise channel
   combination in the runtime fusion for Denoeux-2008
   source-independence.  Future work.

5. **Standalone publication scope**.  Recommend a separate technical
   note covering only the factorized NHSVM head + non-leaf
   extension + tree-distance Δ derivation + benchmark protocol +
   the uplift-gate framing, decoupled from Atelier-specific
   deployment context.  Audience: hierarchical-classification
   research community + open-source ML practitioners.

---

## Recommendation

**Architectural phase complete.**  All scaffolding is in place:
factorized NHSVM, mass-function output, metrology + refinement loop,
dual-gate exit contract, end-to-end orchestrator, target-data-source
diagnostic.  The first gate run executed end-to-end against live
infrastructure (PGlite + Qdrant + the production cosine collection)
and produced the expected diagnostic — corpus_v2's SVM is not yet
deployment-ready, with a concrete per-code regression table pointing
at the next refinement cycle's targets.

**Open the next iteration cycle.**  Run the corpus regeneration with
volume cap + diversity gating + metrology-driven refinement targeting
TARGET_ACCURACY = 0.95, with the gate re-evaluating at the end.  Per-
cycle convergence data accumulates in
`build/svm_corpus_v2/refinement_history.json` and successive
`uplift_report.json` files.

**Hold the structural-decisions conversation in parallel.**  The
metrology's `abandon_structural` outputs and the gate's per-code
regression table give the owner conversation concrete data points.
Decisions about catch-all codes and BILL/SHIP collapse should be made
before too many refinement cycles spend agent authorship time on
codes that are fundamentally unlearnable from value shape.

**Next-phase scope**:
- Corpus regeneration + refinement convergence trajectory
- Structural code-set decisions (owner conversation)
- DST discount recalibration against the new channel shape
- Standalone factorized NHSVM technical write-up for external publication
- Channel-independence audit pattern across the full runtime ensemble
