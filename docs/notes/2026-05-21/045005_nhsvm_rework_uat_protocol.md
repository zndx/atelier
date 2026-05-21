<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# NHSVM rework — UAT empirical validation protocol

**Status:** for UAT consideration; no production change recommended pending
empirical confirmation against real annotations.

**Authoring artifacts:** `scripts/audit_nhsvm_adversarial.py`,
`build/audit/nhsvm/sensitivity_sweep.json`,
`src/atelier/classify/svm_classifier.py` (current production NHSVM).


## Purpose

A code audit of the training-time NHSVM identified a geometry mismatch
between the production implementation and Choi et al. (2015) Eq. 5.
A local sensitivity analysis against a synthetic adversarial taxonomy
characterized three candidate variants across four DST factor sweeps.

Under the DST framework's operational intent — *combine signals without
doing harm; let the LLM reason its way to success on the second look* —
the analysis concludes the production implementation (Variant A) is
correctly calibrated for the framework's iterative-reconsideration role.

This doc frames the empirical questions that UAT, with access to real
annotations and production traffic, can answer to confirm or refute
that conclusion before any rework is committed.


## Background — what the audit found, and why it was reframed

The production training-time NHSVM uses one-vs-rest LinearSVC trained
on label-conditional Kronecker-expanded features and applies a
universal-expansion at inference (every node block populated by
`sqrt(α_n) * x`).  Choi et al.'s Structured Shared Frobenius Norm
formulation implies a structured-output SVM with per-class expansion
at inference (`argmax_y <w, Λ(y) ⊗ x>`).

The audit proposed two alternatives: Variant B (same training, per-class
inference) and Variant C (Crammer-Singer joint multi-class training,
per-class inference — the closest sklearn-expressible approximation to
the paper's formulation).

The local sensitivity analysis revealed two things:

1. **Variant C is mathematically tighter but operationally identical to B.**
   Across all sweeps and all tiers, C's accuracy and probability
   distributions tracked B within numerical noise.  The geometry
   mismatch is captured entirely by inference-time expansion; the
   training-objective choice (OvR vs joint) adds nothing measurable.

2. **A and B are not on a Pareto frontier — they have mirror failure
   modes that share a parameter region.**  No single DST tuning knob
   resolves the tradeoff (yager === dempster; raising SVM discount
   kills both good and bad SVM-pulls; LLM-discount sweep shows the
   tradeoff is symmetric in parameter space).

The reframe that landed the conclusion: under the DST framework's
*actual* purpose — invite the LLM to reconsider with additional
evidence — Variant A's confident SVM voice is the design feature,
not a defect.  A's "overpulling" on semantic-conflict cases is the
mechanism by which productive friction surfaces hard cases for
human consideration and triggers bootstrap revisit.  Variant B's
softer voice mutes that signal below the design point and produces
*silent under-classification* — the harm the framework was built to
avoid.


## Three variants under test

| label | training | inference at column x | status |
|---|---|---|---|
| **A** | one-vs-rest LinearSVC on `Λ(y_i) ⊗ x_i` features | universal expansion (every block populated by `√α_n`) → `predict_proba` | **PRODUCTION** |
| **B** | same as A | per-class expansion (only `path(y)` blocks active per candidate `y`), normalized across candidates | candidate for swap-in |
| **C** | Crammer-Singer joint multi-class LinearSVC on the same expanded features | per-class expansion (as B) | tighter-math candidate — dropped: ≡ B on outcomes |


## Local analysis results (synthetic, 23-case adversarial test set)

Default parameters: `svm_discount=0.22`, `llm_confidence=0.85`,
`llm_discount=0.15`, `fusion=dempster`.

| tier | n | A acc | B acc | C acc | notes |
|---|---|---|---|---|---|
| easy / hard / contested / sparse | 15 | 1.00 | 1.00 | 1.00 | variants indistinguishable on baseline cases |
| semantic-conflict (reframed) | 3 | 1.00 | 0.33 | 0.33 | values commit → A pulls correctly; B/C hold at parent |
| svm-was-right | 5 | 0.80 | 0.00 | 0.00 | uninformative column name + clear values → A pulls correctly |
| **overall** | **23** | **22/23 = 96%** | **16/23 = 70%** | **16/23 = 70%** | |

Inflection-point summary (the parameter value at which each variant's
per-tier accuracy crosses 0.5; `↑` = increasing accuracy with parameter,
`↓` = decreasing):

| tier | variant | svm_disc | llm_conf | llm_disc |
|---|---|---|---|---|
| semantic-conflict | A | ↑ 0.30 | ↑ 0.95 | ↓ 0.10 |
| semantic-conflict | B/C | ↑ 0.25 | ↑ 0.85 | ↓ 0.20 |
| svm-was-right | A | ↓ 0.30 | ↓ 0.95 | ↑ 0.10 |
| svm-was-right | B/C | ↓ 0.20 | ↓ 0.80 | ↑ 0.25 |

A's parameter region of correctness on both tiers is the same window
(`svm_disc < 0.30`, `llm_conf < 0.95`).  B's region is narrower and
doesn't include the production defaults on the svm-was-right tier.

Fusion strategy: dempster ≡ yager (zero difference at every cell —
LLM/SVM conflict on these cases doesn't reach the regime where the
strategies diverge).


## What UAT needs to answer empirically

The synthetic test set is hand-curated and small (n=23).  The relative
frequency of the failure-mode tiers in real traffic is unknown.  Three
empirical questions need real-data answers:

### Q1.  Tier distribution in production traffic

How often does each failure-mode tier actually occur on UAT columns?

- *Semantic-conflict frequency:* columns whose name carries a
  directional or role qualifier (origin-, destination-, source-,
  target-, primary-, secondary-) AND whose values commit to a leaf
  with a contradictory role connotation.
- *SVM-was-right frequency:* columns with uninformative names
  (field_N, col_N, attr_N, generic prefixes) AND values that
  unambiguously commit to a specific leaf.

The synthetic test had these at 3:5.  If real traffic has them at
10:1, the analysis flips — B's failure mode becomes negligible and
A's becomes dominant.  If 1:10, the opposite.  This ratio is the
single most important empirical input.

### Q2.  Bootstrap revisit gate sensitivity

When A pulls a confident SVM leaf against an LLM parent vote, does
the disagreement actually trigger an iteration-2 LLM revisit?

The gate (`Pl − Bel` gap, K conflict, `top1_margin` disjoint margin)
needs to fire on the semantic-conflict and svm-was-right cases for
the DST dialectic to complete.  If it doesn't fire, A's productive
friction is wasted and the design intent isn't being realized.

### Q3.  Iteration-2 LLM update rate

When the LLM does revisit under SVM-LLM disagreement, does it
update its commit, or does it double down on iteration 1?

LLMs are known to exhibit self-consistency bias; the second look
may not actually change the answer.  If the LLM doubles down, the
reconsideration mechanism isn't producing the value the framework
promises, and A's design choices (loud SVM voice, calibrated to
trigger revisit) need a different downstream change to pay off.


## Protocol — running the audit script against UAT data

The script `scripts/audit_nhsvm_adversarial.py` is structured to be
re-pointed at real data with surgical changes:

1. **Real taxonomy:** replace `build_adversarial_taxonomy()` with a
   loader that reads `working_set.json` (or whatever vocab artifact
   the UAT deployment registered with the active taxonomy registry).

2. **Real training data:** replace `generate_training_data(...)` with
   `load_enrichment_payloads(cfg=cfg)` + `generate_user_taxonomy_corpus(...)`
   to produce the corpus the production NHSVM cache would use.

3. **Real adversarial test set:** sample columns from UAT traffic
   into the same 5-tuple `(name, expected, tier, values, llm_vote_code)`
   shape.  For each:
   - `easy`/`hard`/`contested`/`sparse`: pull from known-classifiable
     columns where the LLM's answer is uncontroversial
   - `semantic-conflict`: pull columns where the column name carries
     a directional/role qualifier and the LLM voted at parent
   - `svm-was-right`: pull columns with uninformative names where the
     LLM voted at parent but the values clearly commit to a leaf

4. **Re-run the sensitivity sweeps.**  The output structure
   (`build/audit/nhsvm/sensitivity_sweep.json`) is unchanged.
   Inflection points become *real-data* inflection points.

5. **Cross-check the bootstrap revisit firing rate** (Q2) by
   instrumenting `bootstrap.py` to log when the gate triggers on
   columns in the semantic-conflict and svm-was-right tiers.

6. **Measure the iteration-2 LLM update rate** (Q3) by comparing
   iteration-1 and iteration-2 LLM commits on the revisit set.


## Decision criteria

**Default decision:** keep Variant A.  Switch to B (and skip C
entirely; the local analysis already ruled it out) only if UAT data
shows ALL of:

1. The semantic-conflict failure mode is materially more frequent
   than svm-was-right in real traffic (ratio ≥ 3:1 in favor of
   semantic-conflict).

2. The bootstrap revisit gate's firing rate on A's semantic-conflict
   disagreements is < 30% — i.e., A's friction isn't getting the
   LLM to reconsider in the cases that need it.

3. Customer-visible incidents over the rc2 soak window are
   correlated with A's semantic-conflict false-positives, not with
   B's silent under-classification.

If any of those three conditions fail, A is the correct variant.

If all three hold, B is worth a UAT-only swap.  C remains off the
table — joint training adds no measurable accuracy and pays
implementation cost.


## Out of scope for this protocol

Two adjacent concerns surface naturally during this analysis but
are not NHSVM-rework questions:

### Taxonomy semantic-loading

Leaf labels like `shipping-address` carry implicit role connotations
(shipping → destination) that create friction no inference calibration
can resolve.  The `origin_doc → shipping-address` production
observation that triggered this whole review is, fundamentally, a
taxonomy-vocabulary disagreement: "is `shipping-address` the address
used FOR shipping (which can be either endpoint) or the destination
address OF shipping?"  No SVM change answers this; vocabulary tightening
does.  Tracked separately.

### Second-look mechanism robustness

The DST framework's operational value depends on the LLM reconsidering
effectively on iteration 2 under disagreement.  Q2 and Q3 above
characterize whether the revisit + reconsideration loop is doing
its job.  If it isn't, that's a separate dev-effort target — likely
larger than any NHSVM rework — and no inference-geometry change makes
up for a broken second look.


## References

- `scripts/audit_nhsvm_adversarial.py` — local sensitivity analysis;
  three variants, four sweeps, inflection-point detector
- `build/audit/nhsvm/sensitivity_sweep.json` — raw sweep data
- `src/atelier/classify/svm_classifier.py` — current production NHSVM
  (`HierarchicalFeatureExpander.expand_with_labels` /
  `expand_universal`, `nhsvm_reweight`, `SVMClassifier._fit_hierarchical`)
- `src/atelier/classify/mass_functions.py` — `svm_to_mass`,
  `nhsvm_to_mass`, `llm_to_mass`
- `src/atelier/classify/belief.py` — `HierarchicalClassification.from_combined_evidence`,
  `top1_margin`
- Choi, J. et al. (2015). *Hierarchical Multi-Label Classification
  Using Hierarchical Loss Functions.*  arXiv:1508.02479.
- `docs/src/appendix/sprint-2026-05-20.md` §1 — sprint summary of
  training-time NHSVM landing
