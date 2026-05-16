<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# DST Sensitivity Study — Findings

**2026-05-16, late-interaction cosine + channel-decomposed Dempster**

Driver: `scripts/dst_sensitivity_study.py`.  10 invariants across
2,549 synthetic cells over a branching 11-node taxonomy (7 leaves,
4 internal).  Pure-CPU; ~110 ms total run time.

The original 6 tests (P3.12) covered DST primitives: mass conservation,
monotonicity, K transit, ranking stability, verifier_pass_rate.  A
hierarchical-aggregation battery (P3.13) was added after operator
observation that the parent ↔ leaf accuracy regression (22-25 % of
errors in the running sweep) likely correlates with how
``_significant_subtree`` routes residual mass between leaves and
parent focal elements.  Two of the four new tests surface concerns
that match that correlation.

## Summary

| Invariant | Cells | Violations | Headline finding |
|---|---:|---:|---|
| Mass conservation | 441 | 0 | Clean — sum-to-1 holds to 1e-9 across the grid |
| Monotonicity wrt positive_score | 21 | 0 | Clean — belief is non-decreasing in target's positive |
| Anti-monotonicity wrt negative_score | 21 | 0 | Clean — singleton mass non-increasing as negative grows |
| Conflict K transit (channel-decomposed Dempster) | 441 | 0 | **K bounded at 0.24 across the grid** — by-construction observation worth surfacing (Finding 1) |
| Ranking stability under ε=0.01 perturbation | 1,000 | 0 | Confident configs perfectly stable; near-tie configs flip in proportion to margin (correct behavior — Finding 3) |
| Verifier_pass_rate effect | 21 | 0 | Monotonic with a notable crossover discontinuity (Finding 2) |
| Aggregation threshold cliff (`_significant_subtree`) | 441 | 0 | **Single-step parent_mass jump of 0.203** at sibling_p = 0.65→0.70 — the 0.50 concentration threshold is structurally a cliff (Finding 4) |
| Aggregation vs leaf competition | 121 | 0 | Top-1 leaf retains its mass dominance even when parent FE accumulates — predicted code stays at the leaf, but parent FE mass is operator-visible (Finding 5) |
| Internal-node top-1 switch | 21 | 0 | **0.57 single-step mass swing at internal_p = 0.65→0.70** as top-1 transitions from leaf to internal node — most dramatic discontinuity in the suite (Finding 6) |
| Aggregation under anti-example | 21 | 0 | Correct direction (parent_mass shrinks under negative on parent) but magnitude is tiny (Δ ≈ 0.0015) — same β=0.30 cap as Finding 1 (Finding 7) |

**Zero violations across the suite.**  The DST machinery satisfies its
expected mathematical invariants.  The interesting findings are
*constructional* — design choices whose operational consequences are
worth recording before they cause surprise.

## Finding 1 — β = 0.30 caps channel-conflict K at ≈ 0.24

The channel-decomposed Dempster construction (P3.6) combines the
positive channel ($m^+$ on singletons) with the negative channel
($m^-$ on complements $\Theta \setminus D(x)$ for top-K anti-example
tags).  The negative channel's total mass budget is bounded by
$\beta = 0.30$ (the `weight_anti_examples` config value).

Direct K measurement across a 21 × 21 grid of $(p_{\text{target}},
n_{\text{target}}) \in [0, 1]^2$ with default $\beta = 0.30$:

- **K range observed: $[0.000, 0.2400]$**
- **0 of 441 cells with K ≥ 0.5** (the log-warning threshold)
- **0 of 441 cells with K ≥ 0.99** (the Yager-fallback threshold)

The K-maximum cells cluster at $p = 1.0,\, n \in [0.10, \infty)$, with
$K = 0.24$ — close to $\beta \times m^+(\{x\}) \approx 0.30 \times 0.80$.

**Operational consequence:** the `INFO`-level channel-conflict log in
`late_interaction_to_mass` (fires when $K > 0.5$) will *never trigger*
under default configuration.  The Yager fallback path (engages at $K
= 1$) is similarly dead code under default $\beta$.

**Architectural choice this surfaces:** "channel conflict K as
operator-visible signal" — a load-bearing claim in the
DST-Reborn brief — does not naturally manifest at $\beta = 0.30$.  To
make K an actionable signal:

1. Raise $\beta$.  At $\beta = 0.85$ (the documented ceiling under
   verifier-attenuation), K would saturate to ~0.68 in the same
   adversarial corner; the 0.5 log threshold becomes reachable but
   not guaranteed.
2. Or, lower the log threshold.  $K > 0.20$ would fire frequently
   (~ 30 % of cells in this study) — possibly too noisy.
3. Or, redefine the signal entirely.  Operators may care less about
   raw K than about *positive-and-negative-both-strong* — a derived
   metric that doesn't depend on $\beta$ at all.

Recommendation: this is calibration work for the post-enrichment A/B
sweep.  Don't change $\beta$ blindly; measure what fraction of real
columns produce non-trivial $K$ at various $\beta$ levels and pick
the threshold from that distribution.

## Finding 2 — Verifier-pass-rate produces a top-1-crossover discontinuity

Sweeping `verifier_pass_rate` $\in [0, 1]$ on a target tag while
holding competitors at $\text{vr} = 1.0$:

| vr | singleton mass on target |
|---:|---:|
| 0.00 | 0.0328 |
| ... | gradual rise |
| ~0.42 | crossover region |
| ... | sharp jump |
| 1.00 | 0.8000 |

**Maximum single-step delta: 0.308 (Δmass for Δvr = 0.05).**

This is the margin-aware allocation kicking in: when target's
attenuated positive crosses the top competitor's positive, the
`top1_share = α × margin_w` allocation activates and target's mass
jumps discontinuously.  Below the crossover, target gets only its
softmax tail; above it, target gets the concentrated decisive-winner
share.

**This is by design** (margin-aware allocation per Haenni-Hartmann
2006 reliability shaping in `cosine_to_mass`), but operationally
relevant:

- Small changes to verifier_pass_rate near the top-1 crossover produce
  large jumps in operator-facing confidence
- Operators inspecting `cosine_attribution` may see counter-intuitive
  per-decision swings on annotations whose verifier confidence is
  marginal

**Mitigation already in code:** the verifier-pass-rate multiplier is
applied per-tag *symmetrically* to positive and negative scores
(P3.6), so a poorly-verified annotation goes quiet on both channels
rather than contributing biased one-sided evidence.  The discontinuity
is in *how confident an annotation's prediction is*, not in *which
prediction wins*.

**Recommendation:** no immediate change needed.  When the live A/B
sweep lands, examine the distribution of verifier_pass_rate values
the enrichment pipeline produces.  If many annotations cluster near
0.5 (the crossover region), consider smoothing the margin function
(e.g., longer `sigma_pos_marg`) to soften the discontinuity.

## Finding 3 — Ranking stability is graded by input margin (correctly)

Five base configurations × 200 perturbation trials each at ε = 0.01:

| Config | Description | Input margin | Top-1 instability |
|---|---|---:|---:|
| 0 | confident leaf | 0.550 | 0.00 % |
| 1 | cross-subtree near-tie | 0.020 | 0.00 % |
| 2 | positive + competing negative | 0.250 | 0.00 % |
| 3 | tight cross-subtree margin | 0.005 | **27.5 %** |
| 4 | three-way zero-margin tie | 0.000 | **67.0 %** |

This is **correct** behavior.  When inputs are within ε of a true
tie (configs 3 and 4), the perturbed rankings should flip in
proportion to how close the margin is to ε.  No brittleness beyond
what the margin predicts — the system is well-behaved.

**Operational note:** for the operator UX, this means columns whose
top-1 ranking is within ε of a tie should be surfaced as
`needs_clarification` regardless of their absolute confidence.  The
indep-tier revisit gate doesn't currently use top-1-margin as a
trigger — it uses cross-source disagreement mass.  Worth considering
adding margin to the revisit-fire condition in a future phase.

## Finding 4 — `_significant_subtree`'s 0.50 concentration threshold is a structural cliff

`mass_functions._significant_subtree` walks up from the leaf top-1 and
routes residual evidence mass to the most-specific ancestor whose
softmax descendant concentration meets a hard `concentration_threshold =
0.50`.  Below 0.50, no parent mass; at-or-above, the parent FE absorbs
the residual.  This is a step function by construction.

Test: a 21 × 21 sweep over (sibling_p, competitor_p) with top-1 leaf
fixed at 0.70 produces parent_mass in [0.000, 0.318]; **173 of 441
cells (39 %) carry non-zero parent_mass**.  Along the sibling_p axis
with competitor_p held at 0.30, the largest single-step jump is
**Δparent_mass = 0.203 between sibling_p = 0.65 and 0.70** — a one-
twentieth move in the input produces a fifth of the mass volume
re-routing to the parent FE.

**Operational consequence:** this is a plausible structural driver of
the parent_instead_of_leaf / child_instead_of_parent error cluster
(22-25 % of errors observed in the running sweep).  Columns whose
sibling concentration sits near the 0.50 boundary will produce
discontinuously different predictions under small input perturbations.
A column the cosine pipeline scores at concentration 0.49 surfaces no
parent signal; the same column scored at 0.51 surfaces ~0.20 of mass
on the parent FE.

**Mitigation options for the post-pivot phase:**

- *Smooth the threshold.*  Replace the hard `concentration >= 0.50`
  test with a sigmoid that ramps over a configurable width.  Same
  asymptotic behavior, no cliff.  Calibration question: what's the
  right sigmoid steepness?
- *Lower the threshold and re-discount.*  A lower threshold fires
  parent aggregation more often but allocates less mass per cell;
  net effect is more graceful.  Risks: more parent_instead_of_leaf
  errors when concentration is genuinely weak.
- *Make the cliff visible to operators.*  Surface
  `subtree_concentration` in the per-column result so operators see
  *why* the prediction landed where it did and can tune at the
  taxonomy level (split overly-dense subtrees, etc.).

## Finding 5 — Top-1 leaf retains mass dominance under reasonable sibling clustering

In an 11 × 11 sweep of (top1_p, sibling_p) over the operationally
realistic range (top1_p ∈ [0.5, 1.0], sibling_p ∈ [0.0, 0.5]),
**parent_mass exceeds leaf_mass in 0 of 121 cells**.  The top-1 leaf
always retains its position as the dominant focal element, even when
the parent FE picks up moderate aggregation share.

**This is reassuring** at the *top-1-prediction* level: the
hierarchical aggregation does not flip the predicted code from leaf
to parent at any realistic operating point.  The cliff in Finding 4
moves the *parent FE belief*, not the *predicted leaf*.

The operationally relevant consequence: operators inspecting the
`cross_subtree_belief` or `cautious_promoted_code` fields will see
parent-level mass appear or disappear with sibling concentration, but
the headline classification stays at the leaf.  This matches the
existing design: parent FE mass is a *disjunctive* signal ("the
answer is somewhere in this subtree") not a competing prediction.

## Finding 6 — Internal-node-as-top-1 switching produces the suite's largest discontinuity

P3.8 added explicit handling for internal-node tags as first-class
prediction targets.  When an internal-node tag receives strong direct
positive evidence (e.g., via the late-interaction `label_view` or
`parent_path_view` MaxSim), it can become top-1 — at which point
`_significant_subtree` is *skipped* (per Smets least-commitment, an
internal-node top-1 already represents the appropriate granularity).

Test: sweep `left_a` (internal node) positive_score from 0 to 1
while `left_a_1` leaf is fixed at 0.70 and competitors are moderate.
At `internal_p = 0.65` top-1 is the leaf; at `internal_p = 0.70`
top-1 switches to the internal node.

**The single-step transition:**

| internal_p | leaf_mass | internal_mass | top-1 |
|---:|---:|---:|---|
| 0.65 | (leaf-dominant) | (small) | leaf |
| 0.70 | drops by **0.5677** | rises by **0.2584** | internal |

A 0.05 step in input produces a **0.57 mass swing**.  This is the
largest single-step discontinuity in the entire suite — larger than
the cliff in Finding 4, larger than the verifier-pass-rate
discontinuity in Finding 2.

**Operational consequence:** the leaf/internal-node-as-top-1
transition is a high-volatility regime.  Two practical implications:

1. **Calibration of late-interaction positive weights matters.**
   If the late-interaction pipeline produces internal-node positive
   scores that hover near the leaf-top-1 crossover (e.g., because
   the parent's prototype values aren't sharply distinguishable
   from its descendant leaves' prototype values), predictions will
   be brittle and noisy.
2. **Operator UX should flag the regime.**  When `cosine_attribution`
   surfaces an internal-node top-1 with a leaf-runner-up close in
   score, an explicit "internal-node aggregation engaged" marker in
   the result dict would help operators recognize the regime.

**Mitigation for the post-pivot phase:** consider a *hysteresis* on
the top-1 kind decision — once top-1 is an internal node, require a
larger margin to switch back to leaf, and vice versa.  Smooths the
discontinuity without changing asymptotic behavior.

## Finding 7 — Anti-example on a parent correctly attenuates its mass, but tiny in magnitude

A test designed to verify that negative-channel evidence on a parent
internal node correctly carves out the descendant subtree (per P3.8).
Positive evidence clusters in `left_a`'s subtree; anti-example
score on `left_a` rises from 0 to 1.

| neg_on_parent | parent_mass on left_a FE | competing_subtree leaf_mass |
|---:|---:|---:|
| 0.0 | 0.0181 | 0.0026 |
| 1.0 | 0.0167 | 0.0034 |

**Direction is correct** (parent_mass shrinks, competing leaf rises)
but **magnitude is tiny** (Δparent ≈ 0.0015).  Same root cause as
Finding 1: β = 0.30 caps the negative channel's mass budget, so its
effect on any individual focal element is bounded.

The combination of Findings 1 + 7 has a sharper architectural
implication than either alone: **the negative channel is mechanically
correct but operationally weak under default configuration.**
Anti-example evidence on an internal node carves the descendant
subtree's *complement* with the right shape, but the magnitude of
mass redistribution is insufficient to flip predictions on its own.
The negative channel needs the positive channel to already be
ambiguous before it can tip the balance — anti-examples are a *tie-
breaker*, not a primary driver of classification.

**Implication for taxonomy enrichment:** the enrichment pipeline
should treat anti-examples as targeted-disambiguation evidence for
known-confusable cases, not as broad-coverage suppression of
generic-bucket misclassification.  An anti-example that says "this
column is NOT an instance of `generic_bucket` because it looks like a
status enum" needs to be paired with positive evidence on a specific
`status_enum` tag's prototype values; without the positive support,
the anti-example by itself won't move the operator-visible prediction.

## Hierarchical aggregation findings — collective interpretation

Findings 4, 5, 6, 7 jointly characterize the parent-vs-leaf dynamic:

- **The cliff at concentration 0.50** (Finding 4) is a real
  discontinuity that plausibly contributes to the observed
  parent ↔ leaf error pattern.
- **The leaf remains top-1** under the swept ranges (Finding 5) —
  but that's an *aggregate* result; specific input regimes could
  still produce parent-top-1 outcomes that the sweep doesn't cover.
- **The internal-node top-1 switch** (Finding 6) is by far the most
  volatile transition; its operational consequences depend entirely
  on whether the late-interaction pipeline produces internal-node
  positive scores in the unstable regime.
- **The negative channel can correctly suppress a parent** (Finding 7)
  but lacks the magnitude to flip predictions alone.

**Single-paragraph synthesis for the operator:** the parent ↔ leaf
mass distribution has two sharp boundaries (concentration cliff at
0.50, leaf-vs-internal-node top-1 crossover) that are by-construction
sources of brittleness near specific input values.  The negative
channel is too weak at default β to compensate.  The architecture is
*sound* — no math violations — but its *calibration* for parent ↔ leaf
disambiguation depends on the enrichment pipeline producing positive
evidence that stays clear of the cliff regions, plus on β being
tuned high enough that anti-examples can actually flip predictions
when the positive channel is borderline.

## Anti-patterns NOT found

For completeness, the following anti-patterns were specifically
tested for and did *not* manifest:

- Mass leak (sum diverging from 1.0): never observed.
- Non-monotonicity in either direction: never observed.
- Brittleness on confident configs: never observed.
- Verifier-pass-rate ramp going non-monotonic: never observed.
- Numerical instability at boundary inputs (0, 1, near-zero
  positive with non-zero negative): no NaN, no inf, no division-
  by-zero in the harness logs.

## What the study did *not* exercise

- Real embedding similarities (the study uses synthetic positive /
  negative scalars, not actual cosine output from sentence-transformers)
- The full pipeline fusion (positive_channel ⊕ negative_channel is
  measured directly, but the late-interaction mass's combination with
  LLM / CatBoost / SVM source masses isn't part of this study)
- Hierarchical aggregation interactions (one of the originally
  proposed tests, deferred — the existing BDD coverage in P3.9 /
  P3.10 / P3.11 already exercises hierarchical paths)
- The Anthropic-backed enrichment generator (the study runs on
  hand-crafted TagScore inputs; the actual generator is the deferred
  stub)

## Recommendations for the post-enrichment phase

1. **Make K a measured quantity, not just a logged side-effect.**
   Return $K$ from `late_interaction_to_mass` (or expose it via the
   attribution surface), so its distribution across real columns can
   be empirically characterized.
2. **Tune $\beta$ from the real distribution.**  Once enriched data
   lands, plot K-as-a-function-of-input across the corpus and pick
   $\beta$ so the channel-conflict log threshold catches the
   operationally-meaningful tail.
3. **Add top-1-margin to the revisit-fire condition.**  Columns whose
   ranking is within ε of a tie are unstable by construction and
   should be flagged for operator attention — separately from the
   indep-tier mass-disagreement gate.
4. **Smooth `_significant_subtree`'s concentration threshold.**
   Replace the hard 0.50 cliff with a sigmoid ramp.  Calibration
   width tunable from the real K / concentration distributions once
   they land.  Closes Finding 4 directly.
5. **Add hysteresis on the leaf/internal-node top-1 kind switch.**
   Once top-1 is an internal node, require a larger margin to switch
   back to leaf (and vice versa).  Smooths the volatile transition
   from Finding 6.
6. **Surface `subtree_concentration` and `top_kind` in the per-column
   result.**  Operators can then see *why* a prediction landed where
   it did — both at the concentration cliff and at the leaf/internal-
   node boundary.  No accuracy change; UX-only.
7. **Re-run this sensitivity study after the cautious-review
   ablation lands**, with the same harness against real
   enrichment-driven mass functions.  The synthetic cells exercise
   *form* correctness; the real cells will exercise *operational*
   calibration — and the parent ↔ leaf failure cluster (22-25 % of
   errors) should respond directly to the mitigations above if
   Finding 4 / 6 are indeed the dominant drivers.

## Outputs

- `build/sensitivity/dst-2026-05-16T21-23-35Z/summary.json` —
  per-invariant rollup
- `build/sensitivity/dst-2026-05-16T21-23-35Z/violations.json` —
  empty list (no violations)
- This note: `docs/notes/2026-05-16/dst-sensitivity-findings.md`
