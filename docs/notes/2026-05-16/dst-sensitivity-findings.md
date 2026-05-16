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

Driver: `scripts/dst_sensitivity_study.py`.  6 invariants across 1,945
synthetic cells over a branching 11-node taxonomy (7 leaves, 4
internal).  Pure-CPU; ~80 ms total run time.

## Summary

| Invariant | Cells | Violations | Headline finding |
|---|---:|---:|---|
| Mass conservation | 441 | 0 | Clean — sum-to-1 holds to 1e-9 across the grid |
| Monotonicity wrt positive_score | 21 | 0 | Clean — belief is non-decreasing in target's positive |
| Anti-monotonicity wrt negative_score | 21 | 0 | Clean — singleton mass non-increasing as negative grows |
| Conflict K transit (channel-decomposed Dempster) | 441 | 0 | **K bounded at 0.24 across the grid** — by-construction observation worth surfacing (below) |
| Ranking stability under ε=0.01 perturbation | 1,000 | 0 | Confident configs perfectly stable; near-tie configs flip in proportion to margin (correct behavior) |
| Verifier_pass_rate effect | 21 | 0 | Monotonic with a notable crossover discontinuity (below) |

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
4. **Re-run this sensitivity study after the cautious-review
   ablation lands**, with the same harness against real
   enrichment-driven mass functions.  The synthetic cells exercise
   *form* correctness; the real cells will exercise *operational*
   calibration.

## Outputs

- `build/sensitivity/dst-2026-05-16T21-23-35Z/summary.json` —
  per-invariant rollup
- `build/sensitivity/dst-2026-05-16T21-23-35Z/violations.json` —
  empty list (no violations)
- This note: `docs/notes/2026-05-16/dst-sensitivity-findings.md`
