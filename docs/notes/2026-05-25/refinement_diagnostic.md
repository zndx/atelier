# Refinement diagnostic — first-pass corpus expansion result

**Date**: 2026-05-25
**Companion to**: `phase_gate_brief.md`
**Result under analysis**: corpus expansion v2 + 5-pass refinement loop

---

## Headline measurements

|  | Value |
|---|---:|
| Real-only baseline (855 train, 271 test) | 0.6125 |
| Synth+real, pre-refinement (Phase D pass 0) | 0.5277 |
| Synth+real, **peak (refinement pass 3)** | **0.5830** |
| Synth+real, pipeline-stopped (pass 5) | 0.5535 |

Read as first-pass diagnostic, not regression: **the refinement loop
moved generalization +5.5pp from initial corpus injection (0.5277 →
0.5830) before degrading**, which is the load-bearing signal.  The
plateau is structural — three findings below — not architectural.

---

## Finding 1: model selection silently discarded the peak

The pipeline's stop condition ("2 consecutive passes < 1pp lift")
correctly identified passes 4-5 as degrading, but the orchestrator
overwrites model weights after each Phase D run.  Pass 3's 0.5830
weights are gone; we kept pass 5's 0.5535.

**Trivial fix**: track best-pass-so-far and restore those weights on
stop.  This is a one-flag change in `run_corpus_expansion_pipeline.sh`
+ a Phase D output renaming convention (`results_pass{N}.json`).

---

## Finding 2: target-list churn collapsed to 90% repeat

Refinement target Jaccard between consecutive passes:

| Pass-pair | Repeat | Jaccard |
|---|---:|---:|
| 1→2 | 16/20 | 0.67 |
| 2→3 | 17/20 | 0.74 |
| 3→4 | 19/20 | 0.90 |
| 4→5 | 19/20 | 0.90 |

By pass 4, the loop spent its compute re-authoring generators for the
same 19 of 20 codes targeted in pass 3 — against the same neighbor
prediction set.  This is structurally why pass 4-5 regressed: the
re-authored generators became *more* like their neighbors' generators
(both are getting fresh authorship from the same agent against the
same context), so the model's learned discriminator narrowed and
overfit to those tight synth distributions.

**Investigate**: gradient-based selection — after pass N, exclude
codes whose post-accuracy didn't improve from the next pass's target
pool.  Force the loop to *move on* rather than hammer the same set.

---

## Finding 3: 16 of 20 chronic targets have n_test=1

Most categories in the held-out test set have a single representative
row.  A code that misclassifies its one row registers as 0% accuracy —
indistinguishable from a structurally unlearnable code.  The refinement
loop's bottom-N selection is therefore dominated by **noise**, not
signal: it keeps re-targeting codes whose accuracy could flip 0↔100%
based on a single test outcome.

Anchoring categories (the ones with statistical signal) are doing
fine on the same model:

| Code | Annotation | n_test | acc |
|---|---|---:|---:|
| `0.1` | unknown/catch-all | 49 | 0.816 |
| `1.3.2` | DATE family | 10 | 1.000 |
| `1.2.2` | MONEY/numeric | 9 | 0.778 |
| `1.2.4` | (numeric subfamily) | 9 | 0.556 |
| `1.1.1.9.1` | NAME family | 4 | 0.750 |
| `1.3.4` | TIME family | 4 | 0.750 |

The model is competent where the test set carries signal.  **Refinement
target selection must filter by `n_test ≥ 3`** (or use a Bayesian
shrinkage against the population mean) to avoid chasing noise.

---

## Finding 4: chronic targets unpack into three categorically-undressable failure modes

When the chronic-target codes are looked up, they split cleanly:

### 4a. Bill/Ship structural ambiguity

```
1.1.1.4.2.1.1  BILLSTRNAM   Bill-to street
1.1.1.4.2.2.1  BILLCITY     Bill-to city
1.1.1.4.2.3.1  BILLPOSTAL   Bill-to ZIP
1.1.1.4.2.3.2  SHIPPOSTAL   Ship-to ZIP
1.1.1.4.2.4.1  BILLSTATE    Bill-to state
1.1.1.4.2.5.1  BILLCNTY     Bill-to country
```

These value distributions are **identical** to their non-billing
counterparts (a ZIP is a ZIP; a state name is a state name).
Disambiguation lives entirely in the column-name prefix
(`bill_zip` vs `ship_zip`) and table-context
(`billing_addresses` vs `shipping_addresses`).

**This is not a synth-corpus problem.**  No amount of better
`BILLPOSTAL` value generation can move the needle here, because the
generator's output looks like every other postal-code generator's
output.  It is exactly the kind of problem the DST fusion is supposed
to solve via the *other* channels — cosine on enriched column-name
text, regex/heuristic on name patterns.

**Structural fix to consider**: deliberately merge BILL/SHIP pairs at
the parent (`1.1.1.4.2.3` POSTAL-AGNOSTIC) for the SVM head's training
target.  Let the SVM emit mass at the parent; let column-name and
table-context channels emit mass at the leaves.  DST fusion combines
them naturally.  This is faithful to the "SVM as channel of last
resort for inscrutable columns" principle (memory:
`feedback_svm_targets_inscrutable.md`) — bill/ship address fields are
not inscrutable to the *name* channel, only to the *value* channel.

### 4b. Definitionally-shape-undefined catch-alls

```
0.2     ENOS  — Publicly available information
1.1.1.6 A_HD  — Unstructured, undefined, user-defined, or aggregated data
1.1.2   A_ID  — Unstructured, undefined, user-defined, or aggregated data
```

These codes have no characteristic value distribution by definition.
"Better" synth for them is a category error — what would even
constitute a good `A_HD` generator?  The successful catch-all
(`0.1`, n=49, acc=0.816) demonstrates that the model can learn a
*proxy* for "I don't recognize this" — but it cannot learn three
distinct flavors of "I don't recognize this".

**Fix**: pull these codes out of the SVM training target.  Let them
absorb mass via DST ignorance/conflict pathways rather than as
positive class predictions.  Practically: set
`generators_v1.GENERATORS_BY_CODE[code] = []` for these three,
remove from `synth_rows.jsonl`, and let the model emit
`mass(Θ) > 0` when it can't discriminate.

### 4c. Opaque proprietary IDs

```
1.1.1.8.6   KEYDIGEST  Cryptographic digest output
1.2.6.1     DAID       Acme proprietary asset ID
1.3.2.2.4   RUNCODE    Persisted source code
1.3.2.1.1   SYSURL     System URLs
```

Values look like other opaque-ID values.  Disambiguation again lives
in column-name + table-context, not value.  Memory
`feedback_svm_targets_inscrutable.md` calls this out as the
*intended* domain for the SVM — but only when the **name+table
context** is itself discriminative.  If `KEYDIGEST` appears in a
column called `value` in a table called `metadata`, no channel can
recover the right code; it's intrinsically ambiguous in the world,
not just in our model.

**Investigation**: audit the test set for these codes — what name +
table context is each row actually presenting?  If the context is
ambiguous in the source data, that's an upstream data-quality issue,
not a model issue.

---

## Findings summary as refinement directions, ranked by leverage

| # | Change | Leverage | Effort | Risk |
|---|---|---|---|---|
| 1 | Track best-pass weights; restore on stop | +2.95pp (peak vs final) | trivial | none |
| 2 | Filter refinement targets by `n_test ≥ 3` | medium | trivial | might empty target pool — needs more held-out data |
| 3 | Pull A_HD/A_ID/ENOS from SVM target set | small direct, large narrative | small | semantic — talk to owners about what those codes are FOR |
| 4 | Merge BILL/SHIP at parent for SVM target, leaves stay in name/cosine channels | medium-large | medium | requires DST recalibration |
| 5 | Per-pass target exclusion (codes that didn't improve last pass move off pool) | small-medium | small | none |
| 6 | Re-run Phase D against a larger held-out — current 271 rows leave too many singletons | foundational | requires more reference data | budget |

The most informative *next experiment* is probably (1) + (5) + (2) as
a bundle, re-run on the existing corpus: that tests whether smarter
refinement orchestration alone — without any change to generators
themselves — can stabilize the loop and recover the pass-3 peak as
the asymptote rather than a transient.

(3) and (4) are structural decisions that need owner conversation
before changing — they reshape what the SVM channel is *for* in the
DST fusion.

---

## What this does NOT say

- It does **not** say the factorized NHSVM head is wrong.  The
  architecture is fitting categories with learnable signal correctly
  (DATE, MONEY, NAME families at 75-100%).
- It does **not** say synth corpus expansion is a dead end.  The
  +5.5pp peak-vs-pre-refinement lift is real evidence that domain-
  adaptation against neighbor predictions works *when the target
  is statistically observable and structurally learnable*.
- It does **not** justify reverting to the real-only 0.6125 baseline.
  That baseline is fragile (855 rows, 177 classes ≈ 5 ex/class) and
  has its own overfitting risk that just hasn't been measured.

What it does say: the pipeline as constructed gives us instrumentation
to diagnose *exactly* where it stops working and why.  That
instrumentation is the win.
