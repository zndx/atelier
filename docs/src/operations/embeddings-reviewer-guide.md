# Reviewer's Guide to the Embeddings Canvas

This guide is for operators auditing classification runs and proposing
algorithm-tuning remediations.  It explains the Dempster–Shafer (DST)
measures the canvas exposes, the rationale behind the curated SQL
Predicate panel, and a concrete walk-through of using the canvas to
diagnose the four root causes called out in
[`audit_2026-05-06_a.md`](../../../audit_2026-05-06_a.md) (runs
`40f07630`, `8d67b1ed`, `e5b0ac26`).

The guide assumes you have an Embeddings page open for one of those
runs and a copy of the audit alongside.

---

## 1. The DST measures, in plain English

Atelier fuses up to six independent evidence sources (name match,
pattern, cosine, LLM, CatBoost, SVM) via Dempster's rule of
combination.  The fused result is a **mass function** over the
taxonomy's frame of discernment.  From that mass function we report
five scalars per column:

| Field         | Formula                              | Meaning |
|---------------|--------------------------------------|---------|
| `belief`      | `Bel(A) = Σ m(B), B ⊆ A`             | **Lower bound** on the probability the prediction is correct.  Mass committed *only* to A or its subsets. |
| `plausibility`| `Pl(A)  = Σ m(B), B ∩ A ≠ ∅`         | **Upper bound**.  Mass *consistent with* A — what hasn't been ruled out. |
| `uncertainty` | `Pl(A) − Bel(A)`                     | The width of the [Bel, Pl] interval.  Epistemic uncertainty, smaller is better. |
| `confidence`  | `BetP(x) = Σ_{x∈A} m(A) / |A|`       | **Pignistic probability** (Smets 1990).  A point inside [Bel, Pl] that uniformly spreads ignorance across singletons.  The decision-theoretic transform — the right number for *acting* on the prediction. |
| `conflict`    | K, the pre-normalization mass on ∅   | Source-disagreement diagnostic.  Under Dempster's rule, K is normalized out of [Bel, Pl] but logged separately; under Yager, K is redirected to ignorance (Θ). |

The invariant: **`Bel(A) ≤ BetP(A) ≤ Pl(A)`** for every column.
Pl + Bel of A's complement always equals 1 (duality).

### Which one is the "rigor" signal?

For a *single positive scalar*, prefer **`belief`**.  It is the
honest floor — mass that cannot be redirected by additional evidence
even in principle.  The cautious-review gate (`bel_threshold = 0.80`)
operates on Bel; bootstrap convergence is on the gap (Pl − Bel);
`needs_clarification` fires when Bel < 0.80 OR gap > 0.20.  The
project's algorithms already treat Bel as the truth proxy; the
reviewer should too.

`confidence` is **not** redundant — it serves a different purpose.
BetP redistributes ignorance optimistically, so a vacuous mass
function over a 16-singleton frame still produces BetP ≈ 0.06 per
singleton.  Comparing `belief` to `confidence` on the same row is
how reviewers build intuition for "how much of this column's
prediction is committed evidence vs. evenly-spread ignorance."  Big
gap between Bel and BetP = the prediction looks confident only
because the rest of the frame is empty.

A worked example from `8d67b1ed`'s row 1 (`fitness_members.row_id`):

```
Bel = 0.834   BetP = 0.834   Pl = 0.933   K = 0.358
```

BetP and Bel align tightly because the mass is concentrated on
singletons (no large compound focal elements to spread).  A
healthy, committed prediction.

Compare with a hypothetical weak prediction:

```
Bel = 0.30    BetP = 0.55    Pl = 0.85    K = 0.10
```

The same headline `confidence = 0.55` masks a Bel of 0.30 — meaning
70% of the mass is sitting on compound focal elements that BetP is
spraying across singletons.  Reviewer's read: this is *not* an 0.55
prediction; it's a 0.30 prediction wearing a 0.55 hat.

### Why `conflict` is no longer the canvas color default

Under Dempster's rule (the default fusion strategy), K is normalized
out of [Bel, Pl] — every fused mass function is renormalized by
`(1 − K)`.  K still gets reported as a diagnostic, but it does not
correlate with prediction quality.  Run `8d67b1ed` averaged
K = 0.27 across all 287 columns; rows with very different beliefs
(Bel = 0.30 vs Bel = 0.85) commonly share the same K.  Coloring by
K painted the canvas a nearly-uniform fog.

`belief` paints the canvas with information.  Low-Bel rows (the
cautious-review candidate pool) cluster in warm colors; committed
predictions cool out.  The 0.80 cliff that drives the cautious
review and `needs_clarification` is *visible* — it's the threshold
between a calm canvas and a hot-spot region that demands human
attention.

If you switch the run to Yager fusion, K is no longer normalized
out — it shows up as ignorance mass, which depresses Bel and widens
the gap.  Reviewers comparing fusion strategies side-by-side should
look at Bel + gap on both, not at K — K means different things
under the two rules.

---

## 2. The curated SQL Predicate panel

Embedding-Atlas's default behavior is to auto-generate one chart per
data column.  With 35 fields in the parquet, that's noise: tooltips
overlap with the canvas, projection coordinates render as histograms,
JSON blobs render as illegible text fields.

The curated panel exposes only fields that map to an algo-tuning
decision.  Order is intentional — top to bottom, the panel walks
the reviewer from "is this run healthy" → "where is the pain
concentrated" → "which feature is driving it."

| # | Field | Chart shape | Why it's there |
|---|-------|-------------|----------------|
| 1 | `belief` | Histogram | Primary quality signal.  The 0.80 cliff is the cautious-review threshold; rows below are the candidate pool.  Brushing this filters the canvas to "weak" predictions. |
| 2 | `confidence` | Histogram | BetP.  Side-by-side with `belief` builds intuition for the Bel-vs-BetP gap.  Wide gap on a row = mass concentrated on compound focal elements. |
| 3 | `review_decision` | Count plot | Categorical: `keep` / `backoff` / `reroute` / `""` (untouched).  This is the *audit's central concern* — Finding 1 names reroute as the instability amplifier. |
| 4 | `predicted_annotation` | Count plot | Compact mnemonic (e.g. `NAMEFULL`, `EMAIL`, `PHONE`) — the dot-codes are unreadable in a small chart, but the annotation tells the same story.  The full label appears in the embedding tooltip. |
| 5 | `needs_clarification` | 2-bar count | Boolean union of `Bel < 0.80 OR gap > 0.20`.  The "demands attention" set, expressed as a single flag. |
| 6 | `llm_confidence` | Histogram | LLM's self-reported confidence.  Low-tail rows are the population at risk for reroute amplification — a weakly-asserted LLM code that DST then has to defend. |
| 7 | `uncertainty` | Histogram | Pl − Bel — gap-driven revisit set.  Bootstrap convergence is on `mean(uncertainty)`; canvas histogram lets reviewers see whether the run actually converged or just hit max-iterations. |
| 8 | `conflict` | Histogram | K, demoted from default but kept as a source-disagreement diagnostic.  Useful when comparing Dempster vs Yager runs (K means different things under each). |
| 9–14 | `shap_top1/2/3_name`, `shap_top1/2/3_value` | Count + histogram pairs | Surfaces *which feature is driving each prediction*.  Top-1 is usually `sample_values`; top-2/3 reveal sibling-context vs column-name dominance.  The intentional inclusion of all three reflects the **steep dropoff in SHAP utility** between top-1 and top-3 — the dropoff is itself the situational signal.  When top-1 dominates by 5×, single-feature explanations work; when top-1/2/3 are flat, the prediction is broadly diffuse and remediation needs to address feature-engineering, not source weights. |
| 15 | `table_name` | Count plot | Hotspot navigation.  Audit calls out `legal_cases` and `loan_applications` as hallucination concentration zones; this chart lets reviewers brush-filter to one. |
| 16 | `column_type` | Count plot | Numeric vs object.  Pattern-signal source is type-conditioned; reviewing remediations to pattern detectors benefits from typed slicing. |

**What's not in the default panel.**  Reference fields
(`reference_code`, `reference_label`, `matches_reference`) are usually
empty for production Hive data.  When a run *does* have a curated
reference set (UAT meta-tagging mounts), reviewers can add
`reference_code` and `matches_reference` via the **SQL Predicates**
control on the panel header — type into the predicate input directly,
or click "Add" and pick the column.  The panel re-renders instantly.
Same mechanism applies to any field the reviewer wants ad-hoc — e.g.,
`predicted_label` for a long-form taxonomy view, or `predicted_code`
when a numeric dot-code is needed for filtering.

---

## 3. Walk-through against `audit_2026-05-06_a.md`

The audit identifies four root causes.  Below: how to reach each one
on the canvas, what the right brushing pattern is, and what the
algo-tuning lens reveals.

### Finding 1 — Three-way reroute as instability amplifier

> "20.6% of columns flip between runs with identical configuration.
> The reroute mechanism turns a minor LLM fluctuation into a major
> classification change."

**Brush:** `review_decision = "reroute"` on chart 3.

**What you see:** the canvas highlights all rerouted rows.  Look at
their distribution — are they clustered in one taxonomy region (a
single subtree's entropy bleeding into neighbors), or are they
scattered (the LLM is fluctuating uniformly)?

**Cross-brush** with `belief` chart 1, brushing the 0.40–0.70 band:
this is the cohort that fails the 0.80 threshold but isn't trivially
weak.  Reroute decisions are most consequential here — an LLM fluke
on a 0.85-Bel row gets rejected by the threshold; on a 0.65-Bel row
it gets handed to the reviewer.  The audit's recommended P1 guard
("reject reroutes where pre-review code matches LLM code with
conf > 0.80") would visibly clip the right edge of this brush.

**Algo-tuning read:** if rerouted rows cluster around `belief` ≈ 0.5
and have `llm_confidence` > 0.80, the audit's P1 guard is the right
remediation.  If they cluster at `belief` < 0.4, the upstream issue
is fusion strength, not the reviewer.

### Finding 2 — LLM annotation-code hallucination (27 columns)

> "The LLM returned annotation mnemonics (`SSN`, `DOB`, `FNAME`)
> instead of numeric taxonomy codes for 27 columns in 40f07630.
> When `llm_code` is an annotation string, the code-resolution layer
> discards it — `evidence_sources.llm = {}`."

**Brush:** `llm_confidence` chart 6, isolate the *low-confidence
tail* below 0.20.  These are columns whose LLM evidence was
discarded (the evidence layer assigns 0 confidence when the code
fails to resolve).

Cross-brush with `belief` (chart 1) — affected columns will pile up
at low Bel because they fall back to cosine alone.

**SHAP signal:** chart 9 (`shap_top1_name`) on the same brush should
show `column_name` or `sibling_context` dominating instead of
`sample_values`.  When SHAP's top-1 is *not* `sample_values`, the
classifier wasn't given enough evidence from the values themselves —
the LLM-evidence loss is showing up as an upstream feature-importance
shift.

**Algo-tuning read:** the audit's P0 ("map annotation mnemonics to
numeric codes in `_resolve_llm_code()`") would *eliminate* this brush
entirely — its impact is visible as the disappearance of a low-tail
cluster on `llm_confidence`.  Reviewer can size the impact:
"≈ 27 columns × mean(belief gain) = X total mass committed."

### Finding 3 — `col_04` and sibling-context poisoning

> "When sibling opaque columns (`col_02`, `col_32`) are all
> misclassified as Shipping Address (because the table name biases
> the embedding), the reviewer uses those wrong sibling labels as
> evidence to perpetuate the error."

**Brush:** `shap_top1_name = "sibling_context"` on chart 9, then
cross-brush `column_name LIKE 'col\\_%'` via the SQL Predicate
input.  This is the at-risk population.

**What you see:** the rerouted opaque columns cluster on the canvas
near their (incorrectly inferred) neighbors.  When the reviewer-bias
poisoning is at work, these clusters will show consistent
`predicted_annotation` *across* the cluster — the error has
propagated.

**Cross-brush** with `review_decision = "reroute"`: rerouted opaque
columns where SHAP shows sibling-context dominance are the precise
target of the audit's P2 remediation ("exclude sibling columns with
opaque names from reviewer context").

**Algo-tuning read:** the size of this brush is the population the
P2 remediation removes.  If SHAP top-2 and top-3 (charts 11, 13)
are also dominated by sibling-context for these rows, the value-side
evidence is *systematically* under-represented and the remediation
needs to extend beyond "exclude opaque siblings" to "rebalance feature
weights when sample-values entropy is high."

### Finding 4 — Baseline 20% non-determinism

> "Between e5b0ac26 and 8d67b1ed — identical configuration, same
> dataset, 5 hours apart — 59 of 287 columns (20.6%) changed their
> final `predicted_code`.  This establishes the non-determinism
> floor."

This finding is cross-run; one canvas can't render it directly.  But
it manifests on the canvas as **confidence vs belief gap** dispersion.
A run with high non-determinism has many columns where `confidence`
diverges from `belief` — these are the rows whose mass is spread
across compound focal elements rather than committed to singletons,
making them sensitive to small evidence perturbations.

**Brush:** in the SQL Predicate input, type `confidence - belief > 0.15`.
The canvas highlights the diffuse-mass cohort.  These are the rows
most likely to flip on the next run.

**Algo-tuning read:** the audit's P2 (raise `bel_threshold` from
0.80 to 0.85–0.90) tightens the cautious-review entry criterion —
fewer borderline rows enter review, fewer reroutes amplify.  Brushing
`belief` ∈ (0.80, 0.85) shows the population the threshold raise
*removes* from review, which is also the high-flip-rate population.
A 5-point threshold raise on the 8d67b1ed canvas removes ≈ 30 columns
from the review pool — pre-computable from the histogram.

---

## 4. Algo-tuning playbook

When you arrive at a fresh canvas, walk top to bottom:

1. **Healthy run check:** `belief` chart, look at the mass below
   0.80.  If < 10% of the corpus is below the cliff, the run
   converged comfortably.  If > 25%, something upstream (LLM,
   alignment, vocab) is weak.
2. **Reroute pressure check:** `review_decision`, count the
   `reroute` bar.  Reroutes >  20% of total = the reviewer is
   doing too much work; consider raising `bel_threshold` (audit P2)
   or constraining the shortlist.
3. **LLM-evidence integrity check:** `llm_confidence`, look for
   the < 0.20 tail.  Population of that tail = approximately the
   annotation-hallucination cohort (audit P0).
4. **Feature-driver check:** `shap_top1_name`, see whether
   `sample_values` dominates.  When it doesn't, the prediction is
   leaning on schema/sibling context — fragile.
5. **Hotspot triage:** `table_name`, see whether failures
   concentrate in a small number of tables.  Per-table failure
   patterns often point to vocabulary alignment gaps that affect
   only certain domains (legal, financial, medical).

The remediations the audit recommends should each have a *visible
signature* on the canvas.  When you propose a fix, predict where on
the canvas the fix will land — and verify against the next run.

---

## 5. Configuration reference

The curated panel is configured in
[`ui/src/pages/Embeddings.tsx`](../../../ui/src/pages/Embeddings.tsx)
via the `defaultChartsConfig` prop on the `EmbeddingAtlas` component.
The `category` field on the embedding spec sets the canvas color;
the `include` array sets the predicate panel contents and order.

Reviewers needing different fields for a one-off audit can use the
**SQL Predicate** control at the top of the predicate panel — type a
SQL expression directly (e.g.
`predicted_code = '1.1.1.9.1' AND review_decision = 'reroute'`) and
brush the result.  The expression composes with all other brushes on
the canvas.

For permanent additions to the default panel, edit the `include`
array; the order in the array is the order in the panel.  Avoid
adding high-cardinality fields (`column_name`, `evidence`,
`embedding_text`) — they render as illegible count plots.

---

## Further reading

- [Classification Pipeline](../architecture/classification.md) — how the six evidence sources are produced.
- [DST Evidence Independence](../architecture/dst-evidence-independence.md) — why source independence matters for the rigor of these measures.
- [Embeddings](../architecture/embeddings.md) — the data flow that produces the parquet feeding this canvas.
- [`audit_2026-05-06_a.md`](../../../audit_2026-05-06_a.md) — the worked example this guide is structured around.

---

# Addendum — Remediation Paper-Trade Observations (2026-05-06)

This addendum captures observations from the static validation of the
algo-tuning playbook against `8d67b1ed`'s parquet, and the paper-trade
of each `audit_2026-05-06_a.md` remediation against the same run.
It is intended both as honest documentation of what worked vs. what
needed adjustment, and as the calibration baseline against which the
post-remediation validation run will be evaluated.

## A.1 Playbook validation findings

Walking the playbook brushes against `8d67b1ed/atelier_embeddings.parquet`
surfaced three corrections to the original guide:

### Correction 1 — `BetP − Bel` brush is empty in practice

The playbook section on Finding 4 (baseline non-determinism) prescribes
brushing `confidence - belief > 0.15` to find the diffuse-mass cohort.
On `8d67b1ed`:

```
mean(BetP - Bel) = 0.0007
rows with gap > 0.15 = 0
rows with gap > 0.05 = 0
```

**Why:** mass concentrates on singleton focal elements in this corpus,
so the pignistic transform has nothing to redistribute.  BetP ≈ Bel
everywhere.  The theoretical intuition (BetP optimistically spreads
ignorance) is sound but only manifests when significant mass lives on
compound focal elements — rare in production runs.

**Replacement brush** for the non-determinism cohort: **`uncertainty > 0.20`**
(Pl − Bel above the cautious-review gap threshold).  This *does*
populate; in `8d67b1ed`, 199/287 rows (69%) carry uncertainty > 0.20,
so for narrowing purposes pair it with `belief < 0.6` to focus on
the genuinely weak predictions.

### Correction 2 — `bel_threshold` direction is the opposite of the audit's claim

The audit's P2 recommendation says:

> Raise `bel_threshold` from 0.80 to 0.85-0.90 → reduces candidate pool

This is **mechanically false**.  The threshold gates *entry* to cautious
review (`cautious_review.py:454`: `if bel < bel_threshold`); raising it
strictly enlarges the candidate pool.  Measured on `8d67b1ed`:

| `bel_threshold` | Candidates |
|---:|---:|
| 0.80 | 199 / 287 (69.3%) |
| 0.85 | 239 / 287 (83.3%) |
| 0.90 | 255 / 287 (88.9%) |

**Decision:** R4 is deferred.  R1 (annotation-mnemonic recovery)
materially lifts Bel for the 33%-of-corpus cohort with previously-empty
LLM evidence; the candidate-pool size after R1 may make a threshold
adjustment unnecessary.  Re-evaluate after the validation run.

### Correction 3 — Audit's "16 hallucination cases" undercount

Audit Finding 2 cites "~16 cols hallucinate annotation in 8d67b1ed."
The actual count of rows whose LLM evidence is absent from the fused
mass is **95 / 287 (33%)** — six times the audit's number.  The
discrepancy is partly because the audit conflated two distinct cases:
true mnemonic emission (which R1 recovers) and LLM voting at a parent
focal element (which `_mass_summary` filters out of the singleton-only
`evidence_sources.llm` field even though the mass is fully present in
the fused result).  The latter is *not* a hallucination — it's an
observability artifact in `_mass_summary`.

**Implication for the canvas:** rows with `evidence_sources.llm = {}`
should not be read as "LLM contributed nothing."  When the row's
`llm_code` is non-empty and falls inside the runtime taxonomy, the
LLM voted at an internal node and contributed mass through that
parent FE.  The brush is more honest as a *resolution-failure*
indicator: filter to rows where `llm_code is non-numeric AND
evidence_sources.llm is empty` to find the genuine mnemonic cohort.

> **2026-05-07 update — R10:** `_mass_summary` now surfaces internal-
> node FEs alongside singletons.  Internal-node entries carry a
> trailing `*` (e.g. `"1.1.1.9.4*": 0.65`) to distinguish "parent FE,
> mass spread across descendants" from a singleton-leaf vote.  The
> singleton-only filter is gone — `evidence_sources.llm = {}` now
> means *the LLM produced no code we could map at all*, which is the
> intended semantics.

## A.2 Per-remediation paper-trade results

Each remediation was paper-traded against `8d67b1ed` after
implementation.  Predicted impact in the leftmost column comes from
the audit's recommendation; observed impact is what the paper-trade
measured.

| ID | Remediation | Audit's predicted impact | Paper-traded impact | Note |
|----|-------------|--------------------------|---------------------|------|
| **R1** | Annotation-mnemonic fallback in `_resolve_to_focal_element` | "Recovers LLM evidence for 27 cols, ~20% reroute candidate reduction" | **38 / 287 columns (13.2% of corpus)** recover full LLM evidence; mean `llm_confidence` on recovered cohort = **0.89**; 5 hr_compensation columns concentrate on `EMPDET → 1.1.1.2.5.3 (Employment Related)` from scattered low-Bel predictions | Significantly higher than audit's 27.  Recovery is concentrated in tables with rich user-vocab mnemonics (`EMPDET`, `PANEXP`, `SHIPADDR`, `BIN`). |
| **R6** | Skip Hive/Hue temp tables (`__tmp_*`) at discovery | (not in audit) | **1 table dropped** (`hue__tmp_ecommerce_orders`); 16 cols removed from classification, 9 of which were R1-recovery candidates → net R1 impact after R6: **29 cols** | R6 supersedes R1 for those 9 cols (correct: temp tables shouldn't classify at all).  Net cohort R1 actually recovers in next run = 29. |
| **R2b** | Markdown-fence + extra-data extraction in `_parse_decision` | "Eliminates 3-5 hard errors per run" | **3 / 11 errored decisions** in `8d67b1ed` had the markdown-fence-with-trailing-prose shape; new `_extract_json_object` parses them cleanly (verified against captured response) | Audit estimate accurate. |
| **R2c** | Shortlist-permissive parsing (NEW — audit conflated with R2b) | (not separately specified) | **8 / 11 errored decisions** rejected codes that *were* valid in the runtime taxonomy but outside the 5-entry shortlist. R2c accepts these as `shortlist_extended` reroutes | Audit's "11 errors" summary should split into two classes; R2b alone would only catch 3/11. |
| **R3** | Exclude opaque siblings (`col_NN`, `var_NN`, `dim_NN`, …) from reviewer context | "Prevents sibling-context poisoning" | Cohort visible in `8d67b1ed` is small (1 rerouted, 2 candidates) because filter was ON; in 40f07630 (filter off) the cohort is 13+ | Paper-trade limited by which run is on hand.  Validation run will need filter OFF or include opaque-name tables to size the impact. |
| **R2a** | Stability guard on cross-subtree reroutes | "Prevents the gaming_profiles.handle failure class" | Three iterations: v1 (naive: pre==llm ∧ conf>0.80) blocked **20 / 64 reroutes**, including legitimate depth corrections.  v2 (top-level-root differs) blocked **5 / 64** but missed sideways moves within the `1.x` namespace.  v3 (neither-is-ancestor — current implementation) blocks **12 / 64** — all visibly cross-subtree, with depth corrections preserved | Audit framing assumed all "LLM+fusion agreed" reroutes are noise; in practice 15 such reroutes were within-subtree backoffs (e.g., `1.1.1.8.2 → 1.1.1.8`).  The neither-is-ancestor rule cleanly separates these. |
| **R5** | Split `llm_agreement` into pre/post-review metrics | "Makes overwatch signal useful" | Purely additive; new `llm_agreement_pre_review` field reports DST-vs-LLM alignment without review reassignment confounding | Diagnostic only; no impact on classification outcomes. |
| **R4** | Raise `bel_threshold` 0.80 → 0.85-0.90 | "Reduces candidate pool" | **Deferred** — see Correction 2.  Audit direction is mechanically wrong.  Re-evaluate after R1 lifts Bel | Expected outcome: post-R1, the threshold may not need adjustment; if it does, the right direction is *down* (0.65-0.70). |

## A.3 Predicted canvas signatures for the validation run

What to look for on the post-remediation canvas to verify each
remediation landed:

| Remediation | Predicted canvas signature |
|---|---|
| **R1** | `belief` histogram shifts right — the mode of the < 0.5 cluster moves toward 0.7-0.8 (LLM evidence now contributing).  The hr_compensation table (5 cols, all currently scattered) collapses onto a single `predicted_annotation` value (`EMPDET`). |
| **R6** | Total column count drops by ~16 (the `hue__tmp_ecommerce_orders` columns).  `table_name` count plot loses one bar. |
| **R2b** | `cautious_review.json`'s `errored` count drops by ~3.  Bedrock-deployed runs benefit most. |
| **R2c** | `cautious_review.json`'s `errored` count drops by ~8 (combined with R2b: total errored drops to 0-1).  New `shortlist_extended` counter in summary > 0. |
| **R3** | `cautious_review.json` row records show `siblings_after_filter < siblings_unfiltered` for tables containing `col_NN` columns.  Reroutes whose rationale referenced sibling labels (e.g., the `col_04 → Shipping Address` case) lose that justification. |
| **R2a** | `cautious_review.json` summary shows `stability_guard_fired > 0`; the guard's blocked reroutes show up as `decision = "keep"` with rationales prefixed `[R2a stability guard fired: ...]`.  Brush by `review_decision = "keep"` AND `review_rationale LIKE '[R2a%'` in the SQL Predicate panel to count. |
| **R5** | Overwatch report's Health Signals table gains a row; `llm_agreement_pre_review > llm_agreement` when reviewer reassigned LLM-aligned predictions. |

## A.4 What the paper-trade cannot validate

- **Cumulative interaction effects.**  R1 raises Bel for 38 cols, which
  changes which cols enter cautious review, which changes the
  shortlist composition for those cols, which changes whether R2c's
  permissive path fires.  Static paper-trade can't model this cascade.
- **Real LLM behavior in cautious review.**  R2c assumes the LLM
  occasionally picks valid-but-out-of-shortlist codes; the true rate
  may differ once the run uses the post-R1 frame (more LLM evidence
  → fewer cautious-review entries → smaller cohort exposed to R2c).
- **Bedrock vs Anthropic-direct response shapes.**  R2b was
  smoke-tested against one captured Bedrock fence-with-prose case;
  other Bedrock formatting variants (mid-stream JSON, Latin-1
  whitespace, multi-block responses) are unobserved in the dataset.
- **R3 sibling-context poisoning size on this corpus.**  With
  `classify_exclude_reference_columns = true` (8d67b1ed's setting),
  the col_04-class cohort is suppressed at discovery; the validation
  run should toggle this off (or include opaque-name tables) if the
  goal is to measure R3's true impact.

## A.5 Expected delta on overwatch's Health Signals table

Pre-remediation (`8d67b1ed`):

| Signal | Configured | Actual | In Contract? |
|---|---|---|---|
| `llm_agreement` | ≥ 0.9895 | 0.6794 | ❌ No |
| `state.failed_columns` | ≤ 2 | 11 | ❌ No |

Post-remediation (validation run prediction):

| Signal | Configured | Predicted | In Contract? |
|---|---|---|---|
| `llm_agreement` (post-review) | ≥ 0.9895 | ~0.85 (R1+R2c+R3+R2a all push it up) | ❌ Still under, but materially closer |
| `llm_agreement_pre_review` (R5, NEW) | (no contract) | ~0.92 | — |
| `state.failed_columns` | ≤ 2 | 0-1 (R2b + R2c eliminate parser/shortlist failures) | ✅ Yes |
| `total_columns` | — | 271 (was 287; R6 drops 16) | — |
| Cohort with empty `evidence_sources.llm` | — | ~57 (was 95; R1 recovers 38) | — |
| `stability_guard_fired` (R2a, NEW) | — | ~12 | — |
| `shortlist_extended` (R2c, NEW) | — | ~8 | — |

If the post-validation overwatch report shows `llm_agreement` still
sub-0.80, the residual gap is in the 55-column "numeric-unresolved"
cohort that R1 doesn't touch.  That points to a follow-up
remediation — likely a frame-coverage gap where the LLM emits codes
the runtime taxonomy doesn't carry.

## A.6 Configuration

Each remediation is gated by an independent flag, so a follow-up A/B
run (if any signature is missing or wrong) can isolate per-remediation
contribution by toggling one flag at a time.

| Flag | Default | Disable for ablation |
|---|---|---|
| `classify.resolve_llm_annotation_mnemonic` | `true` | R1 off |
| `classify.exclude_temp_tables` | `true` | R6 off |
| `classify.cautious_review.shortlist_permissive` | `true` | R2c off |
| `classify.cautious_review.exclude_opaque_siblings` | `true` | R3 off |
| `classify.cautious_review.stability_guard_enabled` | `true` | R2a off |
| `classify.cautious_review.stability_guard_llm_conf` | `0.80` | R2a threshold |

The R2b parser improvement is *not* flag-gated — it's strictly more
correct than the prior greedy regex on every input.

## A.7 Test surface

Unit tests in `tests/classify/test_audit_remediations.py` cover R1,
R2b, R2c, and R6.  R2a and R3 are paper-traded against
`build/results/8d67b1ed/cautious_review.json` rather than unit-tested
because their value lives in cohort behavior (cross-subtree
distribution, sibling filtering effects), not single-decision
transforms.  R5 is a metric addition with no decision logic to test.

```bash
PYTHONPATH=src python3 -m pytest tests/classify/test_audit_remediations.py -v
# 19 tests, all passing as of 2026-05-06.
```
