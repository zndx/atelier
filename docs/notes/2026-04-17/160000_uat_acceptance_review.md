<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# UAT Acceptance Review — Default DB, 2026-04-16

> **Post-hoc correction (2026-04-19):** this review pre-dates the
> authoritative ground truth at
> `build/meta-tagging-clean/ground_truth.csv`.  The "reference
> annotations" mentioned below refer to UAT's provisional labels,
> which carry three known bugs we've since corrected (reference-
> column name leakage, sibling-context leakage, name-index
> parent-vs-leaf mis-mapping).  The 93% baseline cited in this
> review was scored against UAT's buggy labels — see
> `build/results/parity/delta_report.md` for the re-scored
> comparison against our authoritative GT (Atelier 94.56% exact,
> Gopala 65.27% exact, Δ +29.29).

UAT feedback workbook: `build/Atelier_Results_Default_DB_4-16.xlsx`
(gitignored). Environment: Bedrock-only (no direct Anthropic API,
so no overwatch).

**NB**: the reference annotations and data tables used for this UAT are
private (held at `~/local/tmp/meta-tagging/`) and must never be checked
in or quoted in notes/code. This review documents Atelier-side failure
modes only — no ontology labels, annotation mnemonics, or sample
values are reproduced.

## Headline

| Metric | Atelier | Baseline (third-party LLM) |
|---|---|---|
| Overall accuracy | **36.2%** | 93.0% |
| Best per-table | 90.5% | 100% |
| Worst per-table | **3.7%** | 47.1% |

Eight tables evaluated; 455 classified columns; ~222 leaf terms in
the loaded vocabulary (out of 297 candidate codes — non-leaf codes
are filtered by `_build_reference_categories`).

Notable: one table where Atelier *beat* the baseline by ~40 pts —
there's signal in the architecture, but distribution variance is
punishing.

## Failure-mode breakdown

From the per-table sheets' 455 predictions:

- **65.9% of predictions had confidence < 0.30** — evidence fusion is
  producing vacuous mass functions on most columns.
- **137 columns (30.1%) were labeled with the vocabulary's catch-all
  fallback** (`0.1` family) — the pipeline produced no committable
  evidence for them.
- Pattern misfires confirmed on digit-heavy ID columns: 16-digit card
  numbers predicted as SSN, 12-digit bank-account numbers predicted
  as phone numbers (both at confidence ≈ 0.04 — i.e. cosine-only with
  a misleading pattern prior).
- Clear-named columns ("driver-license-shaped-name", "passport-shaped-name")
  landed on a generic catch-all ID term at confidence ≈ 0.19 — this
  implies name-match evidence did **not** fire strongly even though
  the candidate terms are in the loaded vocabulary and labels are
  lexically aligned with the column names.

## Regression diagnosis

Two mechanisms, both introduced after the previously-reported
near-perfect run:

### 1. Monte-Carlo stratification is suppressing LLM coverage

`classify.monte_carlo.min_corpus_size = 200` → MC engages on any
corpus ≥ 200 columns. 455 columns × `sample_fraction = 0.15` + 3 per
stratum ≈ 68–90 columns reach the LLM. The remaining ~370 columns
rely on cosine + SVM + name-match + pattern alone — which, per the
fallback rate above, is insufficient at this vocabulary density and
name ambiguity.

The previous near-perfect run would have run with full LLM sweep (no
MC branch taken).

**Confirmation**: this morning's overwatch analysis on Synthetic
flagged *exactly* this gate as the dominant accuracy lever
(`mc_sample_fraction` 0.15 → 0.35 projected +6 pts there). The
pattern recurs at larger scale and with deeper vocab: the effect is
amplified here.

### 2. Name-match isn't winning the fusion on clear names

`_camel_to_words("Drivers License Number")` → `"drivers license number"`
and the exact-match branch is triggered by a column name of
`drivers_license_number` (per direct test — see
`src/atelier/classify/mass_functions.py:320–341`). Yet the prediction
landed on a generic catch-all ID with confidence 0.195, not the
specific term.

Two plausible explanations (not yet repro'd on local code):

- The vocabulary loaded at UAT time used the **annotation mnemonic**
  (short code) as `cat.label` rather than the **ontology name** (the
  human-readable string). In that case `cat_words` for the matching
  term would be the abbreviated code, not the phrase — and the exact
  branch fails. The loader at `taxonomy.py:375–384` prefers
  `ontology` over `annotation`, but if the Hive row had an empty
  `ontology` field that fallback kicks in.
- Cosine + SVM combined could out-mass a 0.70 singleton when many
  near-siblings exist (the 222-leaf vocab has dense ID-shaped
  clusters). Without seeing the live belief-path for the failing
  columns I can't attribute weighting definitively.

### 3. Pattern prior distortion

Confirmed from morning's overwatch audit (saved as `build/results/a22f1f10/overwatch.md`):
- `phone_pattern` fires on 12-digit strings → commits to "Other Phone
  Number" at high singleton mass, contaminating bank-account / ID
  columns.
- `ssn_pattern` (per the audit) has zero *wrong* fires on Synthetic
  but the UAT results suggest it (or an equivalent 9+ digit pattern)
  is claiming 16-digit card numbers. Needs a re-audit with the UAT
  value distribution.
- `license_plate_pattern` regex `^[A-Z]{2,3}[-\s]?\d{3,4}$` shouldn't
  match typical driver-license values (7+ digits), but the broader
  pattern family lacks suppression for alpha-prefix identifiers.

## Remediation plan

Prioritized by expected impact × operator effort. Everything in Phase
A is now tunable from the Settings page we landed today — no
redeployment required.

### Phase A — Settings overlay (zero code, ~5 min)

On the UAT run's active dataset, apply:

| Setting | From | To | Rationale |
|---|---|---|---|
| `classify.monte_carlo.sample_fraction` | 0.15 | **0.35** | More columns reach LLM |
| `classify.monte_carlo.min_per_stratum` | 3 | **5** | Sparser strata better covered |
| `classify.bootstrap.clarity_target` | 0.10 | **0.20** | Aggressive re-sweep |
| `classify.discounts.name_match_exact` | 0.70 | **0.55** | Exact-name hits wield more mass |

Or effectively disable MC by raising `min_corpus_size` to 1000. A 455-
column corpus then takes a full LLM sweep at ~$5–10.

Expected delta: catches most of the 30% fallback bucket if the LLM
reaches them. Conservative projection: +25–35 pts → 60–70% overall.

### Phase B — Pattern map quarantine (1 hour)

From the morning overwatch audit:

- Remove net-harmful entries from `DEFAULT_PATTERN_MAP` /
  `PATTERN_VALIDATORS` in `features.py`: at minimum `phone_pattern`,
  `date_iso_pattern`, `datetime_iso_pattern`, `vin_pattern`,
  `license_plate_pattern`, `iata_pattern`, `eth_address_pattern`.
- Or, preferred: add **suppression rules** mirroring the phone-pattern
  suppression block at `features.py:312–341`, extending it so that
  any alpha-prefix identifier value suppresses phone and SSN patterns
  for the column.

Either path is regression-tested by the tier-0 pattern BDDs; a new
scenario for "12-digit BAN doesn't match phone_pattern" belongs in
`features/agent/classification.feature`.

Expected delta: +1–3 pts on Synthetic, likely larger on UAT because
digit-heavy ID columns are densely represented.

### Phase C — Name-match instrumentation (2 hours)

Add a per-column trace (behind a debug flag) that logs, for each
classified column:
- the column-name's normalized form
- every term tied for `best_mass` after the scan
- the final `(best_code, best_mass)`

Running this once against the UAT reference reveals whether the
"User ID" landing is a mass-weight tie-break issue or a vocabulary
loader issue. If it's loader-side, fix forks to `_build_reference_
categories` to handle empty-`ontology` rows. If it's mass-weight,
the Phase A discount change will resolve it.

### Phase D — Bedrock LLM latency + context budget (only if Phase A
underperforms)

Bedrock Opus on 50-col batches with 50 candidate codes per column
and 298-term vocab is pushing ~30 kB context per call. If the UAT
run timed out silently on large batches, half the columns never got
an LLM vote. Investigate by:
- Running with `classify.llm.columns_per_call = 25` (half the batch)
- Raising `classify.llm.max_retries` to 5
- Auditing the gateway logs for 408/529 failures during the UAT run

All three tunable from Settings without redeployment.

## What to run first

1. Open `/settings` on the UAT environment.
2. In **Sampling** tab: set `mc_sample_fraction = 0.35`, `mc_min_per_stratum = 5`.
3. In **Convergence** tab: set `clarity_target = 0.20`.
4. In **Evidence & Fusion** tab: set `name_match_exact = 0.55`.
5. Activate the UAT data source and kick a classification run.
6. Compare the resulting classifications to the UAT xlsx. Overwatch
   is unavailable (Bedrock), so validation is manual — use the same
   per-table accuracy columns the UAT evaluator produced.
7. Expect 60-70% overall on that first retry. Anything below 50%
   → Phase C instrumentation before further tuning.

## What not to do

- **Don't** add the meta-tagging reference data to `data/`, `fixtures/`,
  or `build/data/annotations/annotations.json`. The cache currently
  holds a generic 24-term vocab; UAT loads its real 222-leaf vocab
  from Hive at runtime. Keep it that way.
- **Don't** commit the UAT xlsx or any per-column traces derived
  from it. `build/` is gitignored but step carefully when adding new
  analysis scripts — they shouldn't embed reference values.
- **Don't** conclude the architecture is wrong from this single UAT
  outcome. The transaction_data table shows Atelier at 88% vs
  baseline 47% — DST fusion is a real win when it engages. The
  problem is coverage, not the core method.

## Links

- UAT xlsx: `build/Atelier_Results_Default_DB_4-16.xlsx` (gitignored)
- Morning overwatch on Synthetic: `build/results/a22f1f10/overwatch.md`
- Settings plan: `/home/rch/.claude/plans/streamed-booping-naur.md`
- Mass functions: `src/atelier/classify/mass_functions.py:301–390`
- Taxonomy loader: `src/atelier/classify/taxonomy.py:364–428`
- Pattern map: `src/atelier/classify/features.py:25–105`
- MC sampling: `src/atelier/classify/monte_carlo.py`
