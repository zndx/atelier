# Extend Classification Workflow

End-to-end procedure for classifying a Hive corpus that grows over
time: train CatBoost on the stable subset, then *extend* the trained
model to newly-added tables without re-running the full LLM-driven
classification pipeline.

This report documents the procedure and the empirical results from a
session on 2026-05-13 against the
`hive-poc/reference_corpus` source (reference data-governance POC, 40
tables, ~620 columns), running with the Phase-3 DST frame and the
LLM-emission validation + retry mechanism enabled.

---

## Why two-phase classification

A full classify run uses LLM sweeps, multi-source DST fusion, and
cautious-review on top of CatBoost training — minutes to tens of
minutes per 300-column batch with non-trivial LLM cost. An *extend*
run reuses a previous run's CatBoost (and optionally UMAP / SVM)
and applies them directly to new columns — seconds to a couple of
minutes regardless of corpus size, no LLM cost.

The pattern lets data-governance teams:

- Establish a stable baseline classification on the tables they
  already know
- Onboard new tables incrementally without re-running expensive LLM
  sweeps
- Compare new-table predictions against a known model artifact for
  audit and consistency

Empirically, on the corpus we measured, the *extend* output actually
**scored higher** on the operator-flagged ground-truth proxy than
the parent classify run (71.9% strict vs 68.1%) — the cautious-review
backoff in the full pipeline turned out to be over-conservative on
this corpus. The workflow below establishes both runs so you can
compare them and pick the artifact that best matches your governance
team's expectations.

---

## Prerequisites

| | |
|---|---|
| Atelier deployment | CAI Application or local devenv with `cml.data_v1` access |
| Hive source | A `data_sources` row registered for the corpus (e.g. `hive-poc/reference_corpus`) |
| Annotations table | Deployed at `<connection>.<cfg.classify_database>.annotations` (typically `<connection>.default.annotations`) — *not* colocated with the data tables |
| Config | `config/base.conf` editable, or env-var overrides for the toggles below |
| LLM backend | Configured via `ANTHROPIC_API_KEY` / Bedrock credentials so the classify-phase sweep can run |

The classify and extend runs are triggered from the UI's pipeline
panel or via `POST /api/fsm/start` and `POST /api/fsm/extend`
respectively.

---

## The config knobs that drive the workflow

Two HOCON settings under `classify { … }` in `config/base.conf`:

### `classify.table_exclude_patterns`

Comma-separated regex patterns matched against Hive table names
(`re.search` semantics, case-sensitive). Tables whose name matches
any pattern are dropped at `discover_tables` time and never sampled
— same mechanism applies uniformly to classify and extend pipelines.

Empty (default) = no filtering. Operator edits this between runs.

### `classify.svm.enabled`

When `false` (current default), the per-vocabulary SVM evidence
source is skipped — the alignment LLM call doesn't fire, no SVM is
trained, and the pipeline runs with 5 evidence sources instead of
6. Toggle back to `true` after the recipe-driven synth training
described in `docs/src/architecture/...` (separate workstream)
replaces the LLM-mediated alignment.

Both also have env-var overrides
(`ATELIER_CLASSIFY_TABLE_EXCLUDE_PATTERNS`,
`ATELIER_CLASSIFY_SVM_ENABLED`) that take precedence over the HOCON
defaults at load time.

---

## Procedure

### Step 1 — Identify the "stable" subset of the corpus

Decide which tables you want CatBoost to train on. The pattern is
typically: "tables that have been in production long enough to have
operator-validated classifications." Newly-added tables go in the
*excluded* set.

For the documented session, the stable subset was the 20 tables that
existed in the previous classify baseline (`5450b626`), with 20 new
tables added to Hive after that.

Identify the newly-added tables by diffing the current Hive table
list against a previous run's classifications:

```bash
python3 << 'EOF'
import json
from pathlib import Path

parent_run = '5450b626'  # or whichever prior run defines your baseline
new_run = 'f931f469'     # a fresh run that classified the post-addition full source

parent_tables = sorted({c['table_name'] for c in
    json.loads(Path(f'/home/cdsw/build/results/{parent_run}/classifications.json').read_text())
    if c.get('table_name')})
new_tables = sorted({c['table_name'] for c in
    json.loads(Path(f'/home/cdsw/build/results/{new_run}/classifications.json').read_text())
    if c.get('table_name')})

added = sorted(set(new_tables) - set(parent_tables))
print(f'Added tables: {len(added)}')
for t in added:
    print(f'  + {t}')
EOF
```

For each new table, build a fully-anchored regex pattern
(`^name$`) so a future table named e.g. `member_registry_v2` doesn't
accidentally get caught by a pattern targeting `member_registry`.

### Step 2 — Filter the new tables before the classify run

Edit `config/base.conf` to populate `classify.table_exclude_patterns`
with the comma-separated regex list:

```hocon
classify {
  …
  table_exclude_patterns = "^app_developer_records$, ^compliance_documents$, ^component_catalog$, ^contact_supplemental$, ^content_profiles$, ^credential_vault$, ^device_identity_log$, ^engagement_signals$, ^headcount_ledger$, ^health_location_profiles$, ^member_registry$, ^order_shipments$, ^payment_events$, ^program_index$, ^return_billing$, ^screening_records$, ^security_research_assets$, ^staff_registry$, ^system_audit_records$, ^workforce_data$"
  table_exclude_patterns = ${?ATELIER_CLASSIFY_TABLE_EXCLUDE_PATTERNS}
  …
}
```

Or as a single-line env override in `.env.cai.enc`:

```
ATELIER_CLASSIFY_TABLE_EXCLUDE_PATTERNS="^app_developer_records$, …, ^workforce_data$"
```

Verify the config loads correctly:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from atelier.config import load_config
cfg = load_config()
print(f'{len(cfg.classify_table_exclude_pattern_list)} patterns:')
for p in cfg.classify_table_exclude_pattern_list:
    print(f'  {p}')
"
```

### Step 3 — Restart the Application to pick up the new config

In the CAI Workspace UI, **Application → Restart**. The pipeline
loads HOCON values fresh on each `load_config()` call, but the
in-memory Python module cache for `_HOCON_MAP` is initialized once;
a restart guarantees both layers see the new config.

### Step 4 — Run the parent classify against the stable subset

Trigger from the UI's pipeline panel, or:

```bash
curl -s -X POST "$ATELIER_BASE_URL/api/fsm/start" \
  -H 'content-type: application/json' \
  -d '{"source_id": "hive-poc/reference_corpus"}'
```

Expected:
- `discover_tables` enumerates all tables in Hive, drops the
  excluded set, returns the stable subset
- The pipeline runs end-to-end on the filtered set: LLM sweep,
  DST fusion, fit-to-LLM CatBoost training, cautious review,
  SHAP/SAGE if enabled
- Run dir lands at `build/results/<run_id>/` with the full artifact
  set (CatBoost CBM, classes JSON, UMAP, parquet, classifications,
  evaluation_report, etc.)
- Run kind: `classify`. Artifact set: same id as `run_id`.

Note the `run_id` of this baseline — it becomes the
`artifact_set_id` for the extend run.

#### What you should see in `validation_retries.json`

```json
{
  "total_retries": 1-5,  // small number is healthy
  "events": [
    {
      "column_names": ["..."],
      "invalid_codes": ["A_FD", "1.2.1.3.3", ...],
      "retry_idx": 0
    },
    …
  ]
}
```

Each entry is a column where the LLM emitted a code that's not in
the deployed `default.annotations` taxonomy. The retry mechanism
re-prompted the LLM with the specific invalid code named, and the
LLM (almost always at `retry_idx: 0`) emitted a valid code on the
second attempt. After-exhaustion blanking (residual invalid
emissions getting `category_code = None`) is rare; if it happens,
those columns are simply dropped from CatBoost training data.

Empty `events: []` means the LLM emitted only in-taxonomy codes
throughout the sweep — the goal state.

### Step 5 — Clear the filter before the extend run

Edit `config/base.conf`:

```hocon
classify {
  …
  table_exclude_patterns = ""
  …
}
```

Or unset the env var. Restart the Application again.

### Step 6 — Run extend against the artifact from Step 4

```bash
curl -s -X POST "$ATELIER_BASE_URL/api/fsm/extend" \
  -H 'content-type: application/json' \
  -d '{
        "source_id": "hive-poc/reference_corpus",
        "artifact_set_id": "<parent_run_id>",
        "parent_dataset_id": "<parent_run_id>"
      }'
```

Or trigger from the UI's Extend panel against the artifact set
matching the parent's `run_id`.

Expected:
- `discover_tables` enumerates all 40 tables (no filtering)
- `sample_table_metadata` samples each
- The parent run's CatBoost predicts `predict_proba` on every column
- No LLM sweep, no DST fusion, no cautious review — straight
  CatBoost top-1
- Run dir at `build/results/<extend_run_id>/` with parquet,
  classifications, evaluation_report
- Run kind: `extend`. References the parent via `artifact_set_id`
  and `parent_dataset_id`

#### A real cost in elapsed time

For a 40-table / ~620-column corpus, the extend run completes in
roughly 2–3 minutes (dominated by Hive metadata sampling). Compare
to the parent classify which takes 10–30 minutes depending on LLM
batch latency.

---

## Caveats observed during the session

### The annotations database is NOT colocated with the data tables

The deployment has data tables at `hive-poc.reference_corpus` but
the canonical taxonomy at `hive-poc.default.annotations`. The full
classify pipeline handles this via `cfg.classify_database`
(defaults to `"default"`) and an optional `vocab_uri` on the
`data_sources` row. The extend pipeline must do the same — early
in the session a regression was found where extend was querying
`<data_db>.annotations` (which doesn't exist), silently catching
the exception, and producing output with `predicted_annotation`
empty and `predicted_label` echoing `predicted_code`. The fix at
`src/atelier/classify/extend_pipeline.py` reads from
`cfg.classify_database` for annotations, independent of the
data-tables database resolved from `source_id`.

### `validation_retries.json` is the audit trail

Any LLM emission outside the deployed taxonomy is captured in
`build/results/<run_id>/validation_retries.json` with the column
name and the invalid code. Empty events list = clean sweep. The
audit lives alongside the run artifacts so post-mortem doesn't
require pod-log access.

### Cautious-review backoff can be over-conservative

On the documented corpus, the parent classify's cautious-review
mechanism backed off 15 columns from terminal predictions to
parent codes that the extend run subsequently recovered as
correct terminals. The threshold knob
(`classify.cautious_review.bel_threshold`, default 0.80) is the
lever; tightening it to 0.85 or 0.90 will reduce the rate of
backoffs.

### Re-running classify with the filter restored is cheap regression-protection

If the extend output looks worse than expected on the OLD tables,
the parent's artifacts are unchanged and re-deploying is one config
edit + restart. Both runs land in `build/results/` and are
independently auditable.

---

## Results from the 2026-05-13 session

Five classify+extend runs were measured against the same
operator-curated review spreadsheet
(`Atelier-Results-vs-Prompt-solution-522d89ae.xlsx`), which
encodes one operator's expected classifications for the 20 OLD
tables. Three metrics matter:

- **Strict (canonical-validated)** — `predicted_annotation` matches
  the spreadsheet's expected tag, validated against
  `default.annotations` so spreadsheet hallucinations don't count
  as Atelier misses
- **Stem-collapsed** — same as strict but ignoring `A_`/`C_`/`S_`
  prefix differences within a code's annotation family
- **Binary sensitive-vs-public** — predicted sensitive vs non-sensitive
  matches spreadsheet's `Data Sensitivity` field
- **Operator-curated recall** — 15 columns the operator explicitly
  flagged as "Atelier got this wrong"; recall counts how many now
  resolve correctly

| Run | Notes | Strict | Stem | Binary | Op-curated |
|---|---|---|---|---|---|
| 522d89ae | Original baseline (pre-Phase-3, pre-validation) | 69.1% | 44.6% | 84.2% | 0/15 |
| 5450b626 | Pre-Phase-3 retrain (filtered to 20 OLD tables) | 66.7% | 42.8% | 83.2% | 3/15 |
| 1d6e3fae | Phase 3 only (full DST frame, no validation+retry) | 67.4% | 42.1% | 83.9% | 3/15 |
| 2ac4d0a6 | **Phase 3 + validation+retry classify** | **68.1%** | 43.2% | **84.6%** | **4/15** |
| 0146134f | **Phase 3 + validation+retry extend (from 2ac4d0a6)** | **71.9%** | **47.0%** | **84.6%** | **7/15** |

### Three distinct improvements

1. **Validation+retry catches the parent classify up.** 2ac4d0a6
   over 1d6e3fae: +0.7pp strict, +1 op-curated. Driven by the 3
   LLM hallucinations the new mechanism caught and corrected in
   real-time (`A_FD` on monetary columns, `1.2.1.3.3` on
   `case_ref`).

2. **Extend's CatBoost-only path materially outperforms the parent's
   full pipeline.** 0146134f over 2ac4d0a6: +3.8pp strict, +3
   op-curated. Surprise: extend lacks DST fusion and cautious
   review, yet scores higher — the parent's cautious-review backoff
   was over-conservative on this corpus.

3. **Op-curated recall climbs across the whole arc.** 0/15 →
   7/15 over the session's work, without ground-truth supervision
   or model changes — just architectural correctness improvements
   (Phase 3, validation+retry, correct annotations database in
   extend).

### Column-level diff (0146134f vs 2ac4d0a6 on the OLD 20 tables)

```
Of 300 shared OLD-table predictions:
  unchanged:                  263 (88%)
  leaf → parent (regression):   3 (1%)
  parent → leaf (refinement):  15 (5%)
  sibling-within-subtree:      14 (5%)
  cross-subtree:                5 (2%)

Net specificity move: +12 columns more specific in extend than parent
Confidence delta on unchanged: median +0.177, mean +0.196
```

### Specific Phase-3+validation refinements

The 15 parent-to-leaf flips include exactly the failure modes
documented in earlier xlsx reviews:

- `shipping_manifests/tracking_id`: `A_TRID` parent → `TRANSID` leaf
- `legal_cases/party_ref`: `C_PID` parent → `NAMEFULL` leaf
- `gaming_profiles/linked_account`: `ACCOUNT_ID` → `SOCIAL_ID`
- `insurance_claims/alt_contact`: `A_PHN` → `OTHPHNUM`
- `hr_compensation/comp_value`: `INCOME` → `SALARY`
- `shipping_manifests/col_32`: `COUNTRY` → `SHIPCNTY`

### Three column-classes that *still* miss

Of the 8 operator-curated columns 0146134f still misses, all fall
into pre-documented failure modes:

- **TRANSID over-application on permit columns** —
  `permit_ref`, `rec_33` wanting `TRAVPERM`/`WORKPERM`, still
  getting `TRANSID`
- **System-vs-Person URL** —
  `page_ref`, `media_ref` wanting `PRSNURL`/`INPPHOTO`, still
  getting `SYSURL`
- **Network identifier confusion** —
  `network_addr` wanting `DEVMACADDR`, still getting `IPADDR`

These are the targets for the recipe-driven dense-synth SVM
retraining workstream (parked pending implementation).

---

## Reproducibility checklist

For others to reproduce this work end-to-end:

1. Clone the Atelier repo at the commit landed during the
   2026-05-13 session (Phase 3 + validation+retry merged).
2. Configure a Hive connection
   pointing at a corpus that matches the shape (data tables in
   one database, annotations table in `default.annotations`,
   ~10-50 tables).
3. Identify a stable subset and an "added" subset of the corpus.
4. Follow Steps 1–6 above.
5. Compare:
   - `build/results/<parent_run>/evaluation_report.json` vs
     `build/results/<extend_run>/evaluation_report.json` for
     headline metrics
   - `build/results/<parent_run>/classifications.json` vs
     `build/results/<extend_run>/classifications.json` for
     column-level diffs on the overlap
   - `build/results/<parent_run>/validation_retries.json` for
     the LLM-hallucination audit trail
6. If you have an operator-curated review spreadsheet (per
   `docs/src/operations/embeddings-reviewer-guide.md`), apply
   the scoring methodology in this report.

The session's artifacts live at:

```
build/results/5450b626/   # pre-Phase-3 baseline
build/results/1d6e3fae/   # Phase 3 only
build/results/2ac4d0a6/   # Phase 3 + validation+retry classify
build/results/0146134f/   # Phase 3 + validation+retry extend
```

Spreadsheet:
`Atelier-Results-vs-Prompt-solution-522d89ae.xlsx`

Backfill script (used to populate `predicted_annotation` on
extend runs produced before the colocation fix landed):
`scripts/backfill_extend_annotations.py`

---

## What's not in scope for this report

- **Recipe-driven SVM retraining** to address the 8 remaining
  operator-curated misses (parked; needs synth-generator
  densification around documented confusion pairs)
- **Cautious-review threshold tuning** to align parent classify
  predictions more closely with extend (A/B candidate)
- **Multi-reviewer ground truth** to replace the single-operator
  spreadsheet as the evaluation substrate (Tier 0 of the broader
  accuracy-improvement roadmap)
- **Subjective Logic / conformal prediction** for the no-ground-truth
  deployment scenario (architectural discussion captured in
  separate design notes)

Each is tracked separately; the workflow documented here is the
current operationally-ready path.
