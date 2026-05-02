<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# CAI LLM-coverage regression — three fixes for UAT reproducibility

## What the UAT team observed

CAI run `c164f3df` against the meta-tagging corpus (Bedrock Claude backend):

- State: CONVERGED
- Tables: 10, classified columns: **459**, LLM-labeled: 459
- **LLM Coverage: 75.0%** (≈ 115 of 459 columns got no LLM vote)
- LLM Calls: **29**
- Accuracy: 75.4%

The shipped parquet (`build/meta-tagging-clean/atelier_predictions.parquet`
from run `323cfbbc`) reports 263 classifiable columns and 94.56% exact
against the curated reference. The two do not line up, and the gap is
not a metrics difference — it's the CAI environment producing a
materially different tagging output.

## Root cause — three compounding problems

### 1. Reference columns leak into the CAI sample set (459 vs 263)

The meta-tagging loader (`load_meta_tagging_source`) excludes
synth-generator answer-key columns via the `_REFERENCE_COL_RE` regex.
The Hive sampler (`sample_table_metadata`) that CAI uses does **not**
apply that filter, so the 213 paired `attr_*` / `code_*` reference
columns end up in the LLM sweep. 459 = 246 natural-named + 213
reference twins — exactly the observed total.

These columns are trivially classifiable by name parse (the name
literally encodes the answer), so leaving them in inflates the
classified count with rows that carry no signal about classifier
quality. Worse, they show up as siblings of their paired natural-named
neighbor, leaking the code into that neighbor's embedding.

### 2. Bedrock silently caps `maxTokens` per model; pipeline didn't know

Pipeline config sets `classify_llm_max_tokens = 65536`. Bedrock enforces
each model's native output ceiling with **no warning in the response**:

| Model family | Bedrock output ceiling |
|---|---|
| `claude-3-5-sonnet`, `claude-3-5-haiku` | 8192 |
| `claude-3-*` (sonnet, opus, haiku) | 4096 |
| `claude-3-7-sonnet` | 64000 |
| `claude-sonnet-4`, `claude-haiku-4` | 64000 |
| `claude-opus-4` | 32000 |

A request for 65536 against `claude-3-5-sonnet` gets clamped to 8192
without `stopReason` being set to `"max_tokens"`. The response returns
a clean JSON with only some of the requested columns classified, and
the pipeline's existing halving-retry never fires (halving only engages
on `finish_reason in ("length", "max_tokens")`).

With a 25-column batch averaging ~200 output tokens per classification
plus reasoning overhead, 8192 tokens is a ceiling of roughly 16
classifications per batch. Every 25-column batch lost ~9 columns to
silent truncation (~36% per call). Over 29 calls this works out to
the observed ~25% aggregate coverage gap.

### 3. No post-sweep coverage check

The pipeline tracked `LLM Coverage = labeled / total` but never used
it as a retry signal. Missing columns silently fell through to the
ML-only classification path with much lower per-column accuracy,
explaining the 75.4% overall.

## Fixes

### Fix 1 — Universal reference-column exclusion (`pipeline.py`)

Added `exclude_reference_columns()` helper in
`src/atelier/classify/meta_tagging_source.py` that takes a list of
`TableSample`, drops any column whose name matches `_REFERENCE_COL_RE`,
and strips reference names from the remaining columns' sibling lists.

Called in `run_classification_pipeline()` unconditionally after sample
load, regardless of source (Hive, meta-tagging, fixture, synth). The
regex doesn't match production column naming conventions, so this is a
no-op on customer data. Verified against fixture data (50 cols → 50
cols, unchanged) and against a UAT-shape `first_name` +
`attr_1_1_1_9_2_1` pair (drops the answer-key column and cleans
siblings).

### Fix 2 — Bedrock per-model `maxTokens` cap (`llm_backend.py`)

Added `bedrock_max_output_tokens(model_id: str) -> int` that maps each
known Bedrock model family to its actual ceiling (table above). Both
`BedrockBackend.classify_batch()` and
`BedrockStructuredBackend.classify_batch()` now clamp `inferenceConfig.
maxTokens` (or `max_tokens` in the Anthropic Messages body) to the
smaller of the configured value and the model ceiling.

A new `_BedrockMixin.effective_max_tokens()` method exposes the
clamped ceiling so the bootstrap batch sizer
(`_estimate_safe_batch_size`) can scale the initial batch size
appropriately instead of relying on halving round-trips.

### Fix 3 — Partial-response detection + coverage-gap retry (`bootstrap.py`)

`LLMResponse` gained a `partial: bool` field. Every backend now
compares returned `column_name` set against requested `expected_names`
and sets `partial=True` when the backend reported a clean stop but
classifications are missing. The `truncated` property returns True
when either `finish_reason in ("length", "max_tokens")` *or* `partial`,
so halving retry engages on silent drops as well as explicit
truncation.

`_llm_sweep()` now runs a **targeted coverage-gap retry** after the
initial sweep: any column still missing from `state.labels` is retried
at `cfg.min_columns_per_call` batch size, grouped by table so sibling
context is preserved. Columns that still fail after the targeted retry
land in `state.failed_columns` with a logged warning rather than
silently propagating through the pipeline.

## Expected CAI behavior after the fixes

Running the same Bedrock-backed pipeline against the same UAT corpus:

- **Column count** drops from 459 to 246 (curated-reference-resolvable)
  — reference answer keys excluded.
- **LLM Coverage** goes to 100% barring a Bedrock outage or a
  pathological name the model keeps refusing.
- **Initial batch size** is right-sized: ~16 cols/call for
  `claude-3-5-sonnet` (8192 ceiling), ~25 for `claude-sonnet-4`
  (64000 ceiling). Halving retry still covers edge cases.
- **Accuracy** should land at 94.56% ±1 point against
  `curated_reference.csv` (seed/temperature variation; Bedrock Claude
  is highly consistent on this corpus).

## Reproducibility checklist for the UAT review

1. `curated_reference.csv` + `atelier_predictions.parquet` ship in the
   bundle.
2. CAI run against the same 8 tables produces a parquet with the same
   263 resolvable rows after reference-column exclusion (or 246 using
   curated-reference scope).
3. Scoring the CAI parquet with
   `uv run python scripts/parity/rescore_parquet.py <run_id>` against
   the bundle's `curated_reference.csv` should reproduce the 94.56% /
   97.49% numbers reported in `reconciliation.md`.
4. If the numbers diverge, the first diagnostic is
   `build/results/<run_id>/classifications.json` — check for non-empty
   `failed_columns` (coverage-gap survivors) and compare LLM-labeled
   vs predicted-code counts.

## Files touched

- `src/atelier/classify/meta_tagging_source.py` — `exclude_reference_columns` helper
- `src/atelier/classify/pipeline.py` — call the helper after sample load;
  fall back to `cfg.classify_sample_size` / `cfg.classify_tables_limit`
  when the caller passes None
- `src/atelier/classify/sampler.py` — `discover_tables` and
  `sample_table_metadata` honor HOCON values when callers don't pass
  them explicitly, logging a warning when a discover limit truncates
  actual tables
- `src/atelier/classify/llm_backend.py` — `_BedrockMixin`,
  `bedrock_max_output_tokens`, partial detection across 4 backends,
  `LLMResponse.partial` field
- `src/atelier/classify/bootstrap.py` — `_estimate_safe_batch_size`
  respects `backend.effective_max_tokens`; `_llm_sweep` runs targeted
  coverage-gap retry

## Related coverage-config bug fixed alongside

While auditing for 100%-coverage respect, found two more silent
overrides: `run_classification_pipeline`'s `tables_limit: int = 100`
and `sample_size: int = 50` were function-level defaults that
dominated the configured HOCON values (`classify.tables_limit`,
`classify.sample_size`) — gateway/service callers never passed these,
so the HOCON values were effectively dead config.

Fixed by making both parameters default to None and reading from cfg
at the top of the function. The `sample_table_metadata` /
`discover_tables` sampler functions were wired similarly so they too
use the configured values when callers pass None. On the meta-tagging
corpus (10 tables) this changes nothing observable; in a larger Hive
database this would have silently capped discovery at 100 tables.

## Double-checked coverage invariants

- **Monte Carlo passthrough** (`monte_carlo.py:300`): when
  `mc_sample_fraction >= 1.0` OR corpus size < `min_corpus_size`, the
  MC plan returns `frontier_columns = all_names` and
  `is_passthrough = True`. The LLM sweep then iterates over all
  `column_names` unconditionally.
- **Category ceiling** (`bootstrap.py:_estimate_safe_batch_size`):
  adapts *batch size*, not column coverage. No column is ever
  excluded by the sizer.
- **max_total_llm_calls** (default 5000): a budget cap, not a coverage
  cap. On the 459-column CAI corpus even a 50-col-per-batch sweep only
  consumes ~10 calls, well under budget.
- **`state.failed_columns`**: any column that halving-retries down to
  a per-column call and still errors is now surfaced in the audit,
  not silently dropped.
