<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Pipeline Phases (FSM Walk-Through)

A run of the classification pipeline advances through a finite state
machine.  Each state is a named *phase* with a single responsibility,
and the legal transitions between phases — defined authoritatively in
[`src/atelier/classify/fsm.py`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/fsm.py)
— form the workflow that operators see live in the Workflows page and
that this document narrates end-to-end.

This page is the operator-facing companion to two deeper references:

- [Classification Pipeline](./classification.md) — *what* the pipeline
  does mathematically (DST, evidence sources, fusion strategies).
- [DST Evidence Independence](./dst-evidence-independence.md) — *why*
  each evidence source qualifies for Dempster's rule of combination.

Read this one first when you need to walk a reviewer through the run
shape: which phase produces which artifact, where the iteration loop
lives, what makes a run land in `CONVERGED` versus `ERROR`.

## At a glance

```text
                                     ┌──── revisit ────┐
                                     │                 │
                                     ▼                 │
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → LLM_SWEEP → VALIDATING ──┐
                                                                         │
                                                                         ▼
                                                       CLASSIFYING → FUSING → EVALUATING → CONVERGED
                                                                                                │
                                                                                          (any phase)
                                                                                                ▼
                                                                                              ERROR
```

The arrow back from VALIDATING to LLM_SWEEP is the iteration loop;
it's the heart of the algorithm and is described in
[Iteration loop](#iteration-loop-llm_sweep--validating-is-the-algorithm)
below.

## Phases in execution order

| # | State | What it does | Primary output |
|---|---|---|---|
| — | **`IDLE`** | No run in flight. Ready to dispatch the next classification. | — |
| 1 | **`LOADING_VOCAB`** | Load the user-supplied taxonomy (annotations CSV / Hive table / DB) and validate: label collisions, duplicate codes, orphaned aliases, parent-aware frame structure. | `HierarchicalCategorySet`, `FrameOfDiscernment` |
| 2 | **`DISCOVERING`** | Probe the data source via `cml.data_v1` (Hive), the meta-tagging mount (CSV), or the bundled fixtures to enumerate the tables in scope. | `list[str]` of table names |
| 3 | **`SAMPLING`** | For each discovered table, sample column metadata: bare names, types, ~5 representative values, true `COUNT(DISTINCT)` bounded by the sample limit, null ratio, sibling list. Reference-key columns (`attr_1_2_3_*` answer-key shape) are filtered out so they don't trivially leak into evaluation. | `list[ColumnSample]` (canonical bare names — see [`ColumnSample` invariant](#columnsample-canonical-form)) |
| 4 | **`LLM_SWEEP`** | Claude classifies each directly-targeted column into the user vocabulary. Iteration 1 sweeps every column (or the [Monte Carlo](./monte-carlo.md) sampled subset — a stratified slice for large corpora; the remaining columns get label propagation later). Iterations 2…N revisit only the columns flagged for re-look in the previous `VALIDATING` pass. | `state.labels[qualified_name] → category_code`, plus per-column LLM confidence |
| 5 | **`VALIDATING`** | ML re-validation: CatBoost (fit-to-LLM during the loop) and the synth-trained SVM (translated through the LLM-mediated ICE→user-vocab alignment) score the same columns independently of the LLM. Per-column DST mass with conflict K is computed under the parent-aware frame. The disagreement set — driven primarily by belief-gap `Pl − Bel`, with K and coverage as secondary signals — feeds the next iteration's revisit batch. The loop exits when convergence criteria are satisfied; otherwise it re-enters `LLM_SWEEP`. | `state.ml_prediction`, `state.ml_belief`, `state.ml_plausibility`, `state.ml_conflict`, the next iteration's disagreement list |
| 6 | **`CLASSIFYING`** | Final per-column DST evidence fusion. Up to six evidence sources combine: `name_match`, `pattern`, `cosine`, `llm`, `catboost`, `svm`. Each produces a mass function over the parent-aware frame; per-column predicted code, belief, plausibility, and conflict are computed here. | `classifications: list[dict]` (each entry shaped as `classifications.json` rows) |
| 7 | **`FUSING`** | Combine per-column mass functions via the configured fusion strategy. `dempster` normalizes conflict by `(1 − K)`; `yager` redirects conflict mass to Θ (ignorance). Cautious-code review (when enabled) runs *here* — backing off over-specified leaf predictions whose belief sits below the commit threshold to a parent code where it does. | Headline classification per column; `cautious_review.json` (when enabled) |
| 8 | **`EVALUATING`** | Compute corpus-level metrics: accuracy vs reference (when present), per-category precision/recall, K distribution, gap distribution, INOS residual count. Persist artifacts to disk and emit the parquet for the Embeddings page. Overwatch (when enabled) runs at the tail of this phase. | `evaluation_report.json`, `column_trajectories.json`, `taxonomy_findings.json`, `atelier_embeddings.parquet`, ML artifacts, `overwatch.md` (when enabled) |
| — | **`CONVERGED`** | Terminal success. Convergence criteria satisfied; results are on disk under `build/results/{run_id}/` and registered in the DB via the run-end registration path (or recovered later by [`atelier.db.sync.sync_filesystem_to_db`](https://github.com/zndx/atelier/blob/trunk/src/atelier/db/sync.py) on restart). | — |
| — | **`ERROR`** | Terminal failure. The FSM `error` field carries the diagnostic; pipeline logs and `register_error.json` (if any) carry the rest. | — |

The FSM defines two states that the standard inference run does not
visit — `GENERATING_SYNTH` and `TRAINING`.  These belong to the
offline [synth-corpus generation + SVM-training flow](./synth.md) that
produces the bundled SVM artifact (legacy filename
``svm_frontier.pkl`` retained on disk for backward compatibility with
older run directories), and are reachable from `SAMPLING` only on the
explicit synth-generate code path.

## Iteration loop: `LLM_SWEEP ⇄ VALIDATING` is the algorithm

The single most important thing to internalize when reviewing the
pipeline: **`LLM_SWEEP` and `VALIDATING` are not two separate one-shot
phases — they form an iteration loop, and the loop is the convergence
algorithm.**

Each cycle:

1. **`LLM_SWEEP`** labels (or re-labels) the directly-targeted
   column set on iteration 1, the disagreement set on iterations 2…N.
2. **`VALIDATING`** runs ML re-validation, computes per-column belief,
   plausibility, and conflict under the parent-aware DST frame, and
   identifies the next disagreement set.
3. The loop exits when one of the [convergence criteria](#convergence-criteria)
   is satisfied; otherwise it re-enters `LLM_SWEEP`.

The Workflows page draws this as a purple dashed back-edge from
`VALIDATING` to `LLM_SWEEP` precisely because the geometry teaches the
algorithm: this is *bootstrapping with active-learning revisit*, not a
linear pipeline.

The driver of the loop is configurable:

- **Programmatic (default):** `pipeline._llm_revisit` picks revisit
  candidates from `_identify_disagreements` and `_identify_uncertain_columns`.
- **Agent-driven (capability flag):** the [Agent Convergence](#optional-capability-skills)
  skill replaces the programmatic driver — Claude chooses revisit
  candidates and decides when to declare convergence via tool calls.

## Convergence criteria

A run reaches `CONVERGED` when *any* of the following holds at the end
of an iteration in the `LLM_SWEEP ⇄ VALIDATING` loop:

| Criterion | Config key | Default | Notes |
|---|---|---|---|
| **Mean belief-gap below threshold** | `classify.bootstrap.gap_threshold` | 0.05 | The **primary** signal — converging on `mean(Pl − Bel)`, not on K. Locked in by commit `bd7de2c` after the parent-aware DST frame audit. |
| **Coverage met + K acceptable** | `classify.bootstrap.coverage_floor`, `classify.bootstrap.k_threshold` | 0.95, 0.40 | Backstop — a corpus that LLM-labels everything cleanly on the first sweep doesn't need additional iterations. |
| **Iteration cap reached** | `classify.bootstrap.max_iterations` | 4 | Fallback. The convergence reason is recorded as `max_iterations_reached` so the UI can show an honest "ran the full budget" rather than claiming gap convergence the run didn't actually achieve. |
| **Min iterations honored** | `classify.bootstrap.min_iterations` | 2 | Forces at least N revisit cycles before any convergence path can fire. Defends against a single-pass LLM that happens to land cleanly without the ML cross-check having run. |

Conflict K (Dempster's rule's normalization mass) is **diagnostic, not
the gating signal.**  Earlier iterations of the design framed K as the
convergence headline; that framing was retired in commit `bd7de2c` and
matters for any review of older docs or telemetry that still leads
with K.

## `ColumnSample` canonical form

The pipeline's data-model invariant:
[`ColumnSample.name`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/sampler.py#L26)
is always the **bare** column identifier — table-relative, free of any
`f"{table_name}."` prefix.  Cross-table identity uses the
`qualified_name` property (`f"{table_name}.{name}"`) for dict keying.

This invariant is enforced in `__post_init__` and validated at every
source boundary:

- **Hive sampler** (`sampler._strip_table_qualifier`) — strips the
  `f"{table_name}."` qualifier that Hive's JDBC driver returns from
  `SELECT * FROM db.table`.
- **Meta-tagging CSV** (`meta_tagging_source`) — strips the same
  prefix from CSV headers that encode the table name.
- **Synth, OOTB sample, fixtures** — produce bare names by
  construction.

Any new source path that produces qualified names will trip the
`__post_init__` invariant at construction time with a clear
diagnostic, rather than letting them silently propagate into the
embedding text — where a repeated table-name prefix in the column
name *and* the sibling list would drown the actual column signal in
table-theme noise and produce table-wide misclassification.

## Optional capability skills

Three skills attach to specific phases and are gated by the
corresponding capability flag.  Each renders only when its flag is
enabled in `/api/status` config.

| Skill | Attaches to | Behavior | Capability flag |
|---|---|---|---|
| **Agent Convergence** | `VALIDATING` | Claude drives the convergence loop directly via the `agent_loop` tool surface — picks revisit candidates from belief/conflict signals and declares convergence when satisfied — replacing the programmatic loop driver. Bounded by `max_turns`. | `classify_agent_enabled` |
| **Cautious Review** | `FUSING` | Per-column LLM review that backs off over-specified leaf predictions to a parent code where belief crosses the commit threshold. Defends against false-precision claims on opaque or ambiguous columns. | `cautious_review_enabled` |
| **Overwatch** | `EVALUATING` | Single-turn Opus analysis writes `overwatch.md` with pipeline-tuning recommendations after the run lands. Requires direct Anthropic API (not Bedrock) — Bedrock lags Opus releases. | `overwatch_enabled` |

The three skills are visible as orange dashed nodes in the Workflows
page when their flags are enabled, attached to their host phases.
This is the registry-MVP shape — when we hit roughly six surfaces it
graduates to a backend `/api/skills` endpoint reading from a real
registry rather than a hand-coded list.

## Phase ↔ artifact map

For an end-to-end review of any single run, here's what's recoverable
from `build/results/{run_id}/`, indexed by the phase that produced it:

| Phase | Artifact | What's in it |
|---|---|---|
| Run start | `settings_snapshot.json` | The config that drove the run — `source_id`, all overlay values at start, default values, the resolved settings the pipeline actually used. |
| `LLM_SWEEP` ⇄ `VALIDATING` | `column_trajectories.json` | Per-column history across iterations: label changes, ML predictions, belief/plausibility/conflict trajectory, the `revisited` flag per iteration. |
| `LLM_SWEEP` ⇄ `VALIDATING` | `catboost_fit_to_llm.cbm` + `.classes.json` | CatBoost fit to the in-loop LLM labels. Persisted for [Extend runs](./ml-artifacts.md). |
| `LLM_SWEEP` ⇄ `VALIDATING` | `svm_frontier.pkl` + `.classes.json` | Synth-trained SVM with the in-run LLM-mediated alignment. Persisted for Extend runs. (Filename retained for backward compatibility; underlying model is the synth-trained SVM, not the excised M9 in-loop retrain.) |
| `CLASSIFYING` + `FUSING` | `classifications.json` | The per-column output: predicted code, belief, plausibility, conflict, full evidence-source mass distributions, belief path, llm/ML/cautious codes. The headline corpus result. |
| `FUSING` | `cautious_review.json` | Cautious Review skill audit (only when enabled). |
| `EVALUATING` | `evaluation_report.json` | Corpus-level metrics: accuracy, per-category precision/recall, K distribution, gap distribution, INOS residual count. |
| `EVALUATING` | `taxonomy_findings.json` | Notes flagged during taxonomy traversal — orphaned codes, suspicious aliases, near-duplicate labels. |
| `EVALUATING` | `atelier_embeddings.parquet` + `umap.pkl` | Input for the Embeddings page (UMAP projection of the per-column embedding vectors with predicted-code colorings). |
| `EVALUATING` | `sage_importance.json` + `shap_summary.json` | [GPU-accelerated](./gpu-acceleration.md) global feature importance + per-column SHAP attributions (when enabled). |
| `EVALUATING` (post) | `overwatch.md` | Overwatch skill output (only when enabled). |
| Run end | `register_error.json` (rename to `.resolved` after sync) | If DB registration failed mid-run, the sync path on restart picks the run up from this sidecar. |

## Where to look in code

| Concern | File |
|---|---|
| FSM state enum, transitions, `FSMRun` dataclass | `src/atelier/classify/fsm.py` |
| Phase advancement (every `fsm.advance(...)` call) | `src/atelier/classify/pipeline.py` |
| Iteration loop driver (programmatic) | `src/atelier/classify/bootstrap.py` (`_llm_sweep`, `_llm_revisit`, `_identify_disagreements`, `_run_ml_validation`) |
| Iteration loop driver (agent) | `src/atelier/classify/agent_loop.py` |
| Per-column DST fusion | `src/atelier/classify/pipeline.py` (`_classify_column`) |
| Convergence criteria evaluation | `src/atelier/classify/bootstrap.py` (`_mean_gap`, `_mean_k`, `_coverage`, `should_stop_early`) |
| Cautious Review skill | `src/atelier/classify/cautious_review.py` |
| Overwatch skill | `src/atelier/overwatch/agent.py` |
| Workflows page topology (UI) | `ui/src/lib/fsmPipelineLayout.ts` |

## State transition reference

Authoritative state-transition table (from
[`fsm.py:_TRANSITIONS`](https://github.com/zndx/atelier/blob/trunk/src/atelier/classify/fsm.py#L45)):

| From | Legal next states |
|---|---|
| `IDLE` | `LOADING_VOCAB` |
| `LOADING_VOCAB` | `DISCOVERING`, `ERROR` |
| `DISCOVERING` | `SAMPLING`, `ERROR` |
| `SAMPLING` | `CLASSIFYING`, `GENERATING_SYNTH`, `LLM_SWEEP`, `ERROR` |
| `GENERATING_SYNTH` | `TRAINING`, `ERROR` |
| `TRAINING` | `CLASSIFYING`, `ERROR` |
| `LLM_SWEEP` | `VALIDATING`, `CLASSIFYING`, `ERROR` |
| `VALIDATING` | `LLM_SWEEP`, `CLASSIFYING`, `ERROR` |
| `CLASSIFYING` | `FUSING`, `ERROR` |
| `FUSING` | `EVALUATING`, `ERROR` |
| `EVALUATING` | `CONVERGED`, `IDLE`, `ERROR` |
| `CONVERGED` | `IDLE` |
| `ERROR` | `IDLE` |

`SAMPLING → CLASSIFYING` (skipping `LLM_SWEEP`) is the path used by
[Extend runs](./ml-artifacts.md), where ML-only inference is desired
because the LLM has already classified an earlier corpus and the
artifacts are being applied to a new dataset.  `LLM_SWEEP →
CLASSIFYING` (skipping `VALIDATING`) is the "first-sweep convergence"
path on small corpora that don't need iteration.
