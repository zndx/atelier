<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Phase Gate Report #2 — iterative revisit, reasoning capture, and honest uncertainty

**Release:** `v0.2.0-phase-gate-2` · commit [`eb2b49a`](https://github.com/zndx/atelier/commit/eb2b49a) · 2026-04-19
**Audience:** UAT and Product Engineering partners
**Companion:** Phase Gate #1 — `docs/notes/2026-04-17/170000_phase_gate_validation.md`

---

## 1. Context

Phase Gate #1 demonstrated that our DST-fused pipeline did-no-harm against a
synthetic test corpus with known per-column answer keys. Phase Gate #2 moves
past parity to investigate three questions the first report deliberately left
open:

1. **Does iterative revisit improve accuracy?** If we train a small ML model
   on the LLM's first-pass opinions and feed that model's competing
   suggestions back to the LLM on a second pass, does accuracy move?
2. **Can the LLM signal honest uncertainty** instead of silently abstaining
   on columns it cannot commit on?
3. **What research artifact is worth producing alongside the predictions?**
   If the LLM emits a thinking trace when classifying, is that trace
   substantive enough to be a primary output of the pipeline?

The answers, in one sentence each: yes (via an abstention-rescue mechanism
that resolves previously-unlabeled columns into committed predictions); yes
(the LLM uses an explicit `UNCERTAIN` token when prompted to do so); and yes
(hundreds of thousands of tokens of coherent per-column rationales are
available at the same cost as the classifications themselves).

This report presents **exactly what works today**, the reproducible sequence
for classifying one Hive table from discovery through parquet write, and the
limitations we own up front.

## 2. Terminology — what "reference" means where

Four distinct kinds of per-column labels appear in our writeups. We name each
explicitly to avoid the conflation our prior reports had let slip in.

| Term | Source | Authority | Where it lives |
|---|---|---|---|
| **Published ground truth** | External, human-curated benchmarks (e.g., SOTAB, GitTables) | Gold standard; memorization-safe check on an LLM's one-shot classification ability | SOTAB pilot artifacts |
| **Curated reference** | Generator-derived (synth pairs each natural-named column with an answer-key "reference column"); spot-checked by hand | Definitive **for the synthetic corpus**; not equivalent to a published GT | `build/meta-tagging-clean/curated_reference.csv` (246 rows) |
| **LLM commitment** | A single LLM's pass-1 or pass-2 output | Classifier opinion; not a truth | parquet `llm_code`, `predicted_code` |
| **CatBoost prior** | CatBoost fit to LLM labels, used for revisit enrichment | *Not independent evidence* — it is a compressed self-consensus of the LLM; valuable **specifically** for rescuing abstentions | parquet `predicted_code` when fused via DST |

**Ablation** — referenced below — means a controlled experiment that holds
most of the pipeline fixed and varies exactly one component at a time, so
changes in accuracy can be attributed to that component rather than to the
combination.

## 3. Pipeline — what happens to a Hive table

The reproducible sequence, one table at a time, with no branches.

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Hive table    │ →  │  Sampling loader │ →  │  Feature extract │
│ (via cml.data_v1│    │  (50 rows/col)   │    │  (12 features)   │
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
┌──────────────────┐    ┌───────────────────┐    ┌─────────────────┐
│  Parquet write   │ ←  │   DST fusion      │ ←  │  Pass-1 LLM     │
│  + atlas layout  │    │   (5 sources)     │    │  classify batch │
└──────────────────┘    └───────────────────┘    └─────────────────┘
         ▲                       ▲
         │                       │
         │              ┌────────┴─────────┐
         │              │ Abstention path: │
         │              │ CatBoost-on-LLM  │
         │              │ → enriched       │
         │              │ pass-2 prompt    │
         │              └──────────────────┘
         │
     …with per-row SHAP, per-batch reasoning traces, and curated-reference join
```

Step by step, for one table:

1. **Discovery.** Operator (or the `_discover_and_register_hive_sources`
   startup seeder) points the gateway at a CAI Data Connection and a Hive
   database. The table is registered as a `data_source` row.

2. **Sampling.** `atelier.classify.real_data_loader.load_real_samples` pulls
   50 rows per column, stripping the `<table_name>.` CSV-header prefix. By
   invariant (`_REFERENCE_COL_RE`), any *reference column* whose name
   encodes a code in its suffix — `attr_1_2_3_4`, `code_7_1` — is dropped
   from the sample set and from siblings. Reference columns are answer
   keys, not inputs.

3. **Feature extraction.** For each natural-named column,
   `atelier.classify.features.extract_features` produces a 12-feature
   `ColumnFeatures` object: column name (humanized), type, sample values
   (5), cardinality, null ratio, value entropy, pattern signals,
   avg value length, numeric ratio, sibling context (first 5 other columns
   in same table), source table, value description.

4. **Pass-1 LLM classification.** Columns are batched (25 at a time) and
   sent to GLM-4.7 via Cerebras. The model emits JSON-structured
   classifications and a separate *thinking trace* (via Cerebras's
   `reasoning_format="parsed"`). Both the labels and the traces are
   retained.

5. **DST evidence fusion.** Five evidence sources contribute per-column
   mass functions: `name_match`, `pattern`, `cosine` (similarity against
   category centroid embeddings), `llm` (the pass-1 output), `catboost`,
   and `svm` (ML classifiers). Dempster's rule produces a fused belief
   assignment per column with belief, plausibility, conflict, and evidence
   trail.

6. **Abstention handling.** Columns the LLM does **not** commit on in
   pass-1 (JSON omits the code, or the model output is malformed) carry an
   empty `llm_code`. CatBoost — which was fit to the LLM's committed
   pass-1 labels — still has a top-3 prediction for these rows because it
   was embedded-text-based and extrapolates to unseen feature territory.

7. **Pass-2 revisit.** A second LLM pass sends all columns back with an
   enriched prompt that includes: the pass-1 label, the CatBoost top-3
   competitor labels with probabilities, the per-column SHAP top features
   (via `run_embedding_shap` on the 12 conceptual features), k-nearest-
   neighbor column labels by embedding cosine, and a *prescriptive*
   instruction set that (a) requires consultation of the new evidence and
   (b) allows the literal token `UNCERTAIN` as an explicit response.

8. **Parquet write.** `_write_parquet` emits `atelier_embeddings.parquet`
   with per-row: predicted code / label / annotation mnemonic; belief /
   plausibility / uncertainty / conflict; evidence trail; SHAP top-3;
   *reference_code* and *reference_label* joined from the curated
   reference (if available); `matches_reference` (boolean, or null when
   no reference exists); atlas-ready `text` field formatted
   `f"{Ontology} - {Annotation}"` (e.g. `"Bank Account - BAN"`).

9. **Reasoning artifact.** Alongside the parquet,
   `reasoning_traces.jsonl` is written: one record per LLM batch with the
   thinking text, the list of column IDs it covers, pass number
   (`pass1` or `pass2_revisit`), and token counts.

## 4. What's working — three named mechanisms

### 4.1 GLM-4.7 reasoning capture

Cerebras's GLM-4.7 backend emits the thinking trace in a field separate
from the final answer (`choices[0].message.reasoning` in
`reasoning_format="parsed"` mode). Earlier runs counted the tokens but
discarded the text. The pipeline now persists the full text per batch.

Observed volume on the SOTAB pilot (20 classes × 20 columns = 400 cols,
run `run_20260419_165954`):

- 16 pass-1 batches: **1,400,619 chars**, 390,017 reasoning tokens
- 16 pass-2 batches: **839,446 chars**, 252,242 reasoning tokens
- Total: **~2.2M chars / ~642K reasoning tokens** on a 400-column run

The per-batch records are coherent per-column rationales that reference
value patterns, category distinctions, and — on pass-2 — the injected
CatBoost / SHAP / UNCERTAIN signals. At GitTables scale (≈ 1.7M columns)
this methodology would produce millions of structured reasoning records
as a byproduct of classification itself.

### 4.2 CatBoost abstention rescue

This is the clean-signal mechanism: on columns where the LLM refused to
commit on pass-1, CatBoost's extrapolation gives the LLM something to
push against on pass-2.

Run `run_20260419_165954`, 400 columns:

| Pass-1 status | n | Pass-2 outcome |
|---|---|---|
| Committed (275) | | |
| → improvements (p1 wrong → p2 right) | 2 | |
| → regressions (p1 right → p2 wrong) | 6 | |
| → changed still wrong | 1 | |
| → unchanged | 236 | |
| → honest UNCERTAIN on pass-2 | 4 | retraction, see §4.3 |
| → unparsed pass-2 | 26 | |
| Abstained (125) | | |
| → **resolved correct on pass-2** | **31** | |
| → resolved wrong on pass-2 | 68 | see limitations |
| → honest UNCERTAIN on pass-2 | 1 | |
| → still unresolved | 25 | |

Net pass-1 → pass-2 fidelity shift on this run: **62.25% → 68.75%**
(Δ +6.50 pts against the published SOTAB GT). A companion run on the
same seed earlier the same day (`run_20260419_091643`) showed a larger
shift (70.00% → 79.00%, Δ +9.00 pts) — a reminder that GLM-4.7's
inference is non-deterministic in this regime and the magnitude of the
effect varies between runs. The *direction* (positive) and the
*mechanism* (abstention rescue dominates the gain) are stable.

### 4.3 Prescriptive prompt + honest `UNCERTAIN` affordance

The pass-2 system prompt is explicitly rules-first per the Cerebras
GLM-4.7 migration guide ("STRICT RULES, MUST, REQUIRED, explicit
directives"). It authorizes the literal token `UNCERTAIN` as a category
response for columns where the evidence is genuinely ambiguous — so the
LLM can signal honest abstention instead of emitting silent blank
output.

On the run above: 5 honest UNCERTAIN responses recorded (1 from the
abstention pool, 4 *retractions* from the labeled pool — columns the
LLM committed on pass-1 and then stepped back from on pass-2 when it
saw the new evidence). This is a small number in absolute terms but
directionally important: the LLM is willing to *retract* a commit when
evidence invites reconsideration, rather than doubling down.

### 4.4 Reasoning-trace citation analysis

A light-touch regex count over pass-2 reasoning traces (see
`scripts/attribution/analyze_reasoning_traces.py`) found that every
injected evidence source is cited in essentially every pass-2 batch:

| Term | Pass-1 citation rate | Pass-2 citation rate |
|---|---|---|
| CatBoost / ML-prior language | 0% | **100%** |
| SHAP / attribution language | 6% | **100%** |
| `UNCERTAIN` token | 0% | **100%** |
| "revised / reconsider" language | 19% | **100%** |
| Prescriptive `STRICT` / `MUST` | 12% | 94% |
| Nearest-neighbor labels | 0% | 44% |

Interpretation (conservative): the pass-2 prompt content is being
consulted by the LLM. The analysis is a surface-level regex count, not
a semantic parse — missing matches may reflect phrasings we didn't
anticipate. Use as directional evidence, not as a proof of causation.

## 5. Known limitations — owned up front

### 5.1 CatBoost is a compressed self-consensus of the LLM, not an independent oracle

`CatBoost-fit-to-LLM` by definition agrees with the LLM on labeled rows.
Its value is specifically in generalizing to *abstention* rows — feature
territory the LLM refused to opine on — where it gives the LLM a
non-null prior. For the labeled-row revisit path, the mechanism
provides less new signal than it appears to: an ablation (holding
CatBoost out, keeping only the UNCERTAIN affordance and prescriptive
language) would isolate how much of the labeled-row delta comes from
each source. That ablation is **not part of this release**; it is named
as the next step.

This limitation is also our entry point to ongoing work on **de novo
Data Element prediction** — producing independent, LLM-free priors
over column semantics by characterizing value patterns + column
morphology + cross-table semantic grouping at GitTables scale. A
prior source that is genuinely independent of any LLM's
self-consensus would turn iterative revisit from a "self-consistency
re-prompt" into an actual fusion of two disjoint evidence streams.

### 5.2 Pass-1 → pass-2 fidelity gain is the product of three simultaneous changes

The measurement cascade that produces the ~+6 to +9 point pass-2
improvement combines three things enabled together:

1. CatBoost top-3 injected as revisit context.
2. The explicit `UNCERTAIN` affordance.
3. Prescriptive prompt language.

We have not yet isolated which of the three carries what share of the
improvement. The regex analysis above gives directional evidence that
all three are consulted, but consulted ≠ causally attributable. Next
planned measurement: each component held out in turn.

### 5.3 Inference non-determinism

Two runs of the same pilot with the same seed on the same day produced
different pass-1 fidelities (62.25% vs 70.00%) and abstention counts
(125 vs 54). This is a property of the Cerebras GLM-4.7 inference
endpoint, not of our seed handling. Reported numbers are the specific
release-coincident run; magnitude varies, the direction of
pass-1 → pass-2 improvement is consistent across observed variance.

### 5.4 Curated reference is not a ground truth

The `curated_reference.csv` in the UAT hand-off bundle is generator-
deterministic (193 of 246 rows come directly from the synth generator's
reference-column encoding). It is authoritative **for this synthetic
corpus**, and we use it as the accuracy ruler for pipeline behavior in
a controlled setting. That setting is a pedagogical stop, not the
destination: the real-world problem is bootstrapping classifications
when *no* external answer key exists, and the mechanisms in this
release are the ones designed to work there.

## 6. Artifacts

| File | Description |
|---|---|
| `build/results/<run_id>/atelier_embeddings.parquet` | Per-column predictions with DST belief intervals, SHAP top-3, reference_code, reference_label, matches_reference, atlas-ready `text` field |
| `build/results/<run_id>/evaluation_report.json` | Accuracy + F1 + per-category breakdown |
| `build/results/<run_id>/classifications.json` | Long-form per-column records with hierarchical belief paths |
| `build/results/<run_id>/reasoning_traces.jsonl` | Per-batch GLM-4.7 thinking traces (pass-1 and pass-2) |
| `build/results/<run_id>/shap_summary.json` | Per-feature SHAP contributions |
| `build/results/<run_id>/catboost_fit_to_llm.cbm` · `svm_frontier.pkl` | Trained ML models — persisted for reproducibility |
| `build/meta-tagging-clean/curated_reference.csv` | The per-column reference labels for the synthetic corpus |
| `build/meta-tagging-clean/curated_reference_summary.json` | Derivation counts, class distribution, exclusion audit |
| `build/meta-tagging-clean.zip` | UAT-ready hand-off: cleaned tables + annotations.csv + curated_reference.csv + README |

Release tag: **`v0.2.0-phase-gate-2`** · head commit: **`eb2b49a`**
GitHub: `https://github.com/zndx/atelier/releases/tag/v0.2.0-phase-gate-2`

### How to reproduce a pilot run

```bash
# 1. Generate the curated reference from the snapshot.
uv run python scripts/parity/build_curated_reference.py

# 2. Run the SOTAB pilot (400 cols, 20 classes × 20 each).
uv run python scripts/attribution/run_sotab_pilot.py \
    --n-classes 20 --per-class 20 --batch-size 25

# 3. Run the pass-2 revisit against the pilot output.
uv run python scripts/attribution/run_sotab_revisit.py <pilot_run_id>

# 4. Count citation rates in the reasoning traces.
uv run python scripts/attribution/analyze_reasoning_traces.py <pilot_run_id>
```

## 7. Forward plan

Three threads, scoped briefly:

1. **Component-ablation study.** Hold out each of {CatBoost top-3
   injection, UNCERTAIN affordance, prescriptive language} individually
   and measure the isolated effect of each on pass-2 accuracy. This is
   the measurement the current release stops just short of; it is the
   next deliverable.

2. **De novo Data Element prediction** as an LLM-independent prior
   source. Trained on patterns in column-value distributions and
   semantic-group structure across wide tables, it would replace the
   current regex library and elevate the sibling-context feature from
   "first few columns in the same table" to "columns in genuine
   semantic co-occurrence at corpus scale." When mature, it becomes
   the independent evidence source DST fusion has been missing.

3. **Reasoning-trace corpus** as a downstream research artifact. A
   400-column pilot produced ~2.2M characters of structured per-column
   reasoning at no incremental inference cost. Scaling the methodology
   to GitTables or similar benchmarks produces a corpus suitable for
   reasoning-distillation into smaller classifiers — training signal
   that existing label-only benchmarks lack.

---

## Appendix A — Parquet schema (atelier_embeddings.parquet)

Columns emitted at pipeline write time (as of `v0.2.0-phase-gate-2`):

| Column | Type | Description |
|---|---|---|
| `text` | string | Atlas tooltip display; `"{Ontology} - {Annotation}"` (e.g. `"Bank Account - BAN"`) |
| `x`, `y` | float | 2D projection coordinates for atlas scatter |
| `table_name` | string | Source table |
| `column_name` | string | Column identifier within the table |
| `column_type` | string | Inferred type |
| `predicted_code` | string | Pipeline's final label code |
| `predicted_label` | string | Ontology label |
| `predicted_annotation` | string | Annotation mnemonic |
| `llm_code` | string | LLM's pass-1 commitment (or empty on abstention) |
| `llm_confidence` | float | |
| `confidence`, `belief`, `plausibility`, `uncertainty`, `conflict` | float | DST interval summary |
| `needs_clarification` | bool | Flagged by DST when conflict-over-belief exceeds threshold |
| `evidence` | string | Human-readable fusion trail |
| `reference_code`, `reference_label` | string | From `curated_reference.csv` |
| `matches_reference` | bool / null | Whether the pipeline's `predicted_code` matches the reference |
| `embedding_text` | string | The 12-feature text fed to the embedder |
| `pattern_signals` | string | Comma-joined matched patterns |
| `dst_belief_path` | JSON string | Belief/plausibility trace from leaf to root |
| `cautious_code` | string | Deepest code where Bel > 0.7 |
| `shap_top{1,2,3}_name`, `shap_top{1,2,3}_value` | string, float | Top-3 SHAP attributions from the embedding classifier |
