# ML Artifact Management + Extend Classification

Each Atelier classify run trains a CatBoost classifier, optionally a
SVM frontier classifier, and (when umap-learn handles the projection)
a fitted UMAP reducer.  The **ML Artifact Set** feature makes those
trained models first-class entities — registered in PG, listed in the
UI, and replayable on new data through a streamlined **Extend
Classification** pipeline that skips the LLM sweep, DST iteration, and
agent loop.

## Why

The classify pipeline costs tens of minutes and (on Bedrock / Anthropic
direct) tens of dollars per run.  When the governance team adds new
tables to an existing Hive database, or stands up a new Hive / Impala
database with the same taxonomy, re-running the full pipeline is the
wrong tool — there's no new ground truth to learn from, and the LLM
sweep adds nothing the trained CatBoost can't reproduce at >100×
speed.  Extend Classification is the right shape: load the trained
artifacts, predict on the new columns, write a parquet, register a
new dataset.  Done.

The data model deliberately tracks lineage in OpenLineage terminology
(Run → Job → Dataset → Facet) so Marquez or a similar lineage backend
can be wired in later without remodeling.  The pathspec scheme (run
id-keyed artifact directories) borrows from Metaflow's DataStore
addressing — every artifact resolves to
`build/results/{run_id}/{filename}`.

## Concepts

| Term | What it is | Where it lives |
|---|---|---|
| **Data Source** | A configured source (Hive DB, Impala DB, OOTB Sample, filesystem mount). | `data_sources` table |
| **Dataset** | One classify or extend run's output parquet, versioned per source. | `datasets` table |
| **FSM Run** | One pipeline invocation (classify or extend). | `fsm_runs` table |
| **ML Artifact Set** | The bundle a classify run produced: CatBoost (.cbm + .classes.json), optional SVM (.pkl + .classes.json), optional UMAP (.pkl), plus vocab signature and embedding-model identity. | `ml_artifact_sets` table |
| **Active Artifact Set** | The single ArtifactSet a future Extend run will use. | `ml_artifact_sets.is_active` (partial unique index enforces only-one-active) |
| **Classify Run** | The full LLM + DST + agent pipeline.  Produces a Dataset AND an ArtifactSet. | `run_kind = 'classify'` on the dataset row |
| **Extend Run** | The streamlined ML-only pipeline.  Consumes an ArtifactSet, produces a Dataset only. | `run_kind = 'extend'` |

## Database schema

The migration `20260427000000_ml_artifact_sets.sql` adds
`ml_artifact_sets` and three lineage columns on `datasets`:

```
ml_artifact_sets:
  id, source_id (→ data_sources.id), fsm_run_id (→ fsm_runs.id),
  parent_artifact_set_id (self-FK),
  catboost_path, catboost_classes_path,
  svm_path?, svm_classes_path?, umap_path?,
  classes (JSON), feature_groups (JSON),
  vocab_signature (sha256(sorted(classes))),
  embedding_model, embedding_dim,
  display_name, summary, is_active, is_archived,
  facets (JSON, OpenLineage projection),
  created_at

datasets (added):
  artifact_set_id (→ ml_artifact_sets.id),
  parent_dataset_id (→ datasets.id),  -- extend lineage
  run_kind ('classify' | 'extend')
```

The **partial unique index**
`idx_ml_artifact_sets_one_active ON (is_active) WHERE is_active = TRUE`
is the Postgres-side guarantee that only one row may be active globally
at any time.  The DAO's
[`set_active_artifact_set`](../../src/atelier/db/dao.py)
runs the demote + promote in a single transaction so the index
constraint never sees two TRUE rows.

## On-disk layout

Each classify run writes to `build/results/{run_id}/`:

```
build/results/{run_id}/
  catboost_fit_to_llm.cbm                  # required
  catboost_fit_to_llm.classes.json         # required (classes + feature_groups sidecar)
  svm_frontier.pkl                         # optional (skipped if fit-to-LLM didn't fire)
  svm_frontier.classes.json                # optional
  umap.pkl                                 # optional (only when CPU umap-learn was used)
  atelier_embeddings.parquet               # the dataset
  classifications.json                     # full per-column results
  evaluation_report.json                   # accuracy stats
  settings_snapshot.json                   # config-at-start
  taxonomy_findings.json                   # vocab QA
  ...
```

`atelier.classify.artifact_set` is the single point of knowledge about
this layout — it builds the artifact-set record from a run dir and
loads the bundle for an Extend run.

## Pipeline writes (classify side)

At the end of EVALUATING, after the dataset row is upserted:

```python
# pipeline.py (paraphrased)
parquet_path = _write_parquet(...)        # also persists umap.pkl
                                          # alongside via joblib
dao.upsert_dataset(
    ..., artifact_set_id=run_id, run_kind='classify',
    parent_dataset_id=None,
)

spec = build_artifact_set_record(
    run_id=run_id, results_dir=results_dir, cfg=cfg,
    n_columns=len(classifications),
    source_id=source_id, fsm_run_id=run_id,
)
if spec is not None:
    dao.register_artifact_set(**spec)
```

The first registered artifact set on a fresh deploy auto-activates
(idempotent — subsequent registrations don't steal active).
Registration failures are non-fatal: the dataset still ships.

## Extend pipeline

`atelier.classify.extend_pipeline.run_extend_classification` orchestrates
the streamlined runner.  Phase walk:

```
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING
     → CLASSIFYING → FUSING → EVALUATING → CONVERGED
```

No new FSM states — `SAMPLING → CLASSIFYING` is already a legal
transition (the full pipeline uses the same edge when synthesis is
disabled).  Production guards run BEFORE the FSM run is created:

1. **Artifact-set existence** — DAO lookup must return non-NULL,
   non-archived row.
2. **File-existence preflight** — every non-NULL path on the row must
   exist on disk (catboost + sidecar required; SVM / UMAP optional but
   when set must be present).  Stale DB pointers fail fast.
3. **Embedding-model identity** — the artifact's
   `embedding_model` field must equal the runtime `cfg.embedding_model`.
   Catches the BGE-large vs MiniLM swap that would silently produce
   nonsense predictions.
4. **Vocab compatibility** — surfaces in `progress.vocab_compatibility`
   as one of `ok | superset | partial | disjoint`.  Warns but does NOT
   block (per the project decision); the artifact's training classes
   drive the runtime taxonomy of the extend run.

Inference is intentionally simple — no DST iteration:

- CatBoost `predict_proba` per column → top-1 = primary prediction.
- (Optional) SVM `predict_proba` → second look; soft confidence
  haircut on disagreement.
- `belief = top1_p`, `plausibility = top1_p + (1 − sum_top3)`,
  `conflict = 0.0` (clear "ML-only inference" marker for the UI).

UMAP transforms via `bundle.umap_model.transform()` when the bundle
includes a fitted reducer (lands in the parent run's coordinate
space).  Falls back to a fresh `fit_transform` when no UMAP was bundled
— Extend coordinates differ from the parent's; the divergence is
recorded in `settings_snapshot.json`.

## Gateway endpoints

```
GET    /api/artifact-sets[?source_id=&include_archived=]
GET    /api/artifact-sets/{id}
POST   /api/artifact-sets/{id}/activate
POST   /api/artifact-sets/{id}/archive
POST   /api/artifact-sets/{id}/unarchive
GET    /api/artifact-sets/{id}/compatibility?source_id=
POST   /api/fsm/extend                  body: {source_id,
                                                artifact_set_id,
                                                parent_dataset_id?}
```

The `/api/fsm/extend` endpoint mirrors `/api/fsm/start`'s background-
thread plumbing so the existing `/api/fsm/status` polling carries the
run through to the UI without any new client-side wiring.  Returns 404
synchronously when `artifact_set_id` is missing from the DB; 409 when
another FSM run is in flight.

## UI

The Status page renders a new **ML Artifacts** panel between the
Classification Pipeline panel and the Data Source panel.  Composition
mirrors `DataSourceCard` for visual continuity:

- Header (`extra` slot): active source / dataset indicator (read-only),
  Refresh button, **Extend Classification** primary button.
- Table columns: **Active** (radio) / Run ID (linked to overwatch when
  fsm_run_id is set) / Created / Summary / Models (CB / SVM / UMAP
  chips with informative tooltips) / Archive (trash icon).

The Data Source panel was reworked to match: it now has a leftmost
**Active** column with `Radio` cells, and the Version column lost its
inline `[active]` chip.  Click row OR click radio → activate.

## OpenLineage projection

`atelier.classify.oplineage_emit.build_run_event` projects an FSM run
into an OpenLineage event dict.  The Job is `atelier.classify` or
`atelier.extend_classify`; the Run is `fsm_runs.id`.  Outputs include
the parquet plus one Dataset entry per artifact file (CatBoost / SVM /
UMAP), each carrying a `zndx_ml_artifact` custom facet with framework,
vocab_signature, embedding_model, classes_count.

Extend runs additionally emit a `ParentRunFacet` linking back to the
classify run that produced the consumed ArtifactSet — the OpenLineage-
canonical way to express "this run is a descendant of run X".

Day one we don't wire the HTTP transport — the projection is pure, and
operators who configure Marquez later only need to add the POST
plumbing.  The custom `zndx_ml_artifact` and `zndx_extend_lineage`
facets follow the OpenLineage custom-facet convention with `_producer`
and `_schemaURL` attributes pointing at our schemas.

## BDD coverage

- `features/agent/artifact_set.feature` (tier-0): vocab signature
  determinism, signature stability under reordering, all four
  compatibility statuses (ok / superset / partial / disjoint).
- `features/agent/extend_pipeline.feature` (tier-1, @gpu): Extend
  produces a Dataset with `run_kind='extend'`, the dataset references
  the consumed artifact set, the run NEVER invokes an LLM (structural
  proof — `run_extend_classification` doesn't accept an `llm_backend`
  parameter), vocab compatibility surfaces, atlas-compatible files
  appear in the run dir.
- `features/gateway/artifact_sets.feature` (tier-1, @gpu): seven
  scenarios covering list / get / activate / compatibility / extend
  body validation / 404 paths.

## Out of scope (deferred)

- Auto-prune retention policy for artifact sets (manual archive only).
- Cross-source vocab translation (mapping artifact's classes onto a
  source with a different taxonomy).
- Full per-table input dataset expansion in OpenLineage events
  (currently emits one aggregate input dataset per source).
- HTTP transport for OpenLineage emission — pure projection only.
