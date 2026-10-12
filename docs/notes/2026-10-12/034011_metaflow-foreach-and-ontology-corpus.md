# Metaflow foreach for System One, and ontology-labeled NHSVM / MaxSim

2026-10-12. The decisions sweep already stores labels from Engine/Decide.
This slice makes the two d3-flash replicas visible in the Metaflow graph
and trains the NHSVM and MaxSim sources from Aegir's known ontology labels.

## Airflow + Metaflow on Kubernetes

The platform pieces were already in place: `require_signals_metaflow`
fail-closes a local datastore, the catalogue DAG id is
`atelier_sdg_classify`, and `airflow create` emits a
`KubernetesPodOperator` for each `@kubernetes` step. The flow body was
still a linear `@step` chain, and `kubernetes-preferred` was stored and
never read. One `sweep` step called the whole DST pipeline, so the second
replica never appeared as a task.

`ClassificationFlow` now fans the sweep out:

```
load → sweep → sweep_shard (foreach) → sweep_join → revisit → fuse → evaluate → end
```

`sweep` sets `sweep_shards = [0, 1]` and `self.next(self.sweep_shard, foreach="sweep_shards")`.
Each shard walks `index % 2 == shard` and calls System One serially
(`parallel=1`). Two shards in flight is the dual-replica overlap. The
join merges labels and fails closed on an overlapping qualified name.
Fusion runs the DST pipeline once with `ReplayDecisionsBackend`, so
MaxSim, NHSVM, and CatBoost see the decision labels and Decide is not
called again. The thinking revisit stays Engine/Complete.

The class body stays `@step`. `python -m atelier.flows.run_classify`
inserts `--with kubernetes` before the Metaflow subcommand, which is how
`airflow create` decorates every step, including the foreach. A caller
that already passed `--with` keeps that choice. `classify_yk_claims()`
is unchanged. The catalogue task stays `enabled=False`.

The in-process `POST /api/fsm/start` path is still one process. It
constructs `DecisionsBackend.from_cfg(cfg, parallel=2)` so both replicas
are busy. A Metaflow shard uses the default `parallel=1`.

This slice did not run `airflow create` against the cluster, and it did
not restart Gaius. The live DAG updates when an operator creates it.

## NHSVM and MaxSim from the projected table

`cfg.aegir_release_dir` (`classify.aegir.release_dir`) was already the
hook and defaulted to empty. When it is set:

- `labeled_rows_from_release` loads `corpus_columns.parquet` with
  `with_reference=True` and keeps columns whose `reference_code` is in
  the taxonomy. That code is the held-back form of Aegir
  `ColumnSpec.slot_ref`, the ontology label on the projected column.
  `build_nhsvm_head` trains on those rows.
- `OntologyLabelGenerator` uses the same column values as MaxSim
  `prototype_values` (padded to three with the ontology label when a
  code is short). The category description on the source row is what
  `compose_annotation_text` already embeds. `build_semantic_collection`
  uses this generator.

An empty release dir keeps the LLM enrichment path and the synthetic
corpus. A directory that is set and then fails to load, or that shares
no codes with the taxonomy, raises `PreconditionError`. Blind
classification still calls `load_aegir_release_samples(..., with_reference=False)`.
`attach_skos_reference` is a name-token match and is not the gold label.
Product MaxSim stays enabled.

## Tests

`uv run pytest tests/classify/test_decisions_backend.py tests/classify/test_ontology_corpus.py tests/flows/test_classification_flow.py tests/engine/test_workload_catalog.py`

39 passed. The graph test asserts `sweep` is a foreach and `sweep_join`
is a join. Five entities split 3 and 2. Replay returns the stored code.
Release rows keep `SDG.A.1` and drop a code outside the taxonomy.
Prototypes are the column values.
