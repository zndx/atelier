# NHSVM and MaxSim read sdg-corpora

The ontology labels for those two DST sources come from the local
`external/sdg-corpora` checkout. A deployment of Atelier has that
repository on disk. The decisions sweep still fans out one Metaflow
shard per d3-flash replica; this note only changes where the training
labels come from.

`src/atelier/classify/ontology_corpus.py` reads one DDL pin
(`naming_map.parquet` + `base_rows.parquet`) and
`vocabulary/annotations.csv`.

- A naming-map `code` or `skos_code`, when the corpora repo carries it,
  is the column's SKOS code. That is the field an upstream publish can
  fill for general use.
- Otherwise the code is the vocabulary row whose `abbrev` matches
  `slot_ref`. A surrogate `__pk__` uses `template_id`. A duplicated
  abbrev resolves when exactly one hit has `family=catalog`.
- Columns that do not resolve are omitted. On the current pin that
  join yields 1,238 columns and 523 codes.
- No overlap with the loaded taxonomy leaves LLM enrichment and the
  synthetic NHSVM corpus in place. A missing or unreadable checkout
  fails closed.
- The decisions shard loads an sdg-corpora sample (`just sdg-sample`)
  or `ootb-sample`.

sdg-strategy's lens snapshots carry aperture labels and vector hashes
and no SKOS codes, so this binding does not read them. The generator
name is `ontology:sdg-corpora`.

`docs/notes/2026-10-12/034011_metaflow-foreach-and-ontology-corpus.md`
records the foreach and an earlier release-directory hook. That hook
is not the source.
