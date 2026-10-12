# In-situ NHSVM and Qdrant from a split relational corpus

The ontology is the label source for training. The projected relational
footprint supplies an unseen corpus: table names, column names, and
cell values. Qdrant prototypes and the NHSVM head are built from
synthetic columns Atelier generates, and the Metaflow precondition
path does not open the naming map.

`src/atelier/classify/insitu_corpus.py` reads `base_rows.parquet` from
the one DDL pin in `external/sdg-corpora`. Columns are stratified by
table. Even indexes within a table are the holdout; odd indexes are
the support census. A one-column table alternates, in table-name
order, starting on the holdout. On the current pin that is 765
support columns and 838 holdout columns, with 517 of 520 tables
present on both sides.

A support column counts toward an ontology code only when its name
matches exactly one vocabulary abbrev or label. Codes with fewer than
2 such columns are under-represented and receive 12 synthetic
columns; codes at or above the floor receive 4. Each synthetic column
is 8 generated values. A keyword shape is used when the ontology text
matches one; every other code gets a value drawn from its own label.
MaxSim prototypes are the same generator (`ontology:insitu`).

The holdout tables are written under `build/relational_split/holdout/tables/`.
`source-id projected-holdout` classifies that half. An absent
checkout still synthesizes every ontology code, which is the bootstrap
for an environment whose relational corpus has not arrived.

`ontology_corpus.py` remains the reader for a published column code.
Precondition does not call it.
