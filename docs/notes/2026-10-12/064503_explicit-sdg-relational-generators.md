# Explicit SDG terms, relational generators

The in-situ NHSVM path no longer stamps one scalar per ontology term, and the vocabulary tests no longer invent codes.

A projected column's label is the published SKOS code already known for that column. `prepare_insitu` receives `(table, column) → code` from `load_labeled_columns`. `insitu_corpus.py` still does not read the naming map. Features are the column name, type, values, and the names of other support columns on the same table. Holdout columns are written for classification and stay out of the training rows.

Each assigned IRI needs a relational generator. The callable returns one table. Exactly one column carries that IRI, and the table has at least one sibling column. `code_iri` is the fragment with `SDG.` removed and dots turned into underscores, and only when `vocabulary/vocabulary.ttl` contains that `skos:Concept`. Codes the support half represents fewer than twice get 12 synthetic tables; the others get 4. A target IRI with no generator fails closed. An empty footprint has no targets and synthesizes nothing.

The harness pack (`qwen_generator_harness`) takes an explicit list of those target codes. It does not walk the whole 1,036-term vocabulary. Qwen was not run. `insitu_extensions.py` is empty until those procedures exist.

Tests load a named slice of the published annotations: `SDG.PROCESS`, `SDG.ROLE`, `SDG.PROCESS.PDF_EXPORT_PROCESS`, `SDG.ROLE.PARENTING_ROLE`, `SDG.ROLE.YOUNG_RESEARCHER_ROLE`, `SDG.PROCESS.CLINICAL_DOCUMENTATION_ACTIVITY`, `SDG.MATERIAL_ENTITY.ORGANIZATION`, and `SDG.DOM.ORGANIZATION`. The catalog-family join uses the real `ORGANIZATION` pair. `uv run pytest` on the insitu, generator-harness, synth-harness, and ontology-corpus tests: 27 passed.

The earlier scalar-IRI note `060456` describes the contract this slice replaces.
