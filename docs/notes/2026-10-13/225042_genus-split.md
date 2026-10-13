# Collection splits use only published sdg-corpora terms

2026-10-13. `align_genus_split` in `src/atelier/classify/collection_split.py`
joins every published collection column to
`external/sdg-corpora/corpus/tagging/answer_key.csv` and assigns each
collection to train, test, or validate. The checkout is sdg-corpora
`9df44148` ("Record each collection column's OWL class and the SKOS
concept that class entails").

The SKOS label is `genus_term` of the key's `reference_code`, and that
code must already be in `vocabulary/annotations.csv`. A leaf gathers to
its genus parent. A code that is already a genus stays itself. An empty
`reference_code` keeps its `class_iri` and an empty `genus_code`. A
code outside the vocabulary raises `InSituError`. `SDG.GENERIC` appears
only when the published parent chain of a coded column lands there.

On this key that is 26,446 columns, 19,836 labeled and 6,610 unlabeled.
The labeled columns use 38 vocabulary genera. The twenty `SDG.GENERIC`
labels are among those 19,836. Zero unlabeled rows received a genus.
Every row has a `class_iri`. The 6,610 unlabeled columns are the key's
empty `reference_code` rows (`genus-unmapped` and `audit-unclaimed`).
They travel with their collection as sibling context. The key publishes
no vocabulary concept for them.

Whole collections stay on one side, including both tables and the
unlabeled columns. Rarest genera are reserved first: one collection in
train, a genus in two or more collections also in test, a genus in
three or more also in validate. The remainder fills toward 80/10/10.
The published corpus lands at 363 train, 45 test, and 45 validate
collections, with every learnable genus on the sides its collection
count allows.

Feature text for a later exam is the pipeline's discrete features
(column name, SQL type, sampled values, siblings, table). The genus
code is not written into that text. `build_nhsvm_head` still reads
`corpus/cta_answer_key.csv` and was not run. No head was fit or
promoted.

`uv run pytest tests/classify/test_collection_split.py` passed (4).
The live test checks the 26,446-row identity, the 19,836/6,610 split,
the 363/45/45 collection sides, `computational_material_records.status`
(`sdg:ComputationalMaterialRecord`, `sdg:status`, `xsd:string`,
`SDG.ARTIFACT`), and `clie_client_engagements.engagement_id`
(`sdg:ClientEngagement`, genus parent of its leaf code).
