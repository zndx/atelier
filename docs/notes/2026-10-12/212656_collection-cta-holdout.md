# Collection holdout for column tagging

Column tagging holds out a whole collection of related tables. A collection is one directory under `corpus/collections/<slug>/`: its tables, foreign keys, stored values, and SQL types stay on one side of the exam. `prepare_collection_cta` names each table `slug/table` and writes the holdout CSV as `slug__table.csv`.

The label is an exact SKOS code. The checkout does not publish one on the collection CSVs. `load_collection_answer_key` reads `corpus/cta_answer_key.csv` (`collection,table,column,code,split`) when that file exists. A missing file yields no assignments. A collection named on both sides of the key is an error. Column names are not labels. Loading `animal-control-officer-1da7ef2d` and `academic-appointment-0bf5440d` with an empty assignment produces the split and zero training rows. `animal_control_officer_id` keeps its SQL type `INTEGER`.

NHSVM text is `ColumnFeatures.to_embedding_text()`. That string is the pipeline feature set, including sampled values and sibling columns. Masking `sample_values` or `sibling_context` drops that field and leaves the other. `build_texts_and_labels` builds this text. A head records `feature_text=column_features`, and `NHSVMHeadAdapter._build_text` renders the same string. Head `a40626cf` has no such key, so scoring it still uses the legacy sibling-suffix string. It was not refit.

Synthetic tables join the plan only when the caller passes generators for codes that already have a real support column. `build_nhsvm_head` does not load the installed generator module. With no answer key it raises `PreconditionError` before the encoder import. The column split is not fit. MaxSim prototypes still use `split_by_code` on the projected footprint.

`tests/classify/test_insitu_corpus.py` passed (16).
