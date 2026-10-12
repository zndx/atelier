# Code-stratified relational split

The index split of the projected footprint left 80 of 523 assigned codes only on the holdout (136 columns). Those codes had no real NHSVM training row. A loaded head emits a distribution over the whole taxonomy, so a miss on one of those columns is conflicting singleton mass under Dempster-Shafer.

`split_by_code` keeps up to the support floor of real columns for every assigned code and holds out the spare columns. A code with one column stays in the support census. Its generator teaches that IRI to entities outside this corpus. On the current pin this is 747 assigned support columns, 491 assigned holdout columns, 478 codes on both sides, 45 singleton codes in support only, and 0 holdout-only codes. Unassigned columns stay on the support side and are not classification targets.

`projected-holdout` loads those holdout CSVs in the classification pipeline and uses the published `vocabulary/annotations.csv` as the frame. The CSV files carry no codes. `conflicting_known_entities` is the check that an NHSVM top-1 matches the assigned code before fusion.

The generator harness reads a code list and continues after a failed batch. A code is skipped after three misses, and five empty batches stop the run. `insitu_extensions.py` is still empty. Qwen has not written the 523 procedures yet.

`uv run pytest` on the insitu, generator-harness, synth-harness, and ontology-corpus tests: 32 passed.
