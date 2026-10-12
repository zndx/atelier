# Each column carries its genus-level term

Column type annotation labels a column with the genus-level SKOS term on its path, not the leaf business term. `genus_term` walks the published parent chain. The nearest structural genus — a node with both a parent and children, the same nodes `structural_genera` gives the decisions opening Choice — wins. On a branch whose nodes have children but no parent, that root is the genus: `SDG.PROCESS.PDF_EXPORT_PROCESS` is `SDG.PROCESS`, and `SDG.ROLE.PARENTING_ROLE` is `SDG.ROLE`. `SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION` is `SDG.ICE.DESCRIPTIVE`, the nearer genus, rather than `SDG.ICE`.

`associate_genera` applies that map to every `(table, column)` assignment. `prepare_collection_cta` trains and fills tag skew on those genus labels. Two leaf columns under Role share one label. The leaf code stays out of the feature text and out of the holdout CSV.

The published vocabulary has 1,036 codes and 89 structural genera. 336 codes sit on the seven branching roots that have no parent (`PROCESS`, `ROLE`, `GENERIC`, `GDC`, `INDEPENDENT_CONTINUANT`, `QUALITY`, `DISPOSITION`); those columns take the root as their genus.

`tests/classify/test_insitu_corpus.py` and `tests/classify/test_decisions_backend.py` passed (33). No head was fit.
