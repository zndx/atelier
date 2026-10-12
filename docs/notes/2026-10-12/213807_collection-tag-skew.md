# Augment each collection split for tag skew

Holding out a whole collection moves every column of that bundle to one side, so the SKOS tag histogram on the support collections and the holdout collections diverge. `distribution_deficit` measures that skew from the real assigned columns.

The pooled share of a code is its count on both sides divided by the number of assigned columns. Each side's target is that share times the side's own real size, rounded up. The deficit is the synthetic relational tables added on that side. A side with no real columns takes the other side's counts, so the empty split still carries every tag. Codes already at the target are left alone.

Support deficits become training rows (`plan.rows`). Holdout deficits become `plan.holdout_rows` and CSVs under `holdout/tables/holdout_*.csv`. Those files contain the generated column names and values. The SKOS code stays on the row. A skewed code with no caller-supplied generator raises `InSituError` and does not load the installed generator module. The code-stratified path still amplifies from the support floor.

On the patrol/clinic fixture the support side has one column each of `PDF_EXPORT`, `PARENTING`, and `YOUNG_RESEARCHER` (3 columns) and the clinic holdout has one `PDF_EXPORT` and one `YOUNG_RESEARCHER` (2 columns). The fill is one synthetic `PDF_EXPORT` and one synthetic `YOUNG_RESEARCHER` on the support side, and one synthetic `PARENTING` table on the holdout side.

`tests/classify/test_insitu_corpus.py` passed (17). No head was fit.
