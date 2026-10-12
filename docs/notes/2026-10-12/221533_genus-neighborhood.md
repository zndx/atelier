# Neighborhood of the 89 structural genera

The decisions opening Choice is the 89 structural genera. NHSVM scores an ancestor path, so column type annotation has to train parent, genus, and child as distinct labels. `genus_neighborhood` collects every code within graph distance `order` of those genera, walking parent and child edges, then closes the set under ancestors.

Order 1 is the CTA default. On the published vocabulary it is 703 codes at depths 0 through 4: the genera, their parents, and their children. `SDG.GDC`, `SDG.ICE`, and `SDG.ICE.DESCRIPTIVE` each keep their own code. `SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION` stays itself. `genus_term` still returns `SDG.ICE.DESCRIPTIVE` for that leaf.

Order 2 is 890 codes. The extra 187 are siblings of the genera, and the maximum depth stays 4. `SDG.INDEPENDENT_CONTINUANT.CONDIMENT_STATION` and `SDG.PROCESS.PDF_EXPORT_PROCESS` are in that ring. At order 1 they take the nearest in-frame ancestor, `SDG.INDEPENDENT_CONTINUANT` and `SDG.PROCESS`. `prepare_collection_cta(..., order=2)` keeps the sibling code.

A taxonomy with no structural genera still uses `genus_term`. The Process and Role collection fixture stays on those roots. The opening Choice of the decisions walk stays the 89 genera.

`tests/classify/test_insitu_corpus.py` and `tests/classify/test_decisions_backend.py` passed (35). No head was fit.
