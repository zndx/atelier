# SDG.GENERIC is not a classification class

`SDG.GENERIC` gathers catalog patterns that have no shared BFO/CCO anchor.
Using it as the label for an unanchored relational genus put people,
departments, sessions, contracts, and typo IRIs in one class.

Those 207 columns are unlabeled (`genus-unmapped`). The OWL class and genus
stay on the row. A column whose recorded name is a `SDG.GENERIC.*` concept
keeps that concept, and `genus_term` does not promote it to the parent.
The split is 26,239 labeled columns, 207 unlabeled, still 363/45/45.
