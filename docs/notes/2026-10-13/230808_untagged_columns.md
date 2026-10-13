# Published columns are all labeled

`align_genus_split` reported 6,610 unlabeled columns because
`reference_code` was empty. The answer key now walks each coined genus to the
BFO/CCO anchor recorded on that class in the published ontology, and uses
`SDG.GENERIC` when the ontology records no specific anchor. Audit timestamps
carry the domain class's code. The split is 26,446 labeled columns, 0 unlabeled,
still 363/45/45 collections.
