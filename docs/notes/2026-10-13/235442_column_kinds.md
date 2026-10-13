# Column kinds close the unanchored genus gap

sdg-corpora now tags every published column. The 207 columns whose recorded
genus does not entail a BFO/CCO anchor take an authored kind
(`ref_basis` `kind-authored`). Each kind is an OWL class under a real BFO or
CCO class, with a SKOS definition and at least one example child, so the
classification label is the kind and an unseen entity of that kind can align
to it.

213 columns carry a kind. That is the 207 gaps plus the coaching client's
surrogate and two foreign keys (off `SDG.MATERIAL_ENTITY.CLIENT`) and the
three culture-dimension denoting columns (off
`SDG.GENERIC.WORKPLACE_CULTURE_DIMENSION`). Leaf concepts such as jurisdiction
stay. Anchored rows that share a name, including financial-advice clients and
process-framed events, stay on their anchor. No column is labeled
`SDG.GENERIC`.

The genus split is 26,446 labeled columns and 363/45/45. Structural genera
are 109. The genus neighborhood is 748 codes at order 1 and 1,005 at order 2.
`best_interest_determinations.topic_id` is
`SDG.ICE.DESCRIPTIVE.DECISION_SUBJECT`. `bus_routes.bus_operator_license_number`
is `SDG.MATERIAL_ENTITY.AGENT_ORGANIZATION`. `prevalence_rate` stays
`SDG.GENERIC.PREVALENCE_RATE`.
