# Exact IRI generators

The sdg-corpora vocabulary is the ontology Aegir published. Every
`SDG.*` code has one SKOS IRI, `https://signals.zndx.org/sdg#` plus
the local name (`SDG.ROLE.PARENTING_ROLE` is
`https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE`). All 1,036
annotation codes map. None of them is a fallback code.

`prepare_insitu` still uses the keyword value shapes when ontology
text matches. For every other term it looks up `insitu_extensions.py`
by that exact IRI. A term with no entry is omitted from the NHSVM
rows. The label-plus-number stamp is gone. A regex is not a way to
bind a generator: one pattern was stamping unrelated terms
(`artifact` onto dataset and eBPF descriptions, `quality` onto
thirteen labels, `information content entity` onto fifteen).

The coding harness (`qwen_generator_harness`, skill
`extend-procedural-generator`) asks local Qwen3.8 thinking to write
`GENERATORS` keyed by the `iri` column. The gate accepts a key only
when it is the requested term's IRI. Genus-abbreviation values for
the grounding nodes (`ICE-`, `INC-`, `GEN-`) were not kept. Thirteen
domain IRIs from the earlier local batches remain. Cerebras was not
called. BFO and CCO genera in this vocabulary are real terms; older
notes that treat "ICE" as the whole ontology, or a dot-code alignment
as a generator, are not this path.
