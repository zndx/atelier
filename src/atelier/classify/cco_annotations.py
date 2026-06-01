# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""CCO ExtendedRelationOntology annotation properties for taxonomy terms.

Atelier's taxonomy terms are Information Content Entities, so CCO's
ExtendedRelationOntology annotation properties apply to them directly — and
several are satisfied by metadata we already produce. Grounding our own terms
in these is the reachable-now slice of the Extended Relation module (the data-
level relations remain EAV/CPA-gated; see docs/src/architecture/cco-coverage.md
and ``cco_modules.json``).

  - ``acronym`` (ont00001753, rdfs:subPropertyOf skos:altLabel) ≡ a term's
    abbrev / mnemonic.
  - ``definition_source`` (ont00001754) ≡ the citation a term's definition was
    drawn from — for the fixture, the DBpedia IRI.
  - ``has_token_unit`` (ont00001752) — the measurement/currency unit of the
    token expressing an ICE; the formal slot a *resolved* unit fills. Left
    unset where the unit is unresolved (that is the semantic absence — see
    :mod:`atelier.classify.semantic_absence`).
"""
from __future__ import annotations

HAS_TOKEN_UNIT = "https://www.commoncoreontologies.org/ont00001752"
ACRONYM = "https://www.commoncoreontologies.org/ont00001753"
DEFINITION_SOURCE = "https://www.commoncoreontologies.org/ont00001754"

# Readable key -> canonical CCO ExtendedRelationOntology property IRI.
CCO_ANNOTATION_PROPERTIES: dict[str, str] = {
    "acronym": ACRONYM,
    "definition_source": DEFINITION_SOURCE,
    "has_token_unit": HAS_TOKEN_UNIT,
}


def ground_term_annotations(
    *,
    mnemonic: str | None = None,
    dbpedia_iri: str | None = None,
    unit: str | None = None,
) -> dict[str, str]:
    """CCO-grounded annotation values for a taxonomy term, from metadata we
    already have. Keys are CCO_ANNOTATION_PROPERTIES names; only the
    properties we can fill are included (an unset ``has_token_unit`` is the
    semantic absence, represented separately).
    """
    ann: dict[str, str] = {}
    if mnemonic:
        ann["acronym"] = mnemonic
    if dbpedia_iri:
        ann["definition_source"] = dbpedia_iri
    if unit:
        ann["has_token_unit"] = unit
    return ann
