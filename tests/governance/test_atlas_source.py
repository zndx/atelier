"""Reading Ægir's Atlas projection into Atelier inputs.

Uses a tiny in-memory stub of the Atlas client so the relationship-shape
and glossary-text parsing are pinned without a live Atlas (the live
round-trip is exercised by the BDD ``@tier-1`` scenario / manual verify).
"""
from __future__ import annotations

from atelier.governance.atlas_source import (
    GlossaryTerm,
    _parse_category,
    _parse_surface_forms,
    read_glossary,
    read_table_schema,
    read_tables,
)

# A real term longDescription as emitted by Ægir's Atlas projector.
_LONG_DESC = (
    "is a process that with supporting evidence\n\n"
    "Axiom (Manchester): Class: {X:Class} SubClassOf: bfo:0000015, "
    "sdg:withSupportingEvidence some {Y:Class}\n\n"
    "Surface forms (SKOS altLabel / retrieval): attestation, evidence, process, supporting"
)


class _StubAtlas:
    """Canned responses matching Atlas' rdbms_* + glossary payload shapes."""

    def __init__(self):
        self._table = {
            "entity": {
                "attributes": {"name": "t_control_basic", "qualifiedName": "footprint.t_control_basic@aegir"},
                "relationshipAttributes": {
                    "columns": [
                        {"displayText": "id", "qualifiedName": "footprint.t_control_basic.id@aegir"},
                        {"displayText": "subject", "qualifiedName": "footprint.t_control_basic.subject@aegir"},
                        {"displayText": "priority", "qualifiedName": "footprint.t_control_basic.priority@aegir"},
                    ]
                },
            }
        }

    def basic_search(self, type_name, *, limit=25, offset=0, **kw):
        return {"entities": [
            {"attributes": {"qualifiedName": "footprint.t_control_basic@aegir"}},
            {"attributes": {"qualifiedName": "corpus.view_t_control_basic@aegir"}},  # filtered out
        ]}

    def get_entity_by_qualified_name(self, type_name, qn):
        return self._table if qn == "footprint.t_control_basic@aegir" else None

    def get_all_glossaries(self):
        return [{"guid": "G1", "name": "Aegir Ontology"}, {"guid": "G0", "name": None}]

    def get_glossary_terms(self, guid, *, limit=100, offset=0):
        if guid != "G1" or offset > 0:
            return []
        return [{
            "name": "attestation_with_supporting_evidence",
            "qualifiedName": "attestation_with_supporting_evidence@Aegir Ontology",
            "shortDescription": "is a process that with supporting evidence",
            "longDescription": _LONG_DESC,
            "usage": "Existential restriction anchored to bfo:Process, in the 'directive_governance' category.",
        }]


def test_parse_surface_forms():
    assert _parse_surface_forms(_LONG_DESC) == ["attestation", "evidence", "process", "supporting"]
    assert _parse_surface_forms("") == []
    assert _parse_surface_forms("no surface line here") == []


def test_parse_category():
    assert _parse_category("... in the 'directive_governance' category.") == "directive_governance"
    assert _parse_category("") == ""


def test_read_table_schema_drops_surrogate_and_keeps_semantic():
    ts = read_table_schema(_StubAtlas(), "footprint.t_control_basic@aegir")
    assert ts is not None
    assert ts.name == "t_control_basic"
    assert ts.database == "footprint"
    # surrogate 'id' dropped, semantic columns retained, no values from Atlas
    assert [c.name for c in ts.columns] == ["subject", "priority"]
    assert all(c.values == [] for c in ts.columns)


def test_read_tables_filters_corpus_views():
    tables = read_tables(_StubAtlas(), db_prefix="footprint", limit=10)
    assert [t.name for t in tables] == ["t_control_basic"]  # corpus view excluded


def test_read_glossary_builds_vocabulary():
    terms = read_glossary(_StubAtlas())
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, GlossaryTerm)
    assert t.name == "attestation_with_supporting_evidence"
    assert t.surface_forms == ["attestation", "evidence", "process", "supporting"]
    assert t.category == "directive_governance"
    assert t.axiom.startswith("Class: {X:Class}")
    # retrieval text == term words + surface forms, de-duped, for name/maxsim
    assert "attestation" in t.retrieval_text()
    assert t.retrieval_text().startswith("attestation with supporting evidence")
