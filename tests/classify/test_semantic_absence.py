"""Epistemic-completeness guard: semantic absence is surfaced, never silent.

The standing assertion behind ``feedback_positively_manage_semantic_absence``
and ``feedback_cco_completeness_is_correctness``: an un-modeled interface axis
(a value's unit / currency / relation) is frame-incompleteness uncertainty
and must be represented explicitly — a measurement / monetary / relational
classification carries an ``UNRESOLVED`` token for that axis, and any claim
depending on it is gated, so no downstream consumer can silently assume.
(The canonical real-world cost of such a silent unit assumption is the Mars
Climate Orbiter; the relational case is a pivot view copied without its
association table.)
"""
from __future__ import annotations

from atelier.classify.semantic_absence import (
    UNRESOLVED,
    cco_property_for_axis,
    gate_claim,
    semantic_absence,
    unresolved_axes,
)


def test_unit_absence_is_grounded_in_cco_has_token_unit():
    # The absence axis is the unfilled instance of a canonical CCO property
    # (ExtendedRelationOntology has_token_unit ont00001752), not an ad-hoc str.
    assert cco_property_for_axis("unit") == (
        "https://www.commoncoreontologies.org/ont00001752"
    )
    assert cco_property_for_axis("currency_unit") == (
        "https://www.commoncoreontologies.org/ont00001752"
    )


def test_measurement_surfaces_unit_absence():
    # A column classified as a Quality (mass/length/width).
    absence = semantic_absence("QUAL")
    assert absence == {"unit": UNRESOLVED}, (
        "a measurement classification must positively represent the missing "
        "unit, not drop it silently"
    )


def test_monetary_surfaces_currency_absence():
    assert semantic_absence("CUR") == {"currency_unit": UNRESOLVED}


def test_non_quantity_has_no_interface_absence():
    # An Information-Entity leaf (name/title/id) carries no value-level unit.
    assert semantic_absence("INFO") == {}
    assert semantic_absence("AGENT") == {}


def test_relation_absence_only_in_relational_context():
    # A pivot/materialized view detached from its association table: the
    # relation axis is absent only when we're reading relationally AND the
    # Extended Relation module is still un-modeled (pending in the manifest).
    assert "relation" not in unresolved_axes("INFO")
    assert "relation" in unresolved_axes("INFO", relational_context=True)


def test_resolving_the_axis_clears_absence():
    # An EAV unit column (or a unit resolver) fills the axis -> no longer absent.
    assert semantic_absence("QUAL", resolved={"unit"}) == {}


def test_claim_gated_until_axis_resolved():
    # "these two mass columns are compatible" depends on {'unit'}.
    ok, blocking = gate_claim("QUAL", {"unit"})
    assert ok is False and blocking == ["unit"], "must refuse, not guess"

    ok2, blocking2 = gate_claim("QUAL", {"unit"}, resolved={"unit"})
    assert ok2 is True and blocking2 == []


def test_classification_exposes_absence_from_category_module():
    # The output wiring: a classification whose predicted category carries a
    # referent cco_module surfaces the absence; an ICE-only category does not.
    from atelier.classify.belief import HierarchicalClassification
    from atelier.classify.taxonomy import ReferenceCategory

    quality_cat = ReferenceCategory(
        code="SDG.QUAL.MASS", label="mass", embedding_text="mass", cco_module="QUAL",
    )
    hc = HierarchicalClassification(category=quality_cat, confidence=0.9, evidence="")
    assert hc.semantic_absences == {"unit": UNRESOLVED}

    ice_cat = ReferenceCategory(code="ICE.X", label="x", embedding_text="x")
    hc_ice = HierarchicalClassification(category=ice_cat, confidence=0.9, evidence="")
    assert hc_ice.semantic_absences == {}
