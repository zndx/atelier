# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Mars-Climate-Orbiter guard: semantic absence is surfaced, never silent.

These tests are the standing assertion behind
``feedback_cco_completeness_is_correctness`` and
``feedback_positively_manage_semantic_absence``: a measurement / monetary /
relational classification must carry an explicit ``UNRESOLVED`` token for its
un-modeled interface axis, and any claim depending on that axis must be
gated — so no downstream consumer can silently assume a unit, currency, or
relation (the kg-vs-lb mismatch that destroyed the Mars Climate Orbiter; the
pivot view copied without its association table).
"""
from __future__ import annotations

from atelier.classify.semantic_absence import (
    UNRESOLVED,
    gate_claim,
    semantic_absence,
    unresolved_axes,
)


def test_measurement_surfaces_unit_absence():
    # A column classified as a Quality (mass/length/width) — the MCO case.
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
