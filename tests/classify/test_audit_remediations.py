# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Unit tests for the audit_2026-05-06_a remediation bundle (R1-R6).

Each remediation is exercised against a synthetic fixture that mirrors
its real-run failure mode, so a regression here would re-open the
exact behavior the fix targets.  Paper-trade validation (against
build/results/8d67b1ed/) lives in the conversation transcript and
docs/src/operations/embeddings-reviewer-guide.md addendum, not here.
"""

from __future__ import annotations

import pytest

from atelier.classify.belief import FrameOfDiscernment
from atelier.classify.cautious_review import (
    _extract_json_object,
    _parse_decision,
)
from atelier.classify.mass_functions import _resolve_to_focal_element, llm_to_mass
from atelier.classify.pipeline import _filter_classifiable_tables
from atelier.classify.sampler import TableSample
from atelier.classify.taxonomy import (
    HierarchicalCategorySet,
    ReferenceCategory,
)


def _cat(code: str, label: str, abbrev: str, parent: str | None) -> ReferenceCategory:
    return ReferenceCategory(
        code=code, label=label, embedding_text=label.lower(),
        abbrev=abbrev, parent_code=parent,
    )


def _build_fixture_vocab() -> HierarchicalCategorySet:
    """Tiny hierarchical vocab matching the audit fixture's shape."""
    cats = [
        _cat("0", "Not Sensitive", "NOTSENS", None),
        _cat("0.1", "Internal Non-Sensitive", "INOS", "0"),
        _cat("1", "Sensitive", "SENS", None),
        _cat("1.1", "Personally Identifiable Data", "PID", "1"),
        _cat("1.1.1", "Personal Data", "PII", "1.1"),
        _cat("1.1.1.9", "Contact Data", "CONTACT", "1.1.1"),
        _cat("1.1.1.9.1", "Name (Full)", "NAMEFULL", "1.1.1.9"),
        _cat("1.1.1.4", "Address Data", "ADDR", "1.1.1"),
        _cat("1.1.1.4.1", "Address (Full)", "ADDRFULL", "1.1.1.4"),
        _cat("1.1.1.4.1.1", "Mailing Address", "MAILADDR", "1.1.1.4.1"),
    ]
    leaves = [c for c in cats if c.code in {
        "0.1", "1.1.1.9.1", "1.1.1.4.1.1",
    }]
    return HierarchicalCategorySet(
        name="test", categories=leaves, all_categories=cats,
    )


@pytest.fixture
def vocab() -> HierarchicalCategorySet:
    return _build_fixture_vocab()


@pytest.fixture
def frame(vocab: HierarchicalCategorySet) -> FrameOfDiscernment:
    return FrameOfDiscernment(vocab)


# ── R1: annotation-mnemonic fallback ─────────────────────────────────


def test_r1_resolves_leaf_mnemonic(frame: FrameOfDiscernment):
    """LLM emits 'NAMEFULL' instead of '1.1.1.9.1' — R1 recovers."""
    fe = _resolve_to_focal_element("NAMEFULL", frame)
    assert fe is not None, "R1 should resolve leaf mnemonic"
    assert fe.codes == frozenset({"1.1.1.9.1"})


def test_r1_resolves_internal_mnemonic(frame: FrameOfDiscernment):
    """LLM emits 'CONTACT' (parent) — R1 resolves to the internal node."""
    fe = _resolve_to_focal_element("CONTACT", frame)
    assert fe is not None
    # Internal-node FE; covers all leaves under 1.1.1.9.
    assert "1.1.1.9.1" in fe.codes


def test_r1_disabled_returns_none(frame: FrameOfDiscernment):
    """When the flag is off, mnemonic does NOT resolve (preserves existing
    behavior for ablation runs)."""
    fe = _resolve_to_focal_element(
        "NAMEFULL", frame, allow_annotation_fallback=False,
    )
    assert fe is None


def test_r1_recovers_in_llm_to_mass(frame: FrameOfDiscernment):
    """End-to-end: llm_to_mass with a mnemonic produces non-vacuous mass."""
    mass = llm_to_mass("NAMEFULL", 0.95, [], frame, discount=0.15)
    # Should not be vacuous — primary FE got mass.
    assert any(len(fe.codes) <= 2 and m > 0.5 for fe, m in mass.masses.items())


def test_r1_unknown_mnemonic_still_vacuous(frame: FrameOfDiscernment):
    mass = llm_to_mass("BOGUS_MNEMONIC", 0.95, [], frame)
    # Vacuous — R1 cannot fabricate codes that aren't in the vocab.
    # The only mass should be on Theta.
    assert len(mass.masses) == 1
    fe = next(iter(mass.masses.keys()))
    assert fe.codes == frame.theta.codes


# ── R2b: markdown fence + extra-data extraction ──────────────────────


def test_r2b_strips_markdown_fence():
    text = '```json\n{"decision": "keep", "rationale": "ok"}\n```'
    raw = _extract_json_object(text)
    assert raw == '{"decision": "keep", "rationale": "ok"}'


def test_r2b_handles_trailing_prose():
    """Bedrock Sonnet sometimes emits trailing commentary after JSON."""
    text = (
        '```json\n{"decision": "reroute", "code": "1.1.1.9.1"}\n```\n\n'
        'Note: I considered keeping but values look like names.'
    )
    raw = _extract_json_object(text)
    decision = __import__("json").loads(raw)
    assert decision["decision"] == "reroute"
    assert decision["code"] == "1.1.1.9.1"


def test_r2b_unfenced_still_works():
    text = '{"decision": "keep", "rationale": "fine"}'
    raw = _extract_json_object(text)
    assert "keep" in raw


def test_r2b_balanced_braces_with_nested():
    """Nested objects don't trip the brace-counter."""
    text = (
        '{"decision": "reroute", "code": "1.1.1.9.1", '
        '"metadata": {"source": "values", "n": 10}}'
    )
    raw = _extract_json_object(text)
    decision = __import__("json").loads(raw)
    assert decision["metadata"]["n"] == 10


def test_r2b_extra_data_after_object_isolated():
    text = '{"decision": "reroute", "code": "1.1.1.9.1"}\n{"side": "data"}'
    raw = _extract_json_object(text)
    decision = __import__("json").loads(raw)
    assert decision["code"] == "1.1.1.9.1"


def test_r2b_no_braces_raises():
    with pytest.raises(ValueError, match="no JSON"):
        _extract_json_object("just prose, no braces")


# ── R2c: shortlist-permissive fallback ───────────────────────────────


def test_r2c_in_shortlist_unchanged():
    text = '{"decision": "reroute", "code": "1.1.1.9.1", "rationale": "x"}'
    decision = _parse_decision(
        text,
        valid_codes={"1.1.1.9.1", "0.1"},
        fallback_codes={"1.1.1.9.1", "0.1", "1.1.1.4.1.1"},
    )
    assert decision["code"] == "1.1.1.9.1"
    assert decision["shortlist_extended"] is False


def test_r2c_outside_shortlist_inside_taxonomy_accepted():
    """LLM picks 1.1.1.4.1.1 — outside shortlist (which has just 0.1, 1.1.1.9.1)
    but inside the runtime taxonomy.  R2c accepts as shortlist-extended."""
    text = '{"decision": "reroute", "code": "1.1.1.4.1.1", "rationale": "x"}'
    decision = _parse_decision(
        text,
        valid_codes={"1.1.1.9.1", "0.1"},
        fallback_codes={"1.1.1.9.1", "0.1", "1.1.1.4.1.1"},
    )
    assert decision["code"] == "1.1.1.4.1.1"
    assert decision["shortlist_extended"] is True


def test_r2c_outside_both_rejected():
    """Truly hallucinated code (not in shortlist or taxonomy) — rejected."""
    text = '{"decision": "reroute", "code": "999.999", "rationale": "x"}'
    with pytest.raises(ValueError, match="hallucinated"):
        _parse_decision(
            text,
            valid_codes={"1.1.1.9.1"},
            fallback_codes={"1.1.1.9.1", "0.1"},
        )


def test_r2c_no_fallback_falls_back_to_strict():
    """When fallback_codes is None, behavior matches the strict pre-R2c
    closed-set check."""
    text = '{"decision": "reroute", "code": "1.1.1.4.1.1", "rationale": "x"}'
    with pytest.raises(ValueError, match="hallucinated"):
        _parse_decision(text, valid_codes={"1.1.1.9.1"})


# ── R6: temp-table filter ────────────────────────────────────────────


def test_r6_drops_hue_tmp():
    samples = [
        TableSample(name="customers", database="default", columns=[]),
        TableSample(name="hue__tmp_ecommerce_orders", database="default", columns=[]),
        TableSample(name="orders", database="default", columns=[]),
    ]
    out = _filter_classifiable_tables(samples, None, exclude_temp_tables=True)
    names = {t.name for t in out}
    assert "hue__tmp_ecommerce_orders" not in names
    assert {"customers", "orders"} <= names


def test_r6_drops_tmp_prefix():
    samples = [
        TableSample(name="tmp_query_a1b2", database="default", columns=[]),
        TableSample(name="real_table", database="default", columns=[]),
    ]
    out = _filter_classifiable_tables(samples, None, exclude_temp_tables=True)
    names = {t.name for t in out}
    assert "tmp_query_a1b2" not in names
    assert "real_table" in names


def test_r6_disabled_keeps_temp_tables():
    samples = [
        TableSample(name="hue__tmp_x", database="default", columns=[]),
        TableSample(name="real", database="default", columns=[]),
    ]
    out = _filter_classifiable_tables(samples, None, exclude_temp_tables=False)
    names = {t.name for t in out}
    assert {"hue__tmp_x", "real"} <= names


def test_r6_case_insensitive():
    samples = [
        TableSample(name="HUE__TMP_X", database="default", columns=[]),
    ]
    out = _filter_classifiable_tables(samples, None, exclude_temp_tables=True)
    assert len(out) == 0


# ── R7: walk-down / walk-up in resolve_annotation ────────────────────


def _frame_with_uncurated_node() -> tuple[FrameOfDiscernment, HierarchicalCategorySet]:
    """Fixture exercising R7 walk-up: simulates a projection where the
    abbrev's code is in ``all_by_abbrev`` but neither ``_singletons``
    nor ``_internal`` cover it.  We construct the frame, then mutate
    ``_singletons`` / ``_internal`` to drop GAPNODE — modeling a
    downstream restriction step that prunes nodes after frame
    construction (e.g., per-source vocabulary masking).
    """
    cats = [
        _cat("0", "Not Sensitive", "NOTSENS", None),
        _cat("0.1", "Internal Non-Sensitive", "INOS", "0"),
        _cat("1", "Sensitive", "SENS", None),
        _cat("1.1", "PID", "PID", "1"),
        _cat("1.1.1", "PII", "PII", "1.1"),
        _cat("1.1.1.9", "Contact", "CONTACT", "1.1.1"),
        _cat("1.1.1.9.1", "Name", "NAMEFULL", "1.1.1.9"),
        _cat("1.1.1.99", "Gap Node", "GAPNODE", "1.1.1"),
    ]
    leaves = [c for c in cats if c.code in {"0.1", "1.1.1.9.1", "1.1.1.99"}]
    cs = HierarchicalCategorySet(name="r7", categories=leaves, all_categories=cats)
    frame = FrameOfDiscernment(cs)
    # Simulate a downstream projection that removes 1.1.1.99 — abbrev
    # stays in cs.all_by_abbrev (LLM may still emit GAPNODE), but the
    # frame no longer covers it.  This is the shape vocabulary
    # restrictions take in practice.
    frame._singletons.pop("1.1.1.99", None)
    frame._internal.pop("1.1.1.99", None)
    return frame, cs


def test_r7_walk_up_to_curated_ancestor():
    """GAPNODE (code=1.1.1.99) post-projection walks up to its nearest
    curated ancestor (1.1.1 / PII)."""
    frame, cs = _frame_with_uncurated_node()
    assert "1.1.1.99" not in frame.singletons
    assert "1.1.1.99" not in frame.internal_nodes
    fe = frame.resolve_annotation("GAPNODE")
    assert fe is not None
    # Should resolve to the PII (1.1.1) internal node — covers the leaf.
    assert "1.1.1.9.1" in fe.codes


def test_r7_walk_down_to_descendants():
    """When an uncurated node has in-frame descendants but no curated FE,
    walk-down builds an ad-hoc FE covering the descendants."""
    cats = [
        _cat("0", "Not Sensitive", "NOTSENS", None),
        _cat("0.1", "Internal Non-Sensitive", "INOS", "0"),
        _cat("1", "Sensitive", "SENS", None),
        _cat("1.1.1.9", "Contact", "CONTACT", "1"),  # parent skips levels
        _cat("1.1.1.9.1", "Name", "NAMEFULL", "1.1.1.9"),
        _cat("1.1.1.9.2", "Phone", "PHONE", "1.1.1.9"),
    ]
    leaves = [c for c in cats if c.code in {"0.1", "1.1.1.9.1", "1.1.1.9.2"}]
    cs = HierarchicalCategorySet(name="r7d", categories=leaves, all_categories=cats)
    frame = FrameOfDiscernment(cs)
    # Mutate _internal so 1.1.1.9 is dropped — simulates a projection.
    frame._internal.pop("1.1.1.9", None)
    fe = frame.resolve_annotation("CONTACT")
    assert fe is not None
    # Walk-down should find both leaves under 1.1.1.9.
    assert fe.codes == frozenset({"1.1.1.9.1", "1.1.1.9.2"})


def test_r7_unresolvable_returns_none():
    """An abbrev not in all_by_abbrev still returns None — R7 doesn't
    fabricate."""
    frame, _ = _frame_with_uncurated_node()
    assert frame.resolve_annotation("NEVER_HEARD_OF_IT") is None


# ── R9: semantic llm_agreement ───────────────────────────────────────


def test_r9_semantic_match_via_mnemonic(vocab: HierarchicalCategorySet):
    """LLM emits 'NAMEFULL'; fusion picks '1.1.1.9.1'.  R9 counts as
    agreement (the mnemonic resolves to the same code)."""
    from atelier.classify.pipeline import _evaluate_results

    classifications = [
        {
            "predicted_code": "1.1.1.9.1", "llm_code": "NAMEFULL",
            "matches_reference": None, "confidence": 0.9,
            "conflict": 0.0, "uncertainty": 0.1,
        },
    ]
    out = _evaluate_results(classifications, vocab)
    assert out["llm_agreement"] == 1.0
    # Strict metric counts as miss.
    assert out["llm_agreement_strict"] == 0.0


def test_r9_strict_match_unchanged(vocab: HierarchicalCategorySet):
    """When llm_code already equals predicted_code, both metrics agree."""
    from atelier.classify.pipeline import _evaluate_results

    classifications = [
        {
            "predicted_code": "1.1.1.9.1", "llm_code": "1.1.1.9.1",
            "matches_reference": None, "confidence": 0.9,
            "conflict": 0.0, "uncertainty": 0.1,
        },
    ]
    out = _evaluate_results(classifications, vocab)
    assert out["llm_agreement"] == 1.0
    assert out["llm_agreement_strict"] == 1.0


# ── R10: _mass_summary surfaces internal-node FEs ────────────────────


def test_r10_internal_node_fe_surfaces(frame: FrameOfDiscernment):
    """An internal-node FE in the assignment is reported with a `*`
    suffix and the parent's code, not filtered out."""
    from atelier.classify.belief import BeliefAssignment
    from atelier.classify.pipeline import _mass_summary

    # 1.1.1 (PII) covers two leaves — qualifies as a multi-code internal FE.
    pii_fe = frame.internal_nodes["1.1.1"]
    name_fe = frame.singletons["1.1.1.9.1"]
    ba = BeliefAssignment(masses={
        pii_fe: 0.6,
        name_fe: 0.25,
        frame.theta: 0.15,
    })
    summary = _mass_summary(ba, frame)
    assert "1.1.1*" in summary
    assert summary["1.1.1*"] == 0.6
    assert summary["1.1.1.9.1"] == 0.25
    # Theta should not be in summary.
    assert all("Θ" not in k for k in summary)


def test_r10_frameless_fallback_uses_label():
    """When called without a frame (e.g., legacy tests), internal-node
    FEs fall back to using the FE's display label as the key."""
    from atelier.classify.belief import BeliefAssignment, FocalElement
    from atelier.classify.pipeline import _mass_summary

    fe = FocalElement(frozenset({"a", "b"}), label="Group")
    leaf = FocalElement(frozenset({"c"}), label="Leaf")
    ba = BeliefAssignment(masses={fe: 0.7, leaf: 0.3})
    summary = _mass_summary(ba)  # no frame
    assert "Group*" in summary
    assert summary["Group*"] == 0.7
    assert summary["c"] == 0.3


# ── R8: abbrev_unreachable_in_frame diagnostic ───────────────────────


def test_r8_no_findings_when_clean(vocab: HierarchicalCategorySet):
    """A frame that covers every abbrev produces no abbrev_unreachable
    findings."""
    from atelier.classify.taxonomy import validate_taxonomy

    frame = FrameOfDiscernment(vocab)
    findings = validate_taxonomy(vocab, frame=frame)
    assert not [f for f in findings if f.kind == "abbrev_unreachable_in_frame"]


def test_r8_flags_unreachable_abbrev():
    """When a projection drops an abbrev's code, all its descendants,
    and all its ancestors from the frame, the validator emits an
    ``abbrev_unreachable_in_frame`` warning (R7 walk-up has nothing to
    find)."""
    from atelier.classify.taxonomy import validate_taxonomy

    cats = [
        _cat("0", "Not Sensitive", "NOTSENS", None),
        _cat("0.1", "Internal Non-Sensitive", "INOS", "0"),
        _cat("1", "Sensitive", "SENS", None),
        _cat("1.1.1.9.1", "Name", "NAMEFULL", "1"),
        _cat("1.99", "Stranded", "STRAND", "1"),
    ]
    leaves = [c for c in cats if c.code in {"0.1", "1.1.1.9.1", "1.99"}]
    cs = HierarchicalCategorySet(name="r8", categories=leaves, all_categories=cats)
    frame = FrameOfDiscernment(cs)
    # Project STRAND and every ancestor out of the frame entirely so
    # walk-up has nowhere to land.
    frame._singletons.pop("1.99", None)
    frame._internal.pop("1", None)

    findings = validate_taxonomy(cs, frame=frame)
    unreachable = [f for f in findings if f.kind == "abbrev_unreachable_in_frame"]
    assert any("STRAND" in f.detail for f in unreachable)


def test_r8_no_frame_skips_check(vocab: HierarchicalCategorySet):
    """validate_taxonomy without a frame does NOT emit unreachable
    findings — preserves the legacy taxonomy-only API surface."""
    from atelier.classify.taxonomy import validate_taxonomy

    findings = validate_taxonomy(vocab)
    assert not [f for f in findings if f.kind == "abbrev_unreachable_in_frame"]
