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
    # Simulate a projection that removes 1.1.1.9 from the active frame
    # entirely — must pop from BOTH _internal and _singletons since the
    # unified frame puts all codes in singletons.
    frame._internal.pop("1.1.1.9", None)
    frame._singletons.pop("1.1.1.9", None)
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
    # walk-up has nowhere to land.  Must pop from BOTH _singletons and
    # _internal since the unified frame puts all codes in singletons.
    frame._singletons.pop("1.99", None)
    frame._singletons.pop("1", None)
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


# ── Stage A — DST sensitivity-study visibility instrumentation ──────
# Tests for Recs 1, 3, 6 from docs/notes/2026-05-16/dst-sensitivity-findings.md
# Land as zero-calibration-risk additions: surface K, subtree
# concentration, top_kind, and a top-1-margin revisit gate so the
# bel × gap sweep output is self-diagnosing.


from types import SimpleNamespace as _NS


def _ts(code: str, pos: float, neg: float = 0.0, vpr: float = 1.0):
    """Minimal TagScore-shaped object for maxsim_to_mass tests."""
    return _NS(
        code=code, positive_score=pos, negative_score=neg,
        verifier_pass_rate=vpr,
    )


def test_rec1_k_populated_with_negative_evidence(frame: FrameOfDiscernment):
    """K is populated when the negative channel has non-vacuous mass."""
    from atelier.classify.mass_functions import maxsim_to_mass

    # Positive evidence for NAMEFULL (1.1.1.9.1), negative against the
    # same code — high-conflict scenario, K should be > 0.
    scores = [
        _ts("1.1.1.9.1", pos=0.9, neg=0.8),
        _ts("1.1.1.4.1.1", pos=0.3, neg=0.0),
    ]
    mass = maxsim_to_mass(scores, frame)
    assert mass.channel_conflict_k is not None, "K must be populated"
    assert mass.channel_conflict_k > 0.0, "non-trivial conflict expected"


def test_rec1_k_zero_on_vacuous_negative_channel(frame: FrameOfDiscernment):
    """K = 0.0 when no anti-example evidence triggers the negative channel."""
    from atelier.classify.mass_functions import maxsim_to_mass

    scores = [
        _ts("1.1.1.9.1", pos=0.9, neg=0.0),  # zero negative → vacuous
        _ts("1.1.1.4.1.1", pos=0.3, neg=0.0),
    ]
    mass = maxsim_to_mass(scores, frame)
    assert mass.channel_conflict_k == 0.0


def test_rec6_subtree_concentration_populated_for_leaf_top1(
    frame: FrameOfDiscernment,
):
    """When top-1 is a leaf, _significant_subtree fires and
    subtree_concentration is non-None on the returned mass."""
    from atelier.classify.mass_functions import maxsim_to_mass

    # Concentrate cosine in a single leaf subtree — _significant_subtree
    # should find an LCA at some concentration > 0.
    scores = [
        _ts("1.1.1.9.1", pos=0.95, neg=0.0),
        _ts("1.1.1.4.1.1", pos=0.10, neg=0.0),
    ]
    mass = maxsim_to_mass(scores, frame)
    assert mass.subtree_concentration is not None, (
        "leaf top-1 must populate subtree_concentration"
    )
    # In a frame this small with three leaves, concentration falls in [0, 1]
    assert 0.0 <= mass.subtree_concentration <= 1.0


def test_rec6_top_kind_derived_at_pipeline_level(frame: FrameOfDiscernment):
    """top_kind == 'leaf' when predicted_code is NOT in frame.internal_nodes,
    'internal' when it IS an internal-node code.  The derivation rule must
    check internal_nodes (not singletons) because the unified frame puts
    ALL codes in singletons."""
    # This is a direct test of the derivation rule — used inline at
    # pipeline.py:_classify_column.
    leaf_code = next(
        c for c in frame.singletons if c not in frame.internal_nodes
    )
    internal_code = next(iter(frame.internal_nodes.keys()))
    assert ("internal" if leaf_code in frame.internal_nodes else "leaf") == "leaf"
    assert ("internal" if internal_code in frame.internal_nodes else "leaf") == "internal"


def test_rec3_top1_margin_below_threshold_triggers_revisit():
    """A column with tight top-1 vs top-2 margin gets flagged even when
    gap and bel are within tolerances (the rank-instability gate the
    earlier code missed)."""
    from atelier.classify.bootstrap import (
        BootstrapConfig, BootstrapState, _identify_uncertain_columns,
    )

    state = BootstrapState()
    state.labels = {"t.a": "X", "t.b": "Y"}
    # Both columns have healthy gap+bel; only 't.a' has narrow top-1 margin.
    state.ml_belief = {"t.a": 0.75, "t.b": 0.75}
    state.ml_plausibility = {"t.a": 0.85, "t.b": 0.85}
    state.ml_top1_margin = {"t.a": 0.02, "t.b": 0.40}

    cfg = BootstrapConfig(
        gap_threshold=0.30, bel_floor=0.50, top1_margin_threshold=0.05,
    )
    uncertain = _identify_uncertain_columns(state, ["t.a", "t.b"], cfg)
    assert "t.a" in uncertain, (
        "narrow top-1 margin must add column to uncertain list"
    )
    assert "t.b" not in uncertain, (
        "wide top-1 margin must not add column when gap/bel are fine"
    )


def test_rec3_wide_margin_skips_revisit_when_gap_and_bel_fine():
    """Wide top-1 margin alone should NOT mark a column uncertain when
    gap and bel are within thresholds."""
    from atelier.classify.bootstrap import (
        BootstrapConfig, BootstrapState, _identify_uncertain_columns,
    )

    state = BootstrapState()
    state.labels = {"t.x": "Y"}
    state.ml_belief = {"t.x": 0.80}
    state.ml_plausibility = {"t.x": 0.85}
    state.ml_top1_margin = {"t.x": 0.50}

    cfg = BootstrapConfig(
        gap_threshold=0.30, bel_floor=0.50, top1_margin_threshold=0.05,
    )
    uncertain = _identify_uncertain_columns(state, ["t.x"], cfg)
    assert "t.x" not in uncertain


def test_rec3_hardcoded_0_3_replaced_with_cfg_gap_threshold():
    """The latent issue from the audit: line 1275 of bootstrap.py used
    a hardcoded 0.3 instead of cfg.gap_threshold.  With the fix, tightening
    cfg.gap_threshold changes which columns qualify."""
    from atelier.classify.bootstrap import (
        BootstrapConfig, BootstrapState, _identify_uncertain_columns,
    )

    state = BootstrapState()
    state.labels = {"t.q": "Z"}
    state.ml_belief = {"t.q": 0.60}
    state.ml_plausibility = {"t.q": 0.85}  # gap = 0.25
    state.ml_top1_margin = {"t.q": 0.40}  # not a margin issue

    # gap=0.25 should be marked uncertain when threshold=0.18 (the new
    # default), but NOT when threshold=0.30.  Pre-fix this column was
    # always evaluated against the hardcoded 0.3, masking the knob.
    cfg_tight = BootstrapConfig(
        gap_threshold=0.18, bel_floor=0.50, top1_margin_threshold=0.05,
    )
    cfg_loose = BootstrapConfig(
        gap_threshold=0.30, bel_floor=0.50, top1_margin_threshold=0.05,
    )
    uncertain_tight = _identify_uncertain_columns(state, ["t.q"], cfg_tight)
    uncertain_loose = _identify_uncertain_columns(state, ["t.q"], cfg_loose)
    assert "t.q" in uncertain_tight, (
        "0.25 gap should trigger when cfg.gap_threshold=0.18"
    )
    assert "t.q" not in uncertain_loose, (
        "0.25 gap should NOT trigger when cfg.gap_threshold=0.30 — "
        "previously hardcoded 0.3 was matching this case incorrectly"
    )


def test_rec3_default_ml_top1_margin_treats_missing_as_stable():
    """If a column isn't in state.ml_top1_margin (pre-Stage-A or pre-
    populated), default 1.0 (maximally stable) means it doesn't get
    flagged on margin alone — preserves prior behavior."""
    from atelier.classify.bootstrap import (
        BootstrapConfig, BootstrapState, _identify_uncertain_columns,
    )

    state = BootstrapState()
    state.labels = {"t.legacy": "L"}
    state.ml_belief = {"t.legacy": 0.80}
    state.ml_plausibility = {"t.legacy": 0.85}
    # state.ml_top1_margin intentionally empty (pre-Stage-A state shape)

    cfg = BootstrapConfig(
        gap_threshold=0.30, bel_floor=0.50, top1_margin_threshold=0.50,
    )
    uncertain = _identify_uncertain_columns(state, ["t.legacy"], cfg)
    assert "t.legacy" not in uncertain, (
        "missing margin must default to 1.0 (stable); no Stage A "
        "instrumentation should change pre-existing column eligibility"
    )


def test_rec3_top1_margin_threshold_field_propagates_from_atelier_config():
    """BootstrapConfig.top1_margin_threshold is wired from
    AtelierConfig.classify_bootstrap_top1_margin_threshold."""
    from atelier.classify.bootstrap import bootstrap_config_from_cfg
    from atelier.config import AtelierConfig

    cfg = AtelierConfig(classify_bootstrap_top1_margin_threshold=0.12)
    bc = bootstrap_config_from_cfg(cfg)
    assert bc.top1_margin_threshold == 0.12


def test_rec3_top1_margin_excludes_ancestor_and_descendant_focal_elements(
    vocab: HierarchicalCategorySet,
):
    """Regression: an earlier formulation iterated singletons + internals
    indiscriminately, which made ancestors/descendants of the top-1 code
    register as the top-2 candidate.  Their belief is mass-accumulated
    via DST hierarchy and equals (or exceeds) top-1's, collapsing margin
    to 0 universally.  The fix restricts the top-2 search to focal
    elements DISJOINT from top-1's FE.
    """
    from atelier.classify.belief import (
        BeliefAssignment,
        HierarchicalClassification,
    )

    frame = FrameOfDiscernment(vocab)

    # Place 0.80 mass on leaf 1.1.1.9.1 (NAMEFULL); small 0.05 on a
    # disjoint subtree (0.1, the Not-Sensitive branch); rest on Θ.
    masses = {
        frame.singletons["1.1.1.9.1"]: 0.80,
        frame.singletons["0.1"]: 0.05,
        frame.theta: 0.15,
    }
    ba = BeliefAssignment(masses=masses)
    top1_cat = next(
        c for c in vocab.all_categories if c.code == "1.1.1.9.1"
    )
    hc = HierarchicalClassification(
        category=top1_cat,
        confidence=0.8,
        evidence="test",
        belief_assignment=ba,
        _frame=frame,
        _category_set=vocab,
    )

    margin = hc.top1_margin()
    # The competing disjoint singleton sits at 0.05; the top-1 leaf at
    # 0.80; margin must be ~0.75 (within float tolerance of 0.80-0.05).
    # The buggy iteration would have collapsed to 0 because ancestors
    # 1.1.1.9, 1.1.1, 1.1, 1 all share top-1's mass via accumulation.
    assert margin > 0.5, (
        f"margin must reflect disjoint competitor (~0.75), "
        f"not ancestor accumulation (~0); got {margin:.4f}"
    )
    # Sanity: the ancestor's belief equals the top-1's by construction.
    assert hc.belief_at("1.1.1.9") >= hc.belief_at("1.1.1.9.1") - 1e-9, (
        "ancestor belief must >= descendant belief (DST invariant); "
        "if this fails, the fixture is wrong, not the margin fix"
    )


def test_rec3_top1_margin_internal_node_top1_excludes_descendants(
    vocab: HierarchicalCategorySet,
):
    """When the top-1 prediction is an internal-node tag, the top-2
    search must exclude leaves within its subtree (whose FE is a
    subset of top-1's accumulated FE)."""
    from atelier.classify.belief import (
        BeliefAssignment,
        HierarchicalClassification,
    )

    frame = FrameOfDiscernment(vocab)
    # Mass on internal node 1.1.1.9 (Contact Data); disjoint mass on 0.1.
    masses = {
        frame.internal_nodes["1.1.1.9"]: 0.70,
        frame.singletons["0.1"]: 0.10,
        frame.theta: 0.20,
    }
    ba = BeliefAssignment(masses=masses)
    top1_cat = next(c for c in vocab.all_categories if c.code == "1.1.1.9")
    hc = HierarchicalClassification(
        category=top1_cat,
        confidence=0.7,
        evidence="test",
        belief_assignment=ba,
        _frame=frame,
        _category_set=vocab,
    )

    margin = hc.top1_margin()
    # Competing disjoint singleton 0.1 sits at 0.10; top-1 internal at
    # 0.70 → expected margin ~0.60.  Descendants like 1.1.1.9.1 sit at 0
    # but their FE is a subset of top-1's FE, so they must be excluded.
    assert margin > 0.4, (
        f"internal-node top-1 must score margin against disjoint "
        f"focal elements (~0.60), not descendants; got {margin:.4f}"
    )
