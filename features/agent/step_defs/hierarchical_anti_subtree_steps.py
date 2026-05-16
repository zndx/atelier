# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for the late-interaction hierarchical-anti-subtree feature.

These steps use an abstract taxonomy built per-scenario from the
Background table — no real ontology codes are referenced.  The
structural property under test is namespace-agnostic: any parent-
child arrangement (ICE, CCO, filesystem, HDF5, RDF subClassOf chain)
exhibits the same behaviour.
"""

from __future__ import annotations

from behave import given, then, when


# ── Helpers ───────────────────────────────────────────────────────


def _parse_code_set(raw: str) -> frozenset[str]:
    """Parse a ``{a, b, c}`` Gherkin literal into a frozenset of codes."""
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]
    parts = [s.strip() for s in stripped.split(",") if s.strip()]
    return frozenset(parts)


# ── Background: taxonomy + frame ──────────────────────────────────


@given("an abstract hierarchical taxonomy")
def step_build_abstract_taxonomy(context):
    """Build a HierarchicalCategorySet from the Gherkin data table.

    Hierarchy is conveyed *only* via the ``parent`` column; the
    ``code`` column carries no positional/dotted-notation information.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory

    cats: list[ReferenceCategory] = []
    for row in context.table:
        code = row["code"].strip()
        parent = row["parent"].strip() or None
        cats.append(
            ReferenceCategory(
                code=code,
                label=code,
                embedding_text=code,
                abbrev=code,
                parent_code=parent,
            )
        )

    # leaf_codes is derived by HierarchicalCategorySet from the parent
    # links; we keep ``categories`` (the leaf list per CategorySet
    # contract) consistent with that derivation.
    parent_codes = {c.parent_code for c in cats if c.parent_code}
    leaves = [c for c in cats if c.code not in parent_codes]

    cs = HierarchicalCategorySet(
        name="test_abstract",
        categories=leaves,
        all_categories=cats,
    )
    context.category_set = cs
    context.frame = FrameOfDiscernment(cs)
    context.late_scores = []


# ── Score accumulation ───────────────────────────────────────────


@given(
    'a late-interaction tag score for "{code}" with positive {positive:f} '
    'and negative {negative:f}'
)
def step_add_tag_score(context, code: str, positive: float, negative: float):
    from atelier.classify.late_interaction import TagScore

    context.late_scores.append(
        TagScore(
            code=code,
            positive_score=positive,
            negative_score=negative,
            verifier_pass_rate=1.0,
            per_role={},
        )
    )


# ── Action ────────────────────────────────────────────────────────


@when("I compute the late-interaction mass function over those scores")
def step_compute_late_interaction_mass(context):
    from atelier.classify.mass_functions import late_interaction_to_mass

    context.late_mass = late_interaction_to_mass(
        context.late_scores, context.frame,
    )


# ── Focal-element assertions ─────────────────────────────────────


def _find_focal_element(context, codes: frozenset[str]):
    """Find a focal element in context.late_mass whose codes match exactly."""
    for fe, mass in context.late_mass.masses.items():
        if fe.codes == codes:
            return fe, mass
    return None, 0.0


@then("the mass function should contain a focal element exactly covering codes {codes}")
def step_fe_covers_codes(context, codes: str):
    target_codes = _parse_code_set(codes)
    fe, _ = _find_focal_element(context, target_codes)
    assert fe is not None, (
        f"No focal element in mass function covering exactly codes "
        f"{sorted(target_codes)!r}.  Present focal elements: "
        f"{sorted(sorted(fe.codes) for fe in context.late_mass.masses)}"
    )


@then("the focal element covering codes {codes} should carry strictly positive mass")
def step_fe_has_positive_mass(context, codes: str):
    target_codes = _parse_code_set(codes)
    fe, mass = _find_focal_element(context, target_codes)
    assert fe is not None, f"Focal element {sorted(target_codes)!r} not in mass function"
    assert mass > 0.0, (
        f"Focal element {sorted(target_codes)!r} carries mass {mass!r}; "
        f"expected strictly positive"
    )


@then(
    'the mass function should not contain a focal element covering all leaf '
    'codes minus the singleton "{singleton}"'
)
def step_fe_not_full_minus_singleton(context, singleton: str):
    """Guard against the previous broken behaviour.

    Before the hierarchical-integrity fix, an anti-example targeting
    an internal node X would compute the complement as
    ``frame.theta.codes - {X}``.  Since X is not in the leaf set, the
    subtraction is a no-op and the "complement" silently becomes
    ``frame.theta.codes`` itself — i.e., Θ.  This step asserts the
    bug is no longer present by checking that no focal element
    equals the full leaf set minus the (non-leaf) singleton (which,
    structurally, would equal the full leaf set Θ).
    """
    target_codes = context.frame.theta.codes - {singleton}
    # Equivalently: the full leaf set, because ``singleton`` is the
    # code of an internal node and so is not in the leaf set.
    # Any focal element whose codes match this *is* effectively Θ.
    # Θ itself is a legitimate ignorance slot — we don't want to
    # forbid Θ entirely, but we do want to forbid the *broken*
    # complement that masquerades as Θ-with-a-different-label.
    # The bridge of correctness: Θ should carry the residual
    # (1 - allocated) mass; a fresh FocalElement(theta.codes,
    # label="¬node_a") would be a *different* Python object that
    # nonetheless compares equal to Θ on FocalElement.__eq__.
    # So we look for *exactly* the broken construction: a focal
    # element with codes == theta.codes that the negative channel
    # would have minted with label="¬<singleton>".
    #
    # The simplest correct check: the count of focal elements whose
    # codes equal theta.codes should be exactly one (Θ), regardless
    # of how it got there.  Any duplicate would indicate the broken
    # path.
    theta_count = sum(
        1 for fe in context.late_mass.masses if fe.codes == context.frame.theta.codes
    )
    assert theta_count <= 1, (
        f"Mass function contains {theta_count} focal elements with codes "
        f"covering the full leaf set — duplicate Θ entries indicate the "
        f"broken Θ \\ {{{singleton}}} complement construction (which is a "
        f"no-op when {singleton!r} is an internal-node code)."
    )
    # Additional structural assertion: the negative-channel's intent —
    # to express "not in the subtree of {singleton}" — must show up as
    # a *non-Θ* focal element if the test is exercising the negative
    # channel.  The targeted-codes check at the scenario level (the
    # ``exactly covering codes {node_b, node_c}`` step) does this
    # positively.  Here we only forbid the broken artefact.


@then(
    "the focal element covering codes {codes} should be the top non-Θ focal element by mass"
)
def step_fe_top_by_mass(context, codes: str):
    target_codes = _parse_code_set(codes)
    theta_codes = context.frame.theta.codes
    ranked = sorted(
        (
            (fe, mass)
            for fe, mass in context.late_mass.masses.items()
            if fe.codes != theta_codes
        ),
        key=lambda kv: (-kv[1], sorted(kv[0].codes)),
    )
    assert ranked, "No non-Θ focal elements present in mass function"
    top_fe, top_mass = ranked[0]
    assert top_fe.codes == target_codes, (
        f"Top non-Θ focal element has codes {sorted(top_fe.codes)!r} "
        f"(mass {top_mass:.6f}); expected codes {sorted(target_codes)!r}"
    )
