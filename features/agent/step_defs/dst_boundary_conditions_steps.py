"""Step definitions for the DST boundary-conditions feature.

Composes with the existing late-interaction taxonomy builder from
``hierarchical_anti_subtree_steps`` (the "abstract hierarchical
taxonomy" step) for the late-interaction-specific scenario, and
provides per-source mass construction + cautious_promoted_code +
revisit-gate steps for the fusion-outcome scenarios.

Operator-oriented framing throughout: scenarios describe the
classification verdict an operator would observe, not the internal
data structures used to compute it.
"""

from __future__ import annotations

from behave import given, then, when


# ── Multi-depth taxonomy builder ─────────────────────────────────


@given("an abstract multi-depth taxonomy")
def step_build_multi_depth_taxonomy(context):
    """Build a HierarchicalCategorySet from the Gherkin data table.

    Distinct from the ``abstract hierarchical taxonomy`` step in the
    sibling steps module — this one is named for the multi-depth scenarios
    and uses the same data-table pattern, so the choice of step name in
    the Background lets the feature read clearly.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory

    cats: list[ReferenceCategory] = []
    for row in context.table:
        code = row["code"].strip()
        parent = row["parent"].strip() or None
        cats.append(ReferenceCategory(
            code=code, label=code, embedding_text=code,
            abbrev=code, parent_code=parent,
        ))
    parent_codes = {c.parent_code for c in cats if c.parent_code}
    leaves = [c for c in cats if c.code not in parent_codes]
    cs = HierarchicalCategorySet(
        name="test_multi_depth", categories=leaves, all_categories=cats,
    )
    context.category_set = cs
    context.frame = FrameOfDiscernment(cs)
    context.source_masses = {}
    context.fused_masses = None
    # Composability: the anti-example scenario uses the late-interaction
    # ``a late-interaction tag score for ...`` step from the sibling
    # step module, which appends to ``context.late_scores``.  Initialize
    # it here so the multi-depth taxonomy works for that scenario too.
    context.late_scores = []


# ── Per-source vote construction ─────────────────────────────────


def _build_source_masses(context, rows) -> dict:
    """Translate a Gherkin table of (source, code, mass) rows into a
    ``{source: BeliefAssignment}`` dict suitable for fusion or revisit-
    gate evaluation.

    Each row contributes ``mass`` on the appropriate focal element for
    its ``code`` — singleton if the code is a leaf, internal-node FE if
    the code is a parent.  Each source's remaining mass goes to Θ.
    """
    from atelier.classify.belief import BeliefAssignment

    frame = context.frame
    by_source: dict[str, dict] = {}
    for row in rows:
        src = row["source"].strip()
        code = row["code"].strip()
        mass = float(row["mass"])
        if code in frame.singletons:
            fe = frame.singletons[code]
        elif code in frame.internal_nodes:
            fe = frame.internal_nodes[code]
        else:
            raise AssertionError(
                f"Code {code!r} is neither a leaf nor an internal node "
                f"in the test taxonomy."
            )
        by_source.setdefault(src, {})[fe] = (
            by_source.setdefault(src, {}).get(fe, 0.0) + mass
        )

    out: dict = {}
    for src, masses in by_source.items():
        allocated = sum(masses.values())
        masses[frame.theta] = max(0.0, 1.0 - allocated)
        out[src] = BeliefAssignment(masses=masses)
    return out


@given("the following per-source classification votes")
def step_source_votes_initial(context):
    context.source_masses = _build_source_masses(context, context.table)


@when("the per-source votes are")
def step_source_votes_replace(context):
    """Replace the source_masses (for the non-regression branch of a scenario)."""
    context.source_masses = _build_source_masses(context, context.table)


# ── Revisit-gate boundary ────────────────────────────────────────


# Composes with the existing bootstrap-state mechanism but in
# operator-oriented terms: "the revisit gate fires for the column" is
# what an operator observes when targeted_revisit promotes a column for
# re-classification.
@when("the indep_revisit_mass_threshold is {threshold:f}")
def step_set_indep_threshold(context, threshold: float):
    context._indep_threshold = float(threshold)


@when("the indep_revisit_mass_threshold is lowered to {threshold:f}")
def step_lower_indep_threshold(context, threshold: float):
    context._indep_threshold = float(threshold)


def _evaluate_revisit_gate(context) -> bool:
    """Return True iff the revisit gate would fire for this column.

    Mirrors ``bootstrap._identify_disagreements`` semantics without
    pulling in the full BootstrapState machinery — the indep-tier
    code is the argmax of the cosine + pattern + name_match
    consensus, and the gate fires when:
      (a) indep_top1 != llm_code, AND
      (b) indep_top1_mass >= indep_revisit_mass_threshold.
    """
    from atelier.classify.belief import combine_multiple

    src = context.source_masses
    if "llm" not in src:
        return False

    # LLM top1 code
    llm_top1_code = _argmax_singleton_code(src["llm"], context.frame)

    # Indep tier: cosine ⊕ pattern ⊕ name_match if present
    indep_sources = [src[name] for name in ("cosine", "pattern", "name_match")
                     if name in src]
    if not indep_sources:
        return False
    try:
        indep_fused, _k = combine_multiple(indep_sources, strategy="dempster", theta=context.frame.theta)
    except ValueError:
        # Total conflict among indep sources — gate cannot fire
        return False

    indep_top1_code, indep_top1_mass = _top_singleton_mass(indep_fused, context.frame)
    if indep_top1_code is None:
        return False

    if indep_top1_code == llm_top1_code:
        return False
    return indep_top1_mass >= float(context._indep_threshold)


def _argmax_singleton_code(ba, frame) -> str | None:
    best_code, best_mass = None, -1.0
    for fe, m in ba.masses.items():
        if fe == frame.theta:
            continue
        if len(fe.codes) == 1:
            code = next(iter(fe.codes))
            if m > best_mass:
                best_code, best_mass = code, m
    return best_code


def _top_singleton_mass(ba, frame) -> tuple[str | None, float]:
    # Also include internal-node focal elements as candidates — per
    # every-tag-is-first-class.
    candidates: list[tuple[str, float]] = []
    fe_to_code: dict = {}
    for code, fe in frame.singletons.items():
        fe_to_code[fe] = code
    for code, fe in frame.internal_nodes.items():
        fe_to_code[fe] = code
    for fe, m in ba.masses.items():
        if fe == frame.theta:
            continue
        code = fe_to_code.get(fe)
        if code is not None:
            candidates.append((code, m))
    if not candidates:
        return None, 0.0
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return candidates[0]


@then("the bootstrap revisit gate should fire for the column")
def step_revisit_fires(context):
    fired = _evaluate_revisit_gate(context)
    assert fired, (
        f"Expected revisit gate to fire under threshold "
        f"{context._indep_threshold}, but it did not."
    )


@then("the bootstrap revisit gate should not fire for the column")
def step_revisit_does_not_fire(context):
    fired = _evaluate_revisit_gate(context)
    assert not fired, (
        f"Expected revisit gate not to fire under threshold "
        f"{context._indep_threshold}, but it did."
    )


# ── Fused mass + cautious_promoted_code ──────────────────────────


@given("fused mass on focal elements")
def step_fused_mass(context):
    """Construct a fused BeliefAssignment from the Gherkin table.

    Table columns: ``focal_element_codes`` (Gherkin literal like
    ``{a, b, c}`` or ``Θ``) and ``mass``.  Theta is normalized to the
    frame's Θ focal element.
    """
    from atelier.classify.belief import BeliefAssignment, FocalElement

    masses: dict = {}
    for row in context.table:
        raw = row["focal_element_codes"].strip()
        mass = float(row["mass"])
        if raw in ("Θ", "theta", "Theta"):
            masses[context.frame.theta] = mass
            continue
        # Parse {a, b, c}
        s = raw.lstrip("{").rstrip("}")
        codes = frozenset(p.strip() for p in s.split(",") if p.strip())
        # Use the frame's labeled FE when one matches by codes (so the
        # cautious_code traversal sees consistent references); FE
        # equality is codes-only anyway, but reusing the labeled FE
        # keeps display strings clean.
        fe: FocalElement | None = None
        for s_code, s_fe in context.frame.singletons.items():
            if s_fe.codes == codes:
                fe = s_fe
                break
        if fe is None:
            for i_code, i_fe in context.frame.internal_nodes.items():
                if i_fe.codes == codes:
                    fe = i_fe
                    break
        if fe is None:
            fe = FocalElement(codes)
        masses[fe] = masses.get(fe, 0.0) + mass

    context.fused_mass = BeliefAssignment(masses=masses)


@when("the cautious commit_threshold is {threshold:f}")
def step_cautious_threshold(context, threshold: float):
    context._cautious_threshold = float(threshold)


@when("the cautious commit_threshold is lowered to {threshold:f}")
def step_cautious_threshold_lower(context, threshold: float):
    context._cautious_threshold = float(threshold)


def _compute_cautious_promoted_code(context) -> str | None:
    """Compute cautious_promoted_code from the fused mass + threshold.

    Mirrors ``HierarchicalClassification.cautious_promoted_code``
    semantics: walk every singleton and internal-node focal element in
    the frame, compute Bel(fe) over the fused mass, and return the
    most-specific code whose Bel meets the threshold.

    Specificity ordering: smaller descendant set first (singleton < parent),
    deterministic tie-break by code.
    """
    candidates: list[tuple[int, str, float]] = []  # (depth_rank, code, bel)
    frame = context.frame
    fused = context.fused_mass

    for code, fe in frame.singletons.items():
        bel = fused.belief(fe)
        if bel >= context._cautious_threshold:
            candidates.append((1, code, bel))
    for code, fe in frame.internal_nodes.items():
        bel = fused.belief(fe)
        if bel >= context._cautious_threshold:
            # Internal nodes are *less* specific — descendant set size matters.
            # Lower specificity_rank = more specific, so internal nodes use
            # 1 + (descendant count) as the rank, while singletons get 1.
            candidates.append((1 + len(fe.codes), code, bel))

    if not candidates:
        return None
    # Most specific (smallest rank) first; tie-break by higher bel,
    # then alphabetically.
    candidates.sort(key=lambda c: (c[0], -c[2], c[1]))
    return candidates[0][1]


@then('cautious_promoted_code should equal "{code}"')
def step_cautious_eq(context, code: str):
    actual = _compute_cautious_promoted_code(context)
    assert actual == code, (
        f"cautious_promoted_code at threshold "
        f"{context._cautious_threshold} expected {code!r}, got {actual!r}"
    )
