# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for Dempster-Shafer classification BDD scenarios."""

from behave import given, when, then


# ── Vocabulary ───────────────────────────────────────────────────────


@given("the mock annotations vocabulary is loaded")
def step_load_mock_vocab(context):
    from atelier.classify.taxonomy import load_universal_vocabulary
    context.category_set = load_universal_vocabulary(hierarchical=True)
    assert len(context.category_set.categories) > 0


@given("a frame of discernment from the vocabulary")
def step_build_frame(context):
    from atelier.classify.belief import FrameOfDiscernment
    if not hasattr(context, "category_set"):
        from atelier.classify.taxonomy import load_universal_vocabulary
        context.category_set = load_universal_vocabulary(hierarchical=True)
    context.frame = FrameOfDiscernment(context.category_set)
    assert len(context.frame.singletons) > 0


# ── Belief Assignment ────────────────────────────────────────────────


@when('I create a belief assignment with mass {m1:g} on "{code}" and {m2:g} on theta')
def step_create_bpa(context, m1, code, m2):
    from atelier.classify.belief import BeliefAssignment
    frame = context.frame
    masses = {
        frame.singleton(code): float(m1),
        frame.theta: float(m2),
    }
    context.bpa = BeliefAssignment(masses=masses)


@then('the belief for "{code}" should be approximately {expected:g}')
def step_check_belief(context, code, expected):
    bel = context.bpa.belief(context.frame.singleton(code))
    assert abs(bel - float(expected)) < 0.05, f"Belief {bel} != ~{expected}"


@then('the plausibility for "{code}" should be approximately {expected:g}')
def step_check_plausibility(context, code, expected):
    pl = context.bpa.plausibility(context.frame.singleton(code))
    assert abs(pl - float(expected)) < 0.05, f"Plausibility {pl} != ~{expected}"


@then('the uncertainty for "{code}" should be approximately {expected:g}')
def step_check_uncertainty(context, code, expected):
    unc = context.bpa.uncertainty(context.frame.singleton(code))
    assert abs(unc - float(expected)) < 0.05, f"Uncertainty {unc} != ~{expected}"


# ── Dempster Combination ─────────────────────────────────────────────


@given('two independent evidence sources both supporting "{code}"')
def step_two_sources(context, code):
    from atelier.classify.belief import BeliefAssignment
    frame = context.frame
    context.source1 = BeliefAssignment(masses={
        frame.singleton(code): 0.6,
        frame.theta: 0.4,
    })
    context.source2 = BeliefAssignment(masses={
        frame.singleton(code): 0.5,
        frame.theta: 0.5,
    })


@when("I combine them via Dempster's rule")
def step_dempster_combine(context):
    from atelier.classify.belief import dempster_combine
    context.combined, context.conflict = dempster_combine(
        context.source1, context.source2
    )


@then('the combined belief for "{code}" should exceed either source alone')
def step_combined_exceeds(context, code):
    singleton = context.frame.singleton(code)
    combined_bel = context.combined.belief(singleton)
    bel1 = context.source1.belief(singleton)
    bel2 = context.source2.belief(singleton)
    assert combined_bel > max(bel1, bel2), (
        f"Combined {combined_bel} should exceed max({bel1}, {bel2})"
    )


@then("the conflict K should be less than {threshold:g}")
def step_conflict_threshold(context, threshold):
    assert context.conflict < float(threshold), (
        f"Conflict {context.conflict} >= {threshold}"
    )


# ── Feature Extraction ───────────────────────────────────────────────


@when('I extract features for column "{name}" of type "{col_type}" with email values')
def step_extract_features_email(context, name, col_type):
    from atelier.classify.features import extract_features
    context.features = extract_features(
        column_name=name,
        column_type=col_type,
        values=["alice@example.com", "bob@test.org", "carol@mail.net"],
    )


@then("all 12 feature names should be present")
def step_check_12_features(context):
    from atelier.classify.features import FEATURE_NAMES
    assert len(context.features.feature_names) == 12
    assert context.features.feature_names == FEATURE_NAMES


@then('the pattern signals should include "{pattern}"')
def step_check_pattern_signal(context, pattern):
    assert pattern in context.features.pattern_signals, (
        f"{pattern} not in {context.features.pattern_signals}"
    )


@then('the embedding text should contain "{text}"')
def step_check_embedding_text(context, text):
    et = context.features.to_embedding_text()
    assert text in et, f"'{text}' not in '{et}'"


# ── Pattern Detection ────────────────────────────────────────────────


@when('I run pattern detection on SSN values "{values_str}"')
def step_detect_ssn(context, values_str):
    from atelier.classify.features import detect_patterns
    values = [v.strip() for v in values_str.split(",")]
    context.detected_patterns = detect_patterns(values)


@when('I run pattern detection on credit card values "{values_str}"')
def step_detect_cc(context, values_str):
    from atelier.classify.features import detect_patterns
    values = [v.strip() for v in values_str.split(",")]
    context.detected_patterns = detect_patterns(values)


@when('I run pattern detection on values "{values_str}"')
def step_detect_generic(context, values_str):
    from atelier.classify.features import detect_patterns
    values = [v.strip() for v in values_str.split(",")]
    context.detected_patterns = detect_patterns(values)


@then('the detected patterns should include "{pattern}"')
def step_check_detected_pattern(context, pattern):
    assert pattern in context.detected_patterns, (
        f"{pattern} not in {context.detected_patterns}"
    )


@then('the detected patterns should not include "{pattern}"')
def step_check_detected_pattern_absent(context, pattern):
    assert pattern not in context.detected_patterns, (
        f"{pattern} unexpectedly in {context.detected_patterns}"
    )


# ── Name Matching ────────────────────────────────────────────────────


@when('I run name matching for column "{column_name}"')
def step_name_match(context, column_name):
    from atelier.classify.mass_functions import name_match_to_mass
    context.name_mass = name_match_to_mass(
        column_name, context.frame, context.category_set
    )


@then("the name match mass function should not be vacuous")
def step_not_vacuous(context):
    masses = context.name_mass.masses
    assert len(masses) > 1, f"Mass function has only {len(masses)} focal elements"


@then('the top singleton should be "{code}"')
def step_top_singleton(context, code):
    best_code = None
    best_mass = 0.0
    for fe, m in context.name_mass.masses.items():
        if len(fe.codes) == 1 and m > best_mass:
            best_code = next(iter(fe.codes))
            best_mass = m
    assert best_code == code, f"Top singleton is {best_code}, expected {code}"


# ── Pignistic Probability ────────────────────────────────────────────


@then('the pignistic probability for "{code}" should exceed {threshold:g}')
def step_check_pignistic(context, code, threshold):
    betp = context.bpa.pignistic_probability(context.frame.singleton(code))
    assert betp > float(threshold), f"BetP({code}) = {betp} <= {threshold}"


# ── HierarchicalClassification ──────────────────────────────────────


@when("I build a HierarchicalClassification from combined evidence")
def step_build_hc(context):
    from atelier.classify.belief import HierarchicalClassification
    source_masses = {"source1": context.source1, "source2": context.source2}
    context.hc = HierarchicalClassification.from_combined_evidence(
        source_masses=source_masses,
        frame=context.frame,
        category_set=context.category_set,
    )


@then('belief at leaf "{code}" should be positive')
def step_hc_leaf_belief(context, code):
    bel = context.hc.belief_at(code)
    assert bel > 0, f"Belief at {code} = {bel}"


@then('belief at parent "{code}" should be at least as high as at "{leaf}"')
def step_hc_parent_belief(context, code, leaf):
    bel_parent = context.hc.belief_at(code)
    bel_leaf = context.hc.belief_at(leaf)
    assert bel_parent >= bel_leaf - 1e-9, (
        f"Parent belief {bel_parent} < leaf belief {bel_leaf}"
    )


@then("the classification should report whether clarification is needed")
def step_hc_needs_clarification(context):
    # Just verify the property is accessible and returns a bool
    result = context.hc.needs_clarification
    assert isinstance(result, bool), f"needs_clarification returned {type(result)}"


# ── Schema Mapping Validation ────────────────────────────────────────


@then('category "{code}" label should be "{expected}"')
def step_check_label(context, code, expected):
    cat = context.category_set.all_by_code.get(code)
    assert cat is not None, f"Category {code} not found"
    assert cat.label == expected, f"Label is '{cat.label}', expected '{expected}'"


@then('category "{code}" abbrev should be "{expected}"')
def step_check_abbrev(context, code, expected):
    cat = context.category_set.all_by_code.get(code)
    assert cat is not None, f"Category {code} not found"
    assert cat.abbrev == expected, f"Abbrev is '{cat.abbrev}', expected '{expected}'"


# ── Pipeline ─────────────────────────────────────────────────────────


@when("I run the classification pipeline with mock data")
def step_run_pipeline(context):
    from atelier.config import load_config
    from atelier.classify import run_pipeline
    from atelier.classify.sampler import load_fixture_samples
    from atelier.classify.mock_llm import RealisticMockLLMBackend

    cfg = load_config()
    samples = load_fixture_samples()
    reference_labels = {c.name: c.reference_code for ts in samples for c in ts.columns if c.reference_code}
    context.pipeline_result = run_pipeline(
        cfg, samples=samples, llm_backend=RealisticMockLLMBackend(reference_labels=reference_labels),
    )


@then("the pipeline should reach CONVERGED state")
def step_pipeline_converged(context):
    state = context.pipeline_result.get("state")
    assert state == "CONVERGED", (
        f"Pipeline state is {state}, error: {context.pipeline_result.get('error')}"
    )


@then("the results should contain at least {n:d} classified columns")
def step_min_columns(context, n):
    count = context.pipeline_result.get("classifications", 0)
    assert count >= n, f"Only {count} classified columns, expected >= {n}"


@then("the accuracy against the curated reference should exceed {threshold:g}")
def step_accuracy_threshold(context, threshold):
    acc = context.pipeline_result.get("accuracy")
    if acc is None:
        assert False, "No accuracy computed (no curated reference?)"
    assert acc > float(threshold), f"Accuracy {acc} <= {threshold}"


@then("the micro-F1 should exceed {threshold:g}")
def step_micro_f1_threshold(context, threshold):
    report = context.pipeline_result.get("evaluation_report")
    assert report is not None, "No evaluation_report in pipeline result"
    micro_f1 = report.get("micro_f1", 0.0)
    assert micro_f1 > float(threshold), f"Micro F1 {micro_f1} <= {threshold}"


# ── Evaluation Report ────────────────────────────────────────────────


@then("the evaluation report should contain per-category metrics")
def step_eval_per_category(context):
    report = context.pipeline_result.get("evaluation_report")
    assert report is not None, "No evaluation_report in pipeline result"
    per_cat = report.get("per_category", [])
    assert len(per_cat) > 0, "No per-category metrics in evaluation report"


@then("every category with support > 0 should have precision and recall")
def step_eval_precision_recall(context):
    report = context.pipeline_result["evaluation_report"]
    for cat in report["per_category"]:
        if cat["support"] > 0:
            assert cat["precision"] >= 0.0, (
                f"Category {cat['code']} missing precision"
            )
            assert cat["recall"] >= 0.0, (
                f"Category {cat['code']} missing recall"
            )


@then("the evaluation report should contain a confusion matrix")
def step_eval_confusion(context):
    report = context.pipeline_result["evaluation_report"]
    cm = report.get("confusion_matrix", [])
    assert len(cm) > 0, "No confusion matrix entries"
    for entry in cm:
        assert "true_code" in entry
        assert "predicted_code" in entry
        assert "count" in entry and entry["count"] > 0


# ── Configurable Discounts ───────────────────────────────────────


@given("custom discount factors with cosine {cosine:g} and svm {svm:g}")
def step_custom_discounts(context, cosine, svm):
    from atelier.classify.mass_functions import DiscountConfig
    context.custom_discounts = DiscountConfig(
        cosine=float(cosine),
        svm=float(svm),
    )


@when("I run the classification pipeline with custom discounts")
def step_run_pipeline_custom_discounts(context):
    from atelier.config import load_config
    from atelier.classify import run_pipeline
    from atelier.classify.sampler import load_fixture_samples
    from atelier.classify.mock_llm import RealisticMockLLMBackend

    samples = load_fixture_samples()
    reference_labels = {c.name: c.reference_code for ts in samples for c in ts.columns if c.reference_code}

    # Run default first
    cfg = load_config()
    context.default_result = run_pipeline(
        cfg, samples=samples, llm_backend=RealisticMockLLMBackend(reference_labels=reference_labels),
    )
    # Run with custom discounts (override config fields)
    cfg2 = load_config()
    cfg2.classify_discount_cosine = context.custom_discounts.cosine
    cfg2.classify_discount_svm = context.custom_discounts.svm
    context.pipeline_result = run_pipeline(
        cfg2, samples=samples, llm_backend=RealisticMockLLMBackend(reference_labels=reference_labels),
    )


@then("the average confidence should differ from default discounts")
def step_check_confidence_differs(context):
    default_conf = context.default_result.get("avg_confidence", 0.0)
    custom_conf = context.pipeline_result.get("avg_confidence", 0.0)
    # With higher discounts (more mass → Theta), confidence should change
    assert abs(default_conf - custom_conf) > 0.001, (
        f"Default avg_confidence={default_conf:.4f} == custom={custom_conf:.4f}"
    )


# ── Pattern Mass Functions ───────────────────────────────────────────


@when('I compute pattern_to_mass for signals {signals_str}')
def step_compute_pattern_mass(context, signals_str):
    import json
    from atelier.classify.mass_functions import pattern_to_mass
    signals = json.loads(signals_str)
    context.pattern_mass = pattern_to_mass(signals, context.frame)


@then('the pattern mass should assign weight to "{code}"')
def step_pattern_mass_weight(context, code):
    singleton = context.frame.singleton(code)
    mass = context.pattern_mass.masses.get(singleton, 0.0)
    assert mass > 0, f"No mass on {code}, masses: {context.pattern_mass.masses}"


@then("the pattern mass should not be vacuous")
def step_pattern_mass_not_vacuous(context):
    masses = context.pattern_mass.masses
    assert len(masses) > 1, f"Mass function has only {len(masses)} focal elements (vacuous)"


@then("the pattern mass should be vacuous")
def step_pattern_mass_is_vacuous(context):
    # Vacuous = 100% mass on Theta only.  One focal element (Theta) with mass 1.0.
    masses = context.pattern_mass.masses
    theta = context.frame.theta
    theta_mass = masses.get(theta, 0.0)
    assert len(masses) == 1 and abs(theta_mass - 1.0) < 1e-9, (
        f"Pattern mass is not vacuous: {len(masses)} elements, theta_mass={theta_mass}; "
        f"masses={masses}"
    )


# ── FSM ──────────────────────────────────────────────────────────────


@given("a fresh AgentFSM")
def step_fresh_fsm(context):
    from atelier.classify.fsm import AgentFSM
    context.fsm = AgentFSM()


@given('a fresh AgentFSM in "{state}" state')
def step_fresh_fsm_state(context, state):
    from atelier.classify.fsm import AgentFSM
    context.fsm = AgentFSM()
    run = context.fsm.start_run()
    context.run_id = run.id


@when("I start a new run")
def step_start_run(context):
    run = context.fsm.start_run()
    context.run_id = run.id


@when('I advance to "{state}"')
def step_advance(context, state):
    from atelier.classify.fsm import FSMState
    context.fsm.advance(context.run_id, FSMState(state))


@when('I attempt to advance to "{state}"')
def step_attempt_advance(context, state):
    from atelier.classify.fsm import FSMState
    try:
        context.fsm.advance(context.run_id, FSMState(state))
        context.transition_error = None
    except ValueError as e:
        context.transition_error = str(e)


@then('the state should be "{expected}"')
def step_check_state(context, expected):
    run = context.fsm.get_status(context.run_id)
    assert run.state.value == expected, f"State is {run.state.value}, expected {expected}"


@then("the transition should be rejected with an error")
def step_transition_rejected(context):
    assert context.transition_error is not None, "Expected transition error"


# ── Pattern map alias resolver ──────────────────────────────────


@given('a vocabulary with abbrev "{abbrev}" at code "{code}"')
@given('a non-ICE vocabulary with abbrev "{abbrev}" at code "{code}"')
def step_non_ice_vocab(context, abbrev, code):
    """Build a tiny test vocabulary whose codes are outside Atelier's
    own ICE namespace.

    Used by scenarios that exercise namespace-agnostic mechanisms
    (resolver, sensitivity-map activation gate) — the codes can be
    drawn from any publicly-grounded ontology (BFO, CCO, DPV) or a
    domain-extension namespace; the mechanism doesn't care.  See
    ``src/atelier/classify/fixtures/PROVENANCE.md`` for the public
    sources of the abbrevs themselves.
    """
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    cats = [
        ReferenceCategory(
            code=code, label="Test Term", embedding_text="", abbrev=abbrev,
        ),
        ReferenceCategory(
            code="cco:GenericInformationContentEntity",
            label="Other",
            embedding_text="",
            abbrev="OTHER",
        ),
    ]
    context.numeric_vocab = HierarchicalCategorySet("test", cats, cats)


@when("I resolve the default pattern map against that vocabulary")
def step_resolve_pattern_map(context):
    from atelier.classify.mass_functions import DEFAULT_PATTERN_MAP, resolve_pattern_map
    context.resolved_pattern_map = resolve_pattern_map(
        DEFAULT_PATTERN_MAP, context.numeric_vocab,
    )


@then('the resolved map binds "{pattern}" to "{code}"')
def step_resolved_binds(context, pattern, code):
    actual = context.resolved_pattern_map.get(pattern)
    assert actual == code, (
        f"Expected {pattern}->{code}, got {pattern}->{actual}; "
        f"resolved={context.resolved_pattern_map}"
    )


@then("the resolved map omits patterns whose target abbrev is not in the vocabulary")
def step_resolved_omits(context):
    # Only TXNAMT is in the vocabulary, so monetary_pattern should be the
    # only resolved entry (or at most one or two more if other defaults
    # happen to map through abbrev/aliases — assert the count is small
    # and ssn_pattern (no SSN abbrev in our test vocab) is absent).
    resolved = context.resolved_pattern_map
    assert "ssn_pattern" not in resolved, (
        f"ssn_pattern should not resolve against a vocab without SSN; got {resolved}"
    )
    assert "monetary_pattern" in resolved, (
        f"monetary_pattern should resolve via TXNAMT abbrev; got {resolved}"
    )


# ── Ontology priors threading ───────────────────────────────────


@given("a column whose values match the monetary pattern")
def step_monetary_column(context):
    from atelier.classify.sampler import ColumnSample
    context.monetary_sample = ColumnSample(
        table_name="acme_table",
        name="acme_table.amount_col",
        column_type="object",
        values=["$3559.80", "$1553.91", "$1887.79", "$223.06", "$1899.48"],
        siblings=["acme_table.row_id"],
        total_count=5, null_count=0, distinct_count=5,
    )


@when("I extract features from that column")
def step_extract_features_for_monetary(context):
    from atelier.classify.features import extract_features
    s = context.monetary_sample
    context.monetary_features = extract_features(
        column_name=s.name,
        column_type=s.column_type,
        values=s.values,
        siblings=s.siblings,
        source_table=s.table_name,
        total_count=s.total_count,
        null_count=s.null_count,
        distinct_count=s.distinct_count,
    )


@then('the ontology_priors list contains a "{label}" entry')
def step_ontology_priors_has(context, label):
    priors = context.monetary_features.ontology_priors
    labels = [p.get("label", "") for p in priors]
    assert label in labels, f"Expected {label!r} in {labels}"


@then('the embedding text contains "{needle}"')
def step_embedding_text_contains(context, needle):
    text = context.monetary_features.to_embedding_text()
    assert needle in text, f"Expected {needle!r} in embedding text:\n{text}"


@then('the embedding text contains the alias "{alias}"')
def step_embedding_text_contains_alias(context, alias):
    text = context.monetary_features.to_embedding_text()
    assert alias in text, f"Expected alias {alias!r} in embedding text:\n{text}"


@then('ablating the "{feature}" feature removes those tokens from the embedding text')
def step_ablation_removes_ontology(context, feature):
    f = context.monetary_features
    full_text = f.to_embedding_text()
    mask = {n: True for n in f.feature_names}
    mask[feature] = False
    ablated_text = f.to_embedding_text(mask)
    assert "Transaction Amount" in full_text
    assert "Transaction Amount" not in ablated_text, (
        f"Ablation of {feature!r} did not remove ontology tokens; "
        f"ablated text still contains:\n{ablated_text}"
    )


@when("I build the LLM batch user prompt for that column")
def step_build_user_prompt(context):
    from atelier.classify.llm_backend import build_batch_user_prompt
    context.user_prompt = build_batch_user_prompt(
        [context.monetary_sample], table_name="acme_table",
    )


@then('the prompt contains "{needle}"')
def step_prompt_contains(context, needle):
    assert needle in context.user_prompt, (
        f"Expected {needle!r} in prompt:\n{context.user_prompt}"
    )


# ── Cosine reliability shaping ──────────────────────────────────


@given("a frame with {n:d} singletons")
def step_frame_with_n_singletons(context, n):
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    cats = [
        ReferenceCategory(
            code=f"ICE.TEST.S{i // 30}.L{i % 30}",
            label=f"L{i}",
            embedding_text="",
            abbrev="",
        )
        for i in range(n)
    ]
    cs = HierarchicalCategorySet("scaled-test", cats, cats)
    context.cosine_frame = FrameOfDiscernment(cs)
    context.cosine_codes = [c.code for c in cats]


@given('cosine similarities with top-1 "{label}" at {sim1:g} and top-2 at {sim2:g}')
def step_cosine_similarities(context, label, sim1, sim2):
    codes = context.cosine_codes
    sims = {code: float(sim2) for code in codes[2:]}
    sims[codes[0]] = float(sim1)
    sims[codes[1]] = float(sim2)
    context.cosine_similarities = sims
    context.cosine_top1_code = codes[0]


@when("I convert similarities to mass")
def step_cosine_to_mass(context):
    from atelier.classify.mass_functions import cosine_to_mass
    context.cosine_mass = cosine_to_mass(
        context.cosine_similarities, context.cosine_frame, discount=0.30,
    )


def _parse_band(spec: str) -> tuple[str, float]:
    spec = spec.strip()
    if spec.startswith("at least"):
        return ("ge", float(spec.split()[-1]))
    if spec.startswith("at most"):
        return ("le", float(spec.split()[-1]))
    raise ValueError(f"Unrecognised band spec: {spec!r}")


@then("the top-1 singleton mass is {band}")
def step_top1_mass_band(context, band):
    op, threshold = _parse_band(band)
    fe = context.cosine_frame.singleton(context.cosine_top1_code)
    actual = context.cosine_mass.masses.get(fe, 0.0)
    if op == "ge":
        assert actual >= threshold - 1e-9, (
            f"top-1 mass {actual:.4f} < {threshold:.4f}"
        )
    else:
        assert actual <= threshold + 1e-9, (
            f"top-1 mass {actual:.4f} > {threshold:.4f}"
        )


@then("the Theta mass is {band}")
def step_theta_mass_band(context, band):
    op, threshold = _parse_band(band)
    theta_mass = context.cosine_mass.masses.get(context.cosine_frame.theta, 0.0)
    if op == "ge":
        assert theta_mass >= threshold - 1e-9, (
            f"Theta mass {theta_mass:.4f} < {threshold:.4f}"
        )
    else:
        assert theta_mass <= threshold + 1e-9, (
            f"Theta mass {theta_mass:.4f} > {threshold:.4f}"
        )


# ── Hierarchical mass + cross-subtree belief ────────────────────


def _build_loan_hierarchy():
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    cats = [
        ReferenceCategory(code="0", label="Not Sensitive", embedding_text="", abbrev="", parent_code=None),
        ReferenceCategory(code="0.1", label="Internal Non-Sensitive", embedding_text="", abbrev="INOS", parent_code="0"),
        ReferenceCategory(code="1", label="Sensitive", embedding_text="", abbrev="", parent_code=None),
        ReferenceCategory(code="1.1", label="PID", embedding_text="", abbrev="", parent_code="1"),
        # Sibling subtree under PID so Financial Data has a strictly
        # smaller descendant set than its parent — exercises the
        # "most-specific internal node" tie-break correctly.
        ReferenceCategory(code="1.1.0", label="Contact", embedding_text="", abbrev="", parent_code="1.1"),
        ReferenceCategory(code="1.1.0.1", label="Email", embedding_text="", abbrev="EMAIL", parent_code="1.1.0"),
        ReferenceCategory(code="1.1.1.1", label="Financial Data", embedding_text="", abbrev="", parent_code="1.1"),
        ReferenceCategory(code="1.1.1.1.1", label="Salary", embedding_text="", abbrev="SALARY", parent_code="1.1.1.1"),
        ReferenceCategory(code="1.1.1.1.2", label="Bonus", embedding_text="", abbrev="BONUS", parent_code="1.1.1.1"),
        ReferenceCategory(code="1.1.1.1.3", label="Stock", embedding_text="", abbrev="STOCK", parent_code="1.1.1.1"),
        ReferenceCategory(code="1.1.1.1.4", label="Financial Documentation", embedding_text="", abbrev="FINDOC", parent_code="1.1.1.1"),
    ]
    leaves = [c for c in cats if c.code in {"0.1", "1.1.0.1", "1.1.1.1.1", "1.1.1.1.2", "1.1.1.1.3", "1.1.1.1.4"}]
    return HierarchicalCategorySet("loan-hier", leaves, cats)


@given('a hierarchy with a "Financial Data" parent over four leaves and an "Internal Non-Sensitive" sibling')
def step_loan_hierarchy(context):
    from atelier.classify.belief import FrameOfDiscernment
    context.loan_cs = _build_loan_hierarchy()
    context.loan_frame = FrameOfDiscernment(context.loan_cs)


@given('cosine top-1 is "Salary" at sim {top1:g} with three siblings within "Financial Data" at sim {sim_low:g}-{sim_high:g}')
def step_cosine_within_subtree(context, top1, sim_low, sim_high):
    context.loan_sims = {
        "0.1": 0.20,
        "1.1.1.1.1": float(top1),     # Salary
        "1.1.1.1.2": float(sim_high), # Bonus
        "1.1.1.1.3": float(sim_low),  # Stock
        "1.1.1.1.4": (float(sim_low) + float(sim_high)) / 2.0,  # FinDoc
    }


@when("I convert similarities to mass with hierarchical aggregation")
def step_convert_with_hierarchical(context):
    from atelier.classify.mass_functions import cosine_to_mass
    context.loan_cosine_mass = cosine_to_mass(
        context.loan_sims, context.loan_frame, discount=0.30,
    )


@then('the "{label}" internal node carries non-zero mass')
def step_internal_node_has_mass(context, label):
    matches = [
        (fe, m) for fe, m in context.loan_cosine_mass.masses.items()
        if fe.label == label and len(fe.codes) > 1
    ]
    assert matches, f"No internal-node focal element with label {label!r} carries mass"
    fe, m = matches[0]
    assert m > 1e-9, f"Internal node {label!r} has effectively zero mass: {m}"


@then('belief at "{label_a}" exceeds belief at "{label_b}"')
def step_belief_exceeds(context, label_a, label_b):
    cs = context.loan_cs
    frame = context.loan_frame
    code_a = next(c.code for c in cs.all_categories if c.label == label_a)
    code_b = next(c.code for c in cs.all_categories if c.label == label_b)
    fe_a = frame.internal_nodes.get(code_a) or frame.singletons.get(code_a)
    fe_b = frame.internal_nodes.get(code_b) or frame.singletons.get(code_b)
    bel_a = context.loan_cosine_mass.belief(fe_a)
    bel_b = context.loan_cosine_mass.belief(fe_b)
    assert bel_a > bel_b, f"Bel({label_a})={bel_a:.3f} not > Bel({label_b})={bel_b:.3f}"


@given('a HierarchicalClassification where the LLM voted "{llm_code}" but cosine localizes to "Financial Data"')
def step_hc_llm_vs_cosine(context, llm_code):
    from atelier.classify.belief import FrameOfDiscernment, HierarchicalClassification
    from atelier.classify.mass_functions import cosine_to_mass, llm_to_mass
    cs = _build_loan_hierarchy()
    frame = FrameOfDiscernment(cs)
    sims = {
        "0.1": 0.20,
        "1.1.1.1.1": 0.50,
        "1.1.1.1.2": 0.48,
        "1.1.1.1.3": 0.46,
        "1.1.1.1.4": 0.47,
    }
    cosine_m = cosine_to_mass(sims, frame, discount=0.30)
    llm_m = llm_to_mass(llm_code, 0.92, [], frame, discount=0.10)
    context.loan_hc = HierarchicalClassification.from_combined_evidence(
        source_masses={"cosine": cosine_m, "llm": llm_m},
        frame=frame, category_set=cs,
    )


@when("I compute cautious_code at threshold {threshold:g}")
def step_compute_cautious(context, threshold):
    context.loan_cautious = context.loan_hc.cautious_code(float(threshold))


@when("I list cross_subtree_belief at threshold {threshold:g}")
def step_compute_cross_subtree(context, threshold):
    context.loan_cross = context.loan_hc.cross_subtree_belief(float(threshold))


@then('cross_subtree_belief includes "{label}" as an internal-node entry')
def step_cross_includes_internal(context, label):
    matches = [r for r in context.loan_cross if r["label"] == label and r["kind"] == "internal"]
    assert matches, (
        f"cross_subtree_belief does not include {label!r} as internal-node entry; "
        f"got: {context.loan_cross}"
    )


@then('cross_subtree_belief includes "{label}" as a leaf entry')
def step_cross_includes_leaf(context, label):
    matches = [r for r in context.loan_cross if r["label"] == label and r["kind"] == "leaf"]
    assert matches, (
        f"cross_subtree_belief does not include {label!r} as leaf entry; "
        f"got: {context.loan_cross}"
    )


@then('the evidence string contains "{needle}"')
def step_evidence_contains(context, needle):
    evidence = context.loan_hc.evidence
    assert needle in evidence, (
        f"Expected {needle!r} in evidence string:\n{evidence}"
    )


@then('cross_subtree_belief includes a code from the "{prefix}" subtree')
def step_cross_includes_subtree(context, prefix):
    matches = [r for r in context.loan_cross if r["code"].startswith(prefix)]
    assert matches, (
        f"cross_subtree_belief contains no code starting with {prefix!r}; "
        f"got: {context.loan_cross}"
    )


@when("I compute cautious_promoted_code at commit threshold {threshold:g}")
def step_compute_promoted(context, threshold):
    context.loan_promoted = context.loan_hc.cautious_promoted_code(
        commit_threshold=float(threshold),
    )


@then("promoted_from is null")
def step_promoted_from_null(context):
    assert context.loan_promoted["promoted_from"] is None, (
        f"Expected promoted_from=null, got {context.loan_promoted}"
    )


@then('the rationale mentions "{needle}"')
def step_rationale_mentions(context, needle):
    rationale = context.loan_promoted.get("rationale", "")
    assert needle in rationale, (
        f"Expected {needle!r} in rationale: {rationale}"
    )


# ── Numerical-methods convergence diagnostics ───────────────────


@given("a BootstrapState with high gap, conflict, and indep-tier disagreement")
def step_state_high_residual(context):
    from atelier.classify.bootstrap import BootstrapState, BootstrapConfig
    state = BootstrapState()
    cfg = BootstrapConfig()
    column_names = ["a", "b", "c", "d", "e"]
    for n in column_names:
        state.labels[n] = "0.1"
        state.ml_belief[n] = 0.45
        state.ml_plausibility[n] = 0.85
        state.ml_conflict[n] = 0.55
    state.independent_top1["a"] = "1.1.1.1.1"
    state.independent_top1_mass["a"] = 0.55
    context.numerical_state = state
    context.numerical_cfg = cfg
    context.numerical_columns = column_names


@when("I record iteration {n:d} metrics")
def step_record_iteration(context, n):
    from atelier.classify.bootstrap import record_iteration_metrics
    context.numerical_state.iteration = n
    record_iteration_metrics(
        context.numerical_state,
        context.numerical_columns,
        disagreement_count=2,
        cfg=context.numerical_cfg,
    )


@when("the state improves on the next iteration")
def step_state_improves(context):
    state = context.numerical_state
    for n in context.numerical_columns:
        state.ml_belief[n] = 0.78
        state.ml_plausibility[n] = 0.85
        state.ml_conflict[n] = 0.18
    state.independent_top1_mass["a"] = 0.20


@when("the state does not change")
def step_state_stalls(context):
    pass  # leave state untouched between iterations


@then("the residual_norm decreases between iterations")
def step_residual_decreases(context):
    metrics = context.numerical_state.iteration_metrics
    assert len(metrics) >= 2, "Need at least 2 iterations recorded"
    assert metrics[-1].residual_norm < metrics[-2].residual_norm, (
        f"Expected residual to decrease; got {metrics[-2].residual_norm:.3f} "
        f"→ {metrics[-1].residual_norm:.3f}"
    )


@then("the contraction_rate at iteration {n:d} is below {threshold:g}")
def step_contraction_below(context, n, threshold):
    metrics = context.numerical_state.iteration_metrics
    rate = metrics[n].contraction_rate
    assert rate < float(threshold), (
        f"Iteration {n} contraction_rate {rate:.3f} not < {threshold}"
    )


@then("the contraction_rate at iteration {n:d} is approximately {expected:g}")
def step_contraction_approx(context, n, expected):
    metrics = context.numerical_state.iteration_metrics
    rate = metrics[n].contraction_rate
    assert abs(rate - float(expected)) < 0.05, (
        f"Iteration {n} contraction_rate {rate:.3f} not ≈ {expected}"
    )


# ── Per-column trajectory + ρ_col ───────────────────────────────


@given("a BootstrapState with three labeled columns")
def step_state_three_columns(context):
    from atelier.classify.bootstrap import BootstrapState, BootstrapConfig
    state = BootstrapState()
    cfg = BootstrapConfig()
    column_names = ["a", "b", "c"]
    for n in column_names:
        state.labels[n] = "ICE.NONSENSITIVE"
        state.label_source[n] = "llm"
        state.ml_belief[n] = 0.55
        state.ml_plausibility[n] = 0.85
        state.ml_conflict[n] = 0.40
    context.traj_state = state
    context.traj_cfg = cfg
    context.traj_columns = column_names


@when("I record three successive iteration metrics")
def step_record_three_iterations(context):
    from atelier.classify.bootstrap import record_iteration_metrics
    state = context.traj_state
    for i in range(3):
        state.iteration = i
        record_iteration_metrics(
            state, context.traj_columns, 0, context.traj_cfg,
            revisited_this_iter=set(),
        )


@then("state.column_history has an entry for each labeled column")
def step_column_history_complete(context):
    state = context.traj_state
    for name in context.traj_columns:
        assert name in state.column_history, (
            f"Expected {name!r} in column_history; got {list(state.column_history)}"
        )


@then("each column's snapshot sequence has length {n:d}")
def step_snapshot_sequence_length(context, n):
    state = context.traj_state
    for name in context.traj_columns:
        actual = len(state.column_history[name])
        assert actual == n, (
            f"Expected {n} snapshots for {name!r}; got {actual}"
        )


@then("each column's iteration sequence is contiguous starting at {start:d}")
def step_snapshot_iter_contiguous(context, start):
    state = context.traj_state
    for name in context.traj_columns:
        iters = [s.iteration for s in state.column_history[name]]
        expected = list(range(start, start + len(iters)))
        assert iters == expected, (
            f"Column {name!r} iteration sequence {iters} != expected {expected}"
        )


@when("I record iteration {n:d} with no revisits")
def step_record_no_revisit(context, n):
    from atelier.classify.bootstrap import record_iteration_metrics
    if not hasattr(context, "traj_state"):
        # Allow this step to be the first if the column-fixture wasn't built.
        from atelier.classify.bootstrap import BootstrapState, BootstrapConfig
        state = BootstrapState()
        cfg = BootstrapConfig()
        cols = ["a", "b", "c"]
        for c in cols:
            state.labels[c] = "ICE.NONSENSITIVE"
            state.label_source[c] = "llm"
            state.ml_belief[c] = 0.55
            state.ml_plausibility[c] = 0.85
            state.ml_conflict[c] = 0.40
        context.traj_state = state
        context.traj_cfg = cfg
        context.traj_columns = cols
    context.traj_state.iteration = n
    record_iteration_metrics(
        context.traj_state, context.traj_columns, 0, context.traj_cfg,
        revisited_this_iter=set(),
    )


@when('I record iteration {n:d} with column "{col}" revisited')
def step_record_with_revisit(context, n, col):
    from atelier.classify.bootstrap import record_iteration_metrics
    context.traj_state.iteration = n
    # Improve the revisited column's residuals slightly so the
    # snapshot reflects post-revisit state.
    context.traj_state.ml_belief[col] = 0.78
    context.traj_state.ml_conflict[col] = 0.18
    context.traj_state.label_source[col] = "llm_revisit"
    record_iteration_metrics(
        context.traj_state, context.traj_columns, 0, context.traj_cfg,
        revisited_this_iter={col},
    )


@then('column "{col}" snapshot at iteration {n:d} has revisited={flag}')
def step_snapshot_revisited(context, col, n, flag):
    state = context.traj_state
    snap = next((s for s in state.column_history[col] if s.iteration == n), None)
    assert snap is not None, f"No snapshot for column {col!r} at iteration {n}"
    expected = flag.lower() == "true"
    assert snap.revisited == expected, (
        f"Column {col!r} snapshot at iteration {n}: revisited={snap.revisited}, expected {expected}"
    )


@given("a BootstrapState with one column whose gap sequence is {g0:g}, {g1:g}, {g2:g}")
def step_state_with_gap_sequence(context, g0, g1, g2):
    from atelier.classify.bootstrap import (
        BootstrapState, BootstrapConfig, record_iteration_metrics,
    )
    state = BootstrapState()
    cfg = BootstrapConfig()
    name = "a"
    state.labels[name] = "ICE.NONSENSITIVE"
    state.label_source[name] = "llm"
    state.ml_plausibility[name] = 0.90
    state.ml_conflict[name] = 0.20
    for i, g in enumerate([float(g0), float(g1), float(g2)]):
        # Set bel so that pl - bel == g.
        state.ml_belief[name] = state.ml_plausibility[name] - g
        state.iteration = i
        record_iteration_metrics(state, [name], 0, cfg, revisited_this_iter=set())
    context.traj_state = state
    context.traj_cfg = cfg
    context.traj_columns = [name]


@then("column_contraction for that column is approximately {expected:g}")
def step_column_contraction_approx(context, expected):
    from atelier.classify.bootstrap import column_contraction
    rho = column_contraction(context.traj_state, context.traj_columns[0])
    assert rho is not None, "column_contraction returned None"
    assert abs(rho - float(expected)) < 0.05, (
        f"column_contraction={rho:.3f} not ≈ {expected}"
    )


@given("a BootstrapState with one column and a single iteration recorded")
def step_state_single_iter(context):
    from atelier.classify.bootstrap import (
        BootstrapState, BootstrapConfig, record_iteration_metrics,
    )
    state = BootstrapState()
    cfg = BootstrapConfig()
    name = "a"
    state.labels[name] = "ICE.NONSENSITIVE"
    state.label_source[name] = "llm"
    state.ml_belief[name] = 0.55
    state.ml_plausibility[name] = 0.85
    state.ml_conflict[name] = 0.40
    state.iteration = 0
    record_iteration_metrics(state, [name], 0, cfg, revisited_this_iter=set())
    context.traj_state = state
    context.traj_cfg = cfg
    context.traj_columns = [name]


@then("column_contraction for that column is None")
def step_column_contraction_none(context):
    from atelier.classify.bootstrap import column_contraction
    rho = column_contraction(context.traj_state, context.traj_columns[0])
    assert rho is None, f"Expected None; got {rho!r}"


# ── Universal vocabulary provenance guard ───────────────────────


@when('I load the "{vocab}" vocabulary fixture')
def step_load_named_vocab(context, vocab):
    import json
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[3]
    paths = {
        "universal": repo_root / "src" / "atelier" / "classify" / "fixtures" / "universal_vocabulary.json",
        "sample": repo_root / "data" / "sample" / "ontology.json",
    }
    if vocab not in paths:
        raise ValueError(f"Unknown vocab fixture: {vocab!r}; expected one of {sorted(paths)}")
    with open(paths[vocab]) as f:
        context.universal_records = json.load(f)
    context.universal_vocab_label = vocab


@when("I load the universal vocabulary fixture")
def step_load_universal_raw(context):
    import json
    from pathlib import Path
    fixtures_dir = (
        Path(__file__).resolve().parents[3]
        / "src" / "atelier" / "classify" / "fixtures"
    )
    path = fixtures_dir / "universal_vocabulary.json"
    with open(path) as f:
        context.universal_records = json.load(f)
    context.universal_vocab_label = "universal"


@then('no abbrev value begins with "{prefix}"')
def step_no_abbrev_prefix(context, prefix):
    offenders = [
        r["code"]
        for r in context.universal_records
        if str(r.get("abbrev", "") or "").startswith(prefix)
    ]
    label = getattr(context, "universal_vocab_label", "vocab")
    assert not offenders, (
        f"[{label}] Found {len(offenders)} entr{'y' if len(offenders) == 1 else 'ies'} "
        f"with abbrev starting '{prefix}': {offenders[:10]}"
        f"{' …' if len(offenders) > 10 else ''}. "
        "See PROVENANCE.md — class-prefix abbrevs are a customer-internal "
        "convention and must not appear in shipped Atelier vocabularies."
    )


@then("the notation field is empty for every entry")
def step_notation_empty(context):
    offenders = [
        (r["code"], r.get("notation"))
        for r in context.universal_records
        if str(r.get("notation", "") or "").strip()
    ]
    label = getattr(context, "universal_vocab_label", "vocab")
    assert not offenders, (
        f"[{label}] Found {len(offenders)} entr{'y' if len(offenders) == 1 else 'ies'} "
        f"with non-empty notation: {offenders[:5]}"
        f"{' …' if len(offenders) > 5 else ''}. "
        "See PROVENANCE.md — numeric notation values are customer-encoded "
        "and must not appear in shipped Atelier vocabularies."
    )


# ── Parent-aware DST frame (Stage 3) ────────────────────────────


def _build_internal_node_frame(context, internal_code: str, leaves: list[str]):
    """Helper: construct a HierarchicalCategorySet plus FrameOfDiscernment
    around one internal node and its leaf descendants.

    Adds a sibling leaf in a sibling subtree (``ICE.NONSENSITIVE``,
    drawn from the publicly-grounded ICE catch-all) so the internal
    node's descendant set is a strict subset of ``leaf_codes`` —
    required for the parent to register as an internal-node focal
    element distinct from Theta.  See
    ``FrameOfDiscernment._build_focal_elements`` and
    ``src/atelier/classify/fixtures/PROVENANCE.md``.
    """
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    from atelier.classify.belief import FrameOfDiscernment

    internal_cat = ReferenceCategory(
        code=internal_code, label=internal_code, embedding_text=internal_code,
    )
    leaf_cats = [
        ReferenceCategory(
            code=leaf, label=leaf, embedding_text=leaf, parent_code=internal_code,
        )
        for leaf in leaves
    ]
    sibling_leaf = ReferenceCategory(
        code="ICE.NONSENSITIVE",
        label="Non-Sensitive Data",
        embedding_text="Non-Sensitive Data",
    )
    cs = HierarchicalCategorySet(
        name="test",
        categories=leaf_cats + [sibling_leaf],
        all_categories=[internal_cat] + leaf_cats + [sibling_leaf],
    )
    context.parent_frame_cs = cs
    context.parent_frame = FrameOfDiscernment(cs)
    context.parent_code = internal_code
    context.leaf_codes = leaves


@given('a hierarchical frame with internal node "{node}" over leaves "{leaf_a}" and "{leaf_b}"')
def step_internal_two_leaves(context, node, leaf_a, leaf_b):
    _build_internal_node_frame(context, node, [leaf_a, leaf_b])


@given('a hierarchical frame with internal node "{node}" over a single leaf "{leaf}"')
def step_internal_single_leaf(context, node, leaf):
    _build_internal_node_frame(context, node, [leaf])


@when('the LLM votes internal node "{code}" at confidence {confidence:g}')
def step_llm_votes_internal_node(context, code, confidence):
    from atelier.classify.mass_functions import llm_to_mass
    context.parent_vote_mass = llm_to_mass(
        category_code=code,
        confidence=float(confidence),
        alternatives=[],
        frame=context.parent_frame,
    )


@then('the resulting mass function carries non-zero mass on the internal-node focal element for "{code}"')
def step_mass_on_internal_node(context, code):
    frame = context.parent_frame
    assert code in frame.internal_nodes, (
        f"{code!r} is not an internal node in the test frame; "
        f"internal_nodes={list(frame.internal_nodes)}"
    )
    fe = frame.internal_nodes[code]
    mass = context.parent_vote_mass.masses.get(fe, 0.0)
    assert mass > 1e-6, (
        f"Expected non-zero mass on internal-node focal element {code!r}; "
        f"got {mass} (masses={ {str(k): v for k, v in context.parent_vote_mass.masses.items()} })"
    )


@then("no mass is allocated to either leaf singleton from that vote")
def step_no_leaf_singleton_mass(context):
    frame = context.parent_frame
    masses = context.parent_vote_mass.masses
    for leaf in context.leaf_codes:
        fe = frame.singletons[leaf]
        m = masses.get(fe, 0.0)
        assert m <= 1e-9, (
            f"Expected zero leaf-singleton mass on {leaf!r}; got {m}"
        )


@then('the resulting mass function carries singleton mass on "{code}"')
def step_mass_on_singleton(context, code):
    frame = context.parent_frame
    assert code in frame.singletons, (
        f"{code!r} is not a singleton in the test frame"
    )
    fe = frame.singletons[code]
    mass = context.parent_vote_mass.masses.get(fe, 0.0)
    assert mass > 1e-6, (
        f"Expected non-zero singleton mass on {code!r}; got {mass}"
    )


# ── Parent-aware headline picker (Stage 4) ───────────────────────


@given('a frame from the loan hierarchy')
def step_loan_frame(context):
    from atelier.classify.belief import FrameOfDiscernment
    context.headline_cs = _build_loan_hierarchy()
    context.headline_frame = FrameOfDiscernment(context.headline_cs)
    context.headline_sources = {}


@given('the LLM votes internal node "{code}" at confidence {conf:g} with discount {discount:g}')
def step_headline_llm_votes_parent(context, code, conf, discount):
    from atelier.classify.mass_functions import llm_to_mass
    context.headline_sources["llm"] = llm_to_mass(
        category_code=code,
        confidence=float(conf),
        alternatives=[],
        frame=context.headline_frame,
        discount=float(discount),
    )


@given('cosine localizes to wrong-subtree leaf "{code}" at mass {mass:g}')
def step_headline_cosine_wrong_subtree(context, code, mass):
    # Build a sharp cosine assignment on the wrong-subtree leaf so the
    # leaf-only argmax would have picked it pre-Stage-4.
    from atelier.classify.belief import BeliefAssignment
    frame = context.headline_frame
    assert code in frame.singletons, f"{code!r} is not a leaf singleton"
    leaf_fe = frame.singletons[code]
    m = float(mass)
    context.headline_sources["cosine"] = BeliefAssignment(
        masses={leaf_fe: m, frame.theta: 1.0 - m}
    )


@given('cosine localizes to in-subtree leaf "{code}" at mass {mass:g}')
def step_headline_cosine_in_subtree(context, code, mass):
    from atelier.classify.belief import BeliefAssignment
    frame = context.headline_frame
    assert code in frame.singletons, f"{code!r} is not a leaf singleton"
    leaf_fe = frame.singletons[code]
    m = float(mass)
    context.headline_sources["cosine"] = BeliefAssignment(
        masses={leaf_fe: m, frame.theta: 1.0 - m}
    )


@given('every source is vacuous')
def step_headline_vacuous(context):
    frame = context.headline_frame
    context.headline_sources = {"llm": frame.vacuous(), "cosine": frame.vacuous()}


@when('I fuse those sources into a HierarchicalClassification')
def step_headline_fuse(context):
    from atelier.classify.belief import HierarchicalClassification
    context.headline_hc = HierarchicalClassification.from_combined_evidence(
        source_masses=context.headline_sources,
        frame=context.headline_frame,
        category_set=context.headline_cs,
    )


@then('the headline predicted_code is the internal node "{code}"')
def step_headline_is_internal(context, code):
    actual = context.headline_hc.category.code
    assert actual == code, (
        f"Expected headline to be internal node {code!r}; got {actual!r} "
        f"({context.headline_hc.category.label})"
    )
    # Confirm it really is treated as an internal node in the frame.
    assert code in context.headline_frame.internal_nodes, (
        f"{code!r} is not an internal node in the frame"
    )


@then('the headline category label is "{label}"')
def step_headline_label(context, label):
    actual = getattr(context.headline_hc.category, "label", None)
    assert actual == label, f"Expected label {label!r}; got {actual!r}"


@then('the headline predicted_code is the leaf "{code}"')
def step_headline_is_leaf(context, code):
    actual = context.headline_hc.category.code
    assert actual == code, f"Expected leaf headline {code!r}; got {actual!r}"
    assert code in context.headline_frame.singletons, (
        f"{code!r} is not a leaf singleton in the frame"
    )


@then('the headline predicted_code is a depth-1-or-deeper code')
def step_headline_depth_floor(context):
    code = context.headline_hc.category.code
    assert code, f"Expected non-empty headline code; got {code!r}"
    depth = code.count(".")
    assert depth >= 1, f"Expected depth ≥ 1; got code {code!r} at depth {depth}"


@then('the headline predicted_code is not the empty string')
def step_headline_nonempty(context):
    code = context.headline_hc.category.code
    assert code, f"Expected non-empty headline code; got {code!r}"
