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


@given('a fictitious vocabulary with abbrev "{abbrev}" at code "{code}"')
def step_fictitious_vocab(context, abbrev, code):
    """Build a tiny test vocabulary with a fictitious code namespace.

    Codes are deliberately drawn from a fictitious ``acme.*`` namespace
    so the BDD source carries no customer-derived encoding (provenance
    audit 2026-04-30; see fixtures/PROVENANCE.md).
    """
    from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
    cats = [
        ReferenceCategory(
            code=code, label="Test Term", embedding_text="", abbrev=abbrev,
        ),
        ReferenceCategory(
            code="acme.misc", label="Other", embedding_text="", abbrev="OTHER",
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
