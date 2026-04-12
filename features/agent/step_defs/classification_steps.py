"""Step definitions for Dempster-Shafer classification BDD scenarios."""

from behave import given, when, then


# ── Vocabulary ───────────────────────────────────────────────────────


@given("the mock annotations vocabulary is loaded")
def step_load_mock_vocab(context):
    from atelier.classify.taxonomy import load_mock_annotations
    context.category_set = load_mock_annotations(hierarchical=True)
    assert len(context.category_set.categories) > 0


@given("a frame of discernment from the vocabulary")
def step_build_frame(context):
    from atelier.classify.belief import FrameOfDiscernment
    if not hasattr(context, "category_set"):
        from atelier.classify.taxonomy import load_mock_annotations
        context.category_set = load_mock_annotations(hierarchical=True)
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


@then('the detected patterns should include "{pattern}"')
def step_check_detected_pattern(context, pattern):
    assert pattern in context.detected_patterns, (
        f"{pattern} not in {context.detected_patterns}"
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


# ── Pipeline ─────────────────────────────────────────────────────────


@when("I run the classification pipeline with mock data")
def step_run_pipeline(context):
    from atelier.config import load_config
    from atelier.classify import run_pipeline
    cfg = load_config()
    context.pipeline_result = run_pipeline(cfg, use_mock=True)


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


@then("the accuracy against ground truth should exceed {threshold:g}")
def step_accuracy_threshold(context, threshold):
    acc = context.pipeline_result.get("accuracy")
    if acc is None:
        assert False, "No accuracy computed (no ground truth?)"
    assert acc > float(threshold), f"Accuracy {acc} <= {threshold}"


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
