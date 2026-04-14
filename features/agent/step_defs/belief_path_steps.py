"""Step definitions for belief path BDD scenarios."""

from behave import given, when, then


@given('evidence strongly supporting "{code}"')
def step_strong_evidence(context, code):
    from atelier.classify.belief import BeliefAssignment
    frame = context.frame
    fe = frame.singleton(code)
    context.bp_evidence = {"strong": BeliefAssignment({fe: 0.8, frame.theta: 0.2})}


@given('weak evidence supporting "{code}"')
def step_weak_evidence(context, code):
    from atelier.classify.belief import BeliefAssignment
    frame = context.frame
    fe = frame.singleton(code)
    context.bp_evidence = {"weak": BeliefAssignment({fe: 0.3, frame.theta: 0.7})}


@when("I classify using belief path evidence")
def step_build_bp_classification(context):
    from atelier.classify.belief import HierarchicalClassification
    context.hc = HierarchicalClassification.from_combined_evidence(
        source_masses=context.bp_evidence,
        frame=context.frame,
        category_set=context.category_set,
    )
    context.belief_path = context.hc.belief_path()


@then('the belief path should start at "{code}"')
def step_path_starts(context, code):
    assert len(context.belief_path) > 0, "Belief path is empty"
    assert context.belief_path[0]["code"] == code, (
        f"Expected path to start at {code}, got {context.belief_path[0]['code']}"
    )


@then('the belief path should end at "{code}"')
def step_path_ends(context, code):
    assert len(context.belief_path) > 0, "Belief path is empty"
    assert context.belief_path[-1]["code"] == code, (
        f"Expected path to end at {code}, got {context.belief_path[-1]['code']}"
    )


@then("belief should increase or stay the same ascending the path")
def step_bel_monotone(context):
    path = context.belief_path
    for i in range(len(path) - 1):
        assert path[i]["bel"] <= path[i + 1]["bel"] + 0.001, (
            f"Bel not monotone: depth {path[i]['depth']} ({path[i]['bel']}) > "
            f"depth {path[i+1]['depth']} ({path[i+1]['bel']})"
        )


@then("plausibility should increase or stay the same ascending the path")
def step_pl_monotone(context):
    path = context.belief_path
    for i in range(len(path) - 1):
        assert path[i]["pl"] <= path[i + 1]["pl"] + 0.001, (
            f"Pl not monotone: depth {path[i]['depth']} ({path[i]['pl']}) > "
            f"depth {path[i+1]['depth']} ({path[i+1]['pl']})"
        )


@then('the cautious code at threshold {tau} should be an ancestor of "{code}"')
def step_cautious_ancestor(context, tau, code):
    tau = float(tau)
    cautious = context.hc.cautious_code(tau)
    ancestors = context.category_set.ancestors(code)
    assert cautious == code or cautious in ancestors, (
        f"Cautious code {cautious} is not {code} or an ancestor of it"
    )


@then("belief at the cautious code should exceed {tau}")
def step_cautious_bel(context, tau):
    tau = float(tau)
    cautious = context.hc.cautious_code(tau)
    bel = context.hc.belief_at(cautious)
    assert bel >= tau - 0.001, (
        f"Belief at cautious code {cautious} is {bel}, expected >= {tau}"
    )


@given("classification results with belief paths")
def step_mock_classifications_with_paths(context):
    """Build mock classification results with belief paths for epistemic eval."""
    from atelier.classify.belief import (
        BeliefAssignment, FrameOfDiscernment, HierarchicalClassification,
    )
    frame = FrameOfDiscernment(context.category_set)

    codes = list(context.category_set.leaf_codes)[:5]
    context.eval_classifications = []

    for code in codes:
        fe = frame.singleton(code)
        m = BeliefAssignment({fe: 0.75, frame.theta: 0.25})
        hc = HierarchicalClassification.from_combined_evidence(
            source_masses={"test": m},
            frame=frame,
            category_set=context.category_set,
        )
        bp = hc.belief_path()
        context.eval_classifications.append({
            "predicted_code": code,
            "ground_truth": code,
            "confidence": hc.confidence,
            "belief": hc.belief_at(code),
            "plausibility": hc.plausibility_at(code),
            "belief_path": bp,
        })


@when("I run epistemic_evaluation")
def step_run_epistemic(context):
    from atelier.classify.evaluation import epistemic_evaluation
    context.epistemic_result = epistemic_evaluation(
        context.eval_classifications,
        category_set=context.category_set,
    )


@then("per-depth metrics should include mean_bel and mean_pl and mean_gap")
def step_check_per_depth(context):
    per_depth = context.epistemic_result["per_depth"]
    assert len(per_depth) > 0, "per_depth is empty"
    for depth, metrics in per_depth.items():
        assert "mean_bel" in metrics, f"Missing mean_bel at depth {depth}"
        assert "mean_pl" in metrics, f"Missing mean_pl at depth {depth}"
        assert "mean_gap" in metrics, f"Missing mean_gap at depth {depth}"


@then("cautious_accuracy should be a valid fraction")
def step_check_cautious_accuracy(context):
    ca = context.epistemic_result["cautious_accuracy"]
    assert 0.0 <= ca <= 1.0, f"cautious_accuracy {ca} not in [0, 1]"


@then("mean_commitment_depth should be a positive number")
def step_check_commitment_depth(context):
    mcd = context.epistemic_result["mean_commitment_depth"]
    assert mcd >= 0, f"mean_commitment_depth {mcd} is negative"
