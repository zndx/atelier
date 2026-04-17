"""Step definitions for DST fusion strategy scenarios."""

from behave import given, when, then


def _frame():
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.taxonomy import load_universal_vocabulary
    vocab = load_universal_vocabulary(hierarchical=True)
    return FrameOfDiscernment(vocab)


@given("two non-conflicting mass functions")
def step_non_conflicting(context):
    from atelier.classify.belief import BeliefAssignment
    frame = _frame()
    leaves = list(frame.singletons.keys())
    fe_a = frame.singleton(leaves[0])
    context.frame = frame
    context.m1 = BeliefAssignment(masses={fe_a: 0.6, frame.theta: 0.4})
    context.m2 = BeliefAssignment(masses={fe_a: 0.5, frame.theta: 0.5})


@given("two conflicting mass functions with K > 0.5")
def step_conflicting(context):
    from atelier.classify.belief import BeliefAssignment
    frame = _frame()
    leaves = list(frame.singletons.keys())
    fe_a = frame.singleton(leaves[0])
    fe_b = frame.singleton(leaves[1])
    context.frame = frame
    context.fe_a = fe_a
    context.fe_b = fe_b
    context.m1 = BeliefAssignment(masses={fe_a: 0.8, frame.theta: 0.2})
    context.m2 = BeliefAssignment(masses={fe_b: 0.7, frame.theta: 0.3})


@given("three mass functions")
def step_three_masses(context):
    from atelier.classify.belief import BeliefAssignment
    frame = _frame()
    leaves = list(frame.singletons.keys())
    fe_a = frame.singleton(leaves[0])
    fe_b = frame.singleton(leaves[1])
    context.frame = frame
    context.assignments = [
        BeliefAssignment(masses={fe_a: 0.6, frame.theta: 0.4}),
        BeliefAssignment(masses={fe_b: 0.5, frame.theta: 0.5}),
        BeliefAssignment(masses={fe_a: 0.3, fe_b: 0.3, frame.theta: 0.4}),
    ]


@when("I combine them with Dempster's rule")
def step_dempster(context):
    from atelier.classify.belief import dempster_combine
    context.dempster_result, context.dempster_k = dempster_combine(
        context.m1, context.m2
    )


@when("I combine them with Yager's rule")
def step_yager(context):
    from atelier.classify.belief import yager_combine
    context.yager_result, context.yager_k = yager_combine(
        context.m1, context.m2, context.frame.theta
    )


@when('I call combine_multiple with strategy "yager" and no theta')
def step_yager_no_theta(context):
    from atelier.classify.belief import combine_multiple
    try:
        combine_multiple(context.assignments, strategy="yager", theta=None)
        context.raised = None
    except ValueError as e:
        context.raised = e


@then("the results have identical mass distributions")
def step_identical(context):
    d = sorted(context.dempster_result.masses.values())
    y = sorted(context.yager_result.masses.values())
    assert len(d) == len(y), f"Different mass count: {len(d)} vs {len(y)}"
    for a, b in zip(d, y):
        assert abs(a - b) < 1e-9, f"Mass differs: {a} vs {b}"


@then("the Θ mass in the result is at least the conflict value")
def step_theta_absorbs(context):
    theta = context.frame.theta
    theta_mass = context.yager_result.masses.get(theta, 0.0)
    assert theta_mass >= context.yager_k - 1e-9, (
        f"Θ mass {theta_mass} < conflict {context.yager_k}"
    )


@then("no singleton mass exceeds its raw conjunctive product")
def step_no_inflation(context):
    # m1(a)*m2(Θ) = 0.8 * 0.3 = 0.24 is the max raw product for fe_a
    # Yager should not inflate beyond this
    fe_a_mass = context.yager_result.masses.get(context.fe_a, 0.0)
    assert fe_a_mass <= 0.24 + 1e-9, (
        f"Yager inflated fe_a: {fe_a_mass} > 0.24"
    )


@then("the singleton masses sum to more than their raw products")
def step_dempster_inflates(context):
    # Dempster normalizes by (1-K), so surviving singletons get bigger
    theta = context.frame.theta
    non_theta = [m for fe, m in context.dempster_result.masses.items()
                 if fe != theta]
    raw_products_sum = 0.8 * 0.3 + 0.7 * 0.2  # fe_a + fe_b raw
    assert sum(non_theta) > raw_products_sum, (
        f"Expected inflation: {sum(non_theta)} should exceed {raw_products_sum}"
    )


@then("the total mass sums to 1.0")
def step_sums_to_one(context):
    total = sum(context.dempster_result.masses.values())
    assert abs(total - 1.0) < 1e-9, f"Total mass {total} != 1.0"


@then("a ValueError is raised")
def step_raised_value_error(context):
    assert context.raised is not None, "Expected ValueError but none raised"
    assert isinstance(context.raised, ValueError)
