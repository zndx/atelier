"""Step definitions for ML classifier mass function BDD scenarios.

Reuses shared steps from classification_steps.py:
- "the mock annotations vocabulary is loaded"
- "a frame of discernment from the vocabulary"
"""

import json

from behave import when, then


@when('I compute CatBoost mass from {proba_json}')
def step_catboost_mass(context, proba_json):
    from atelier.classify.mass_functions import catboost_to_mass
    proba = json.loads(proba_json)
    context.catboost_mass = catboost_to_mass(proba, context.frame)
    context.current_mass = context.catboost_mass


@when('I compute CatBoost mass with variance from {proba_json} and {var_json}')
def step_catboost_mass_with_variance(context, proba_json, var_json):
    from atelier.classify.mass_functions import catboost_to_mass
    proba = json.loads(proba_json)
    variance = json.loads(var_json)
    context.catboost_mass = catboost_to_mass(proba, context.frame, variance)
    context.current_mass = context.catboost_mass


@when("I compute CatBoost mass for empty probabilities")
def step_catboost_empty(context):
    from atelier.classify.mass_functions import catboost_to_mass
    context.catboost_empty = catboost_to_mass({}, context.frame)


@when('I compute SVM mass from {proba_json}')
def step_svm_mass(context, proba_json):
    from atelier.classify.mass_functions import svm_to_mass
    proba = json.loads(proba_json)
    context.svm_mass = svm_to_mass(proba, context.frame)
    context.current_mass = context.svm_mass


@when("I compute SVM mass for empty probabilities")
def step_svm_empty(context):
    from atelier.classify.mass_functions import svm_to_mass
    context.svm_empty = svm_to_mass({}, context.frame)


@then("the ML mass function should not be vacuous")
def step_ml_not_vacuous(context):
    masses = context.current_mass.masses
    assert len(masses) > 1, (
        f"Expected non-vacuous mass (>1 focal element), got {len(masses)}"
    )


@then('the ML top singleton should be "{code}"')
def step_ml_top_singleton(context, code):
    masses = context.current_mass.masses
    best_code = None
    best_mass = -1.0
    for fe, m in masses.items():
        if len(fe.codes) == 1:
            c = next(iter(fe.codes))
            if m > best_mass:
                best_code = c
                best_mass = m
    assert best_code == code, (
        f"Expected top singleton {code}, got {best_code} (mass={best_mass:.4f})"
    )


@then("theta mass should be approximately {expected:f}")
def step_theta_approx(context, expected):
    theta_mass = context.current_mass.masses.get(context.frame.theta, 0.0)
    assert abs(theta_mass - expected) < 0.05, (
        f"Expected theta ~{expected:.2f}, got {theta_mass:.4f}"
    )


@then("theta mass should be greater than {threshold:f}")
def step_theta_greater(context, threshold):
    theta_mass = context.current_mass.masses.get(context.frame.theta, 0.0)
    assert theta_mass > threshold, (
        f"Expected theta > {threshold:.2f}, got {theta_mass:.4f}"
    )


@then("both mass functions should be vacuous")
def step_both_vacuous(context):
    for name, mass in [("CatBoost", context.catboost_empty), ("SVM", context.svm_empty)]:
        assert len(mass.masses) == 1, (
            f"{name} expected vacuous (1 focal element), got {len(mass.masses)}"
        )
        theta_mass = list(mass.masses.values())[0]
        assert abs(theta_mass - 1.0) < 1e-10, (
            f"{name} expected theta=1.0, got {theta_mass}"
        )
