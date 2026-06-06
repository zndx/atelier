"""Step definitions for bootstrap convergence loop BDD scenarios."""

from behave import given, when, then


# ── LLM mass function ───────────────────────────────────────────


@when('I compute llm_to_mass for code "{code}" with confidence {conf:g}')
def step_llm_to_mass(context, code, conf):
    from atelier.classify.mass_functions import llm_to_mass
    context.llm_mass = llm_to_mass(
        category_code=code,
        confidence=float(conf),
        alternatives=[],
        frame=context.frame,
    )


@when('I compute llm_to_mass for "{code}" with confidence {conf:g} and alternative "{alt_code}" at {alt_conf:g}')
def step_llm_to_mass_with_alt(context, code, conf, alt_code, alt_conf):
    from atelier.classify.mass_functions import llm_to_mass
    context.llm_mass = llm_to_mass(
        category_code=code,
        confidence=float(conf),
        alternatives=[{"code": alt_code, "confidence": float(alt_conf)}],
        frame=context.frame,
    )


@then('the LLM mass should assign mass greater than {threshold:g} to "{code}"')
def step_check_llm_mass(context, threshold, code):
    singleton = context.frame.singleton(code)
    mass = context.llm_mass.masses.get(singleton, 0.0)
    assert mass > float(threshold), f"Mass for {code} = {mass}, expected > {threshold}"


@then('the LLM mass should assign less than {threshold:g} to theta')
def step_check_llm_theta_lt(context, threshold):
    theta_mass = context.llm_mass.masses.get(context.frame.theta, 0.0)
    assert theta_mass < float(threshold), (
        f"Theta mass = {theta_mass}, expected < {threshold}"
    )


@then('the LLM mass should have "{code}" as a focal element')
def step_check_llm_alt(context, code):
    singleton = context.frame.singleton(code)
    assert singleton in context.llm_mass.masses, (
        f"{code} not in mass function focal elements"
    )


@then("the total LLM mass should sum to 1.0")
def step_check_llm_sum(context):
    total = sum(context.llm_mass.masses.values())
    assert abs(total - 1.0) < 1e-9, f"Total mass = {total}, expected 1.0"


@then("the LLM mass should be vacuous")
def step_check_llm_vacuous(context):
    masses = context.llm_mass.masses
    assert len(masses) == 1, f"Expected 1 focal element (Theta), got {len(masses)}"
    fe = next(iter(masses))
    assert len(fe.codes) > 1, "Expected Theta (multi-code focal element)"
    assert abs(masses[fe] - 1.0) < 1e-9, "Expected mass = 1.0 on Theta"


# ── Test-only mock LLM backend ──────────────────────────────────


class _MockLLMBackend:
    """Deterministic LLM backend for tier-0 BDD tests.

    Returns curated reference labels from mock fixtures with high confidence.
    """

    def __init__(self, reference_labels: dict[str, str]):
        self._reference = reference_labels

    def classify_batch(self, samples, system_prompt, revisit_context=None, table_name=None):
        from atelier.classify.llm_backend import ColumnClassification, LLMResponse
        classifications = []
        for sample in samples:
            code = self._reference.get(sample.name)
            classifications.append(ColumnClassification(
                column_name=sample.name,
                category_code=code,
                confidence=0.9 if code else 0.0,
                evidence="mock curated reference",
                alternatives=[],
            ))
        return LLMResponse(
            classifications=classifications,
            input_tokens=100,
            output_tokens=50,
            model="mock",
        )

    def health_check(self):
        return True


# ── Bootstrap pipeline ───────────────────────────────────────────


@when("I run the bootstrap pipeline with mock data and mock LLM")
def step_run_bootstrap(context):
    from atelier.config import load_config
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.sampler import load_fixture_samples

    # Collect curated reference labels from fixtures
    samples = load_fixture_samples()
    reference_labels = {}
    for ts in samples:
        for col in ts.columns:
            if col.reference_code:
                reference_labels[col.name] = col.reference_code

    cfg = load_config()
    fsm = AgentFSM()
    mock_backend = _MockLLMBackend(reference_labels)

    context.bootstrap_result = run_classification_pipeline(
        cfg, fsm, samples=samples, llm_backend=mock_backend,
    )


@then("the bootstrap pipeline should reach CONVERGED state")
def step_bootstrap_converged(context):
    state = context.bootstrap_result.get("state")
    assert state == "CONVERGED", (
        f"State is {state}, error: {context.bootstrap_result.get('error')}"
    )


@then("the bootstrap result should report total LLM calls greater than {n:d}")
def step_bootstrap_llm_calls(context, n):
    calls = context.bootstrap_result.get("llm_calls", 0)
    assert calls > n, f"LLM calls = {calls}, expected > {n}"


# ── Realistic mock LLM ────────────────────────────────────────────


@when("I run the bootstrap pipeline with mock data and realistic mock LLM")
def step_run_bootstrap_realistic(context):
    from atelier.config import load_config
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.mock_llm import RealisticMockLLMBackend
    from atelier.classify.sampler import load_fixture_samples

    samples = load_fixture_samples()
    reference_labels = {}
    for ts in samples:
        for col in ts.columns:
            if col.reference_code:
                reference_labels[col.name] = col.reference_code

    cfg = load_config()
    fsm = AgentFSM()
    # Lower accuracy to force disagreements and trigger revisit loop
    mock_backend = RealisticMockLLMBackend(
        reference_labels, base_accuracy=0.55, seed=42,
    )

    context.bootstrap_result = run_classification_pipeline(
        cfg, fsm, samples=samples, llm_backend=mock_backend,
    )


@then("the bootstrap should have iterated more than {n:d} times")
def step_bootstrap_iterations(context, n):
    iters = context.bootstrap_result.get("bootstrap_iterations", 0)
    assert iters > n, (
        f"Bootstrap iterations = {iters}, expected > {n}"
    )


@then("the final accuracy should exceed {threshold:g}")
def step_bootstrap_accuracy(context, threshold):
    acc = context.bootstrap_result.get("accuracy")
    assert acc is not None, "No accuracy in bootstrap result"
    assert acc > float(threshold), (
        f"Bootstrap accuracy {acc:.4f} <= {threshold}"
    )


@then("the number of disagreements should decrease across iterations")
def step_disagreements_decrease(context):
    metrics = context.bootstrap_result.get("iteration_metrics", [])
    assert len(metrics) >= 2, (
        f"Need >= 2 iteration metrics, got {len(metrics)}"
    )
    first_disagreements = metrics[0]["disagreements"]
    last_disagreements = metrics[-1]["disagreements"]
    assert last_disagreements < first_disagreements, (
        f"Disagreements did not decrease: first={first_disagreements}, "
        f"last={last_disagreements}"
    )


@then("the k_convergence_rate should be negative or zero")
def step_k_convergence_rate(context):
    rate = context.bootstrap_result.get("k_convergence_rate", 1.0)
    assert rate <= 0.001, (
        f"k_convergence_rate = {rate}, expected <= 0 (K should decrease over iterations)"
    )


# ── K convergence unit scenarios ─────────────────────────────────


@given("a bootstrap state with metrics showing K plateau")
def step_k_plateau(context):
    from atelier.classify.bootstrap import BootstrapState, IterationMetrics
    state = BootstrapState()
    state.iteration_metrics = [
        IterationMetrics(iteration=0, mean_k=0.3, max_k=0.5, disagreements=10, coverage=0.95, llm_calls=1),
        IterationMetrics(iteration=1, mean_k=0.3, max_k=0.5, disagreements=10, coverage=0.95, llm_calls=2),
        IterationMetrics(iteration=2, mean_k=0.3, max_k=0.5, disagreements=10, coverage=0.95, llm_calls=3),
    ]
    context.k_state = state


@then("should_stop_early should return true")
def step_should_stop(context):
    from atelier.classify.bootstrap import should_stop_early
    assert should_stop_early(context.k_state), "Expected should_stop_early to return True"


@given("a bootstrap state with metrics showing K decrease")
def step_k_decrease(context):
    from atelier.classify.bootstrap import BootstrapState, IterationMetrics
    state = BootstrapState()
    state.iteration_metrics = [
        IterationMetrics(iteration=0, mean_k=0.5, max_k=0.8, disagreements=20, coverage=0.90, llm_calls=1),
        IterationMetrics(iteration=1, mean_k=0.4, max_k=0.6, disagreements=15, coverage=0.92, llm_calls=2),
        IterationMetrics(iteration=2, mean_k=0.3, max_k=0.5, disagreements=10, coverage=0.95, llm_calls=3),
    ]
    context.k_state = state


@then("should_stop_early should return false")
def step_should_not_stop(context):
    from atelier.classify.bootstrap import should_stop_early
    assert not should_stop_early(context.k_state), "Expected should_stop_early to return False"


# Note: the M9 in-loop SVM-on-LLM-labels retrain
# (``train_svm_on_frontier_labels``) was excised in commit 5199379 for
# Denoeux 2008 source-independence reasons.  The BDD steps that tested
# that retrain were removed in P6 (delete-excised-scenarios), since no
# producer of the asserted result-dict keys remains.  Going-forward,
# SVM is trained via the procedural-ML stack (``scripts/train_svm_on_synth.py``);
# tests for that path land under ``features/agent/svm_on_synth.feature``.


# ── Indep-tier disagreement gate ────────────────────────────────


@given('a BootstrapState with LLM "{llm_code}" and indep-tier consensus "{indep_code}" at mass {mass:g}')
def step_state_with_indep(context, llm_code, indep_code, mass):
    from atelier.classify.bootstrap import BootstrapState
    state = BootstrapState()
    state.labels["col"] = llm_code
    state.independent_top1["col"] = indep_code
    state.independent_top1_mass["col"] = float(mass)
    context.bootstrap_state = state


@given('a BootstrapState with LLM "{llm_code}" and no indep-tier consensus')
def step_state_no_indep(context, llm_code):
    from atelier.classify.bootstrap import BootstrapState
    state = BootstrapState()
    state.labels["col"] = llm_code
    context.bootstrap_state = state


@given('the fused ml_prediction also equals "{ml_code}" with conflict K={k:g}')
def step_state_fused_match(context, ml_code, k):
    context.bootstrap_state.ml_prediction["col"] = ml_code
    context.bootstrap_state.ml_conflict["col"] = float(k)


@given('the fused ml_prediction equals "{ml_code}" with conflict K={k:g}')
def step_state_fused_diff(context, ml_code, k):
    context.bootstrap_state.ml_prediction["col"] = ml_code
    context.bootstrap_state.ml_conflict["col"] = float(k)


@when("I call _identify_disagreements with k_threshold {k:g} and indep_revisit_mass_threshold {indep:g}")
def step_call_identify(context, k, indep):
    from atelier.classify.bootstrap import BootstrapConfig, _identify_disagreements
    cfg = BootstrapConfig(k_threshold=float(k), indep_revisit_mass_threshold=float(indep))
    context.disagreements = _identify_disagreements(
        context.bootstrap_state, ["col"], cfg,
    )


@then("the column should appear in the disagreements list")
def step_in_disagreements(context):
    assert "col" in context.disagreements, (
        f"Expected 'col' in disagreements, got {context.disagreements}"
    )


@then("the column should not appear in the disagreements list")
def step_not_in_disagreements(context):
    assert "col" not in context.disagreements, (
        f"Did not expect 'col' in disagreements, got {context.disagreements}"
    )
