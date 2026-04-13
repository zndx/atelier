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

    Returns ground truth labels from mock fixtures with high confidence.
    """

    def __init__(self, ground_truth: dict[str, str]):
        self._gt = ground_truth

    def classify_batch(self, samples, system_prompt, revisit_context=None, table_name=None):
        from atelier.classify.llm_backend import ColumnClassification, LLMResponse
        classifications = []
        for sample in samples:
            code = self._gt.get(sample.name)
            classifications.append(ColumnClassification(
                column_name=sample.name,
                category_code=code,
                confidence=0.9 if code else 0.0,
                evidence="mock ground truth",
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
    from atelier.classify.sampler import load_all_mock_samples

    # Collect ground truth from mock fixtures
    gt = {}
    for ts in load_all_mock_samples():
        for col in ts.columns:
            if col.ground_truth:
                gt[col.name] = col.ground_truth

    cfg = load_config()
    fsm = AgentFSM()
    mock_backend = _MockLLMBackend(gt)

    context.bootstrap_result = run_classification_pipeline(
        cfg, fsm, use_mock=True, llm_backend=mock_backend,
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
    from atelier.classify.sampler import load_all_mock_samples

    gt = {}
    for ts in load_all_mock_samples():
        for col in ts.columns:
            if col.ground_truth:
                gt[col.name] = col.ground_truth

    cfg = load_config()
    fsm = AgentFSM()
    # Lower accuracy to force disagreements and trigger revisit loop
    mock_backend = RealisticMockLLMBackend(
        gt, base_accuracy=0.55, seed=42,
    )

    context.bootstrap_result = run_classification_pipeline(
        cfg, fsm, use_mock=True, llm_backend=mock_backend,
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
