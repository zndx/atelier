"""Step definitions for agent-driven convergence loop BDD scenarios.

Tool handler tests are pure Python — no SDK or API key needed.
The mock Anthropic client returns pre-built tool_use blocks that
exercise the tool dispatch logic without making real API calls.
"""

from behave import given, when, then


def _ensure_frame(context):
    """Lazily build FrameOfDiscernment from category_set if not already set."""
    if not hasattr(context, "frame"):
        from atelier.classify.belief import FrameOfDiscernment
        context.frame = FrameOfDiscernment(context.category_set)
    return context.frame


# ── Mock Anthropic client ────────────────────────────────────────


class _MockContentBlock:
    """Mimics an anthropic ContentBlock (text or tool_use)."""

    def __init__(self, block_type, **kwargs):
        self.type = block_type
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockUsage:
    def __init__(self, input_tokens=50, output_tokens=25):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _MockResponse:
    """Mimics an anthropic Messages response."""

    def __init__(self, content, stop_reason="end_turn"):
        self.content = content
        self.stop_reason = stop_reason
        self.usage = _MockUsage()


class _MockAnthropicClient:
    """Mock client that follows a scripted sequence of tool calls.

    Each turn returns a pre-built response with text reasoning and/or
    tool_use blocks. The sequence defines the agent's "strategy".
    """

    def __init__(self, turns):
        self._turns = list(turns)
        self._call_count = 0

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        if self._call_count >= len(self._turns):
            # Final turn: just declare converged
            return _MockResponse(
                content=[
                    _MockContentBlock("text", text="No more scripted turns."),
                ],
                stop_reason="end_turn",
            )
        resp = self._turns[self._call_count]
        self._call_count += 1
        return resp


def _make_tool_use_block(tool_name, tool_input, tool_id=None):
    """Create a mock tool_use content block."""
    return _MockContentBlock(
        "tool_use",
        name=tool_name,
        input=tool_input,
        id=tool_id or f"mock_{tool_name}_{id(tool_input)}",
    )


# ── Scripted mock client factories ───────────────────────────────


def _client_inspect_then_converge():
    """Mock client that: check_convergence → declare_converged."""
    return _MockAnthropicClient([
        # Turn 0: Check convergence first
        _MockResponse(
            content=[
                _MockContentBlock("text", text="Let me check current metrics."),
                _make_tool_use_block("check_convergence", {}, "t0_check"),
            ],
            stop_reason="tool_use",
        ),
        # Turn 1: Declare converged
        _MockResponse(
            content=[
                _MockContentBlock("text", text="Metrics look good, declaring converged."),
                _make_tool_use_block(
                    "declare_converged",
                    {"reason": "Mean K below threshold after inspection"},
                    "t1_converge",
                ),
            ],
            stop_reason="tool_use",
        ),
    ])


def _client_revisit_then_converge(high_k_columns):
    """Mock client that: get_conflict_report → revisit_columns → check → declare."""
    return _MockAnthropicClient([
        # Turn 0: Get conflict report
        _MockResponse(
            content=[
                _MockContentBlock("text", text="Checking conflict report."),
                _make_tool_use_block(
                    "get_conflict_report", {"k_threshold": 0.2}, "t0_conflict",
                ),
            ],
            stop_reason="tool_use",
        ),
        # Turn 1: Revisit high-K columns
        _MockResponse(
            content=[
                _MockContentBlock("text", text="Revisiting columns with high conflict."),
                _make_tool_use_block(
                    "revisit_columns",
                    {"column_names": high_k_columns[:5]},
                    "t1_revisit",
                ),
            ],
            stop_reason="tool_use",
        ),
        # Turn 2: Check convergence
        _MockResponse(
            content=[
                _MockContentBlock("text", text="Checking metrics after revisit."),
                _make_tool_use_block("check_convergence", {}, "t2_check"),
            ],
            stop_reason="tool_use",
        ),
        # Turn 3: Declare converged
        _MockResponse(
            content=[
                _MockContentBlock("text", text="K improved, declaring converged."),
                _make_tool_use_block(
                    "declare_converged",
                    {"reason": "K decreased after targeted revisit"},
                    "t3_converge",
                ),
            ],
            stop_reason="tool_use",
        ),
    ])


# ── Setup helpers ────────────────────────────────────────────────


def _build_test_state(category_set, frame, samples_by_name, column_names):
    """Run initial LLM sweep + ML validation to populate a BootstrapState."""
    from atelier.classify.bootstrap import (
        BootstrapState,
        BootstrapConfig,
        _run_ml_validation,
    )

    state = BootstrapState()
    boot_cfg = BootstrapConfig()

    # Set labels from ground truth (simulating LLM sweep)
    for name in column_names:
        col = samples_by_name[name]
        if col.ground_truth:
            state.labels[name] = col.ground_truth
            state.confidence[name] = 0.9
            state.label_source[name] = "llm"

    # Run ML validation to get conflict K values
    _run_ml_validation(
        state, boot_cfg, column_names, samples_by_name,
        category_set, frame, has_embeddings=True,
    )

    return state, boot_cfg


# ── Given steps ──────────────────────────────────────────────────


@given("a bootstrap state with {count:d} high-conflict columns")
def step_state_with_high_k(context, count):
    from atelier.classify.bootstrap import BootstrapState

    state = BootstrapState()
    cols = [f"test_col_{i}" for i in range(count)]

    for col in cols:
        state.labels[col] = "ICE.SENSITIVE.PII.EMAIL"
        state.confidence[col] = 0.7
        state.ml_prediction[col] = "ICE.SENSITIVE.PII.USERNAME"
        state.ml_conflict[col] = 0.35 + 0.05 * cols.index(col)
        state.ml_confidence[col] = 0.6

    context.agent_state = state
    context.agent_columns = cols


@given("a bootstrap state with iteration history")
def step_state_with_history(context):
    from atelier.classify.bootstrap import BootstrapState, IterationMetrics

    state = BootstrapState()
    state.labels = {"col_a": "ICE.SENSITIVE.PII.EMAIL"}
    state.ml_conflict = {"col_a": 0.15}
    state.iteration_metrics = [
        IterationMetrics(iteration=0, mean_k=0.4, max_k=0.6, disagreements=5, coverage=0.9, llm_calls=1),
        IterationMetrics(iteration=1, mean_k=0.3, max_k=0.5, disagreements=3, coverage=0.95, llm_calls=2),
    ]
    context.agent_state = state
    context.agent_columns = ["col_a"]


@given("fixture samples with a bootstrap state")
def step_fixture_with_state(context):
    from atelier.classify.sampler import load_fixture_samples

    samples = load_fixture_samples()
    category_set = context.category_set
    frame = _ensure_frame(context)

    samples_by_name = {}
    column_names = []
    for ts in samples:
        for col in ts.columns:
            samples_by_name[col.name] = col
            column_names.append(col.name)

    state, boot_cfg = _build_test_state(
        category_set, frame, samples_by_name, column_names,
    )

    context.agent_state = state
    context.agent_boot_cfg = boot_cfg
    context.agent_columns = column_names
    context.agent_samples = samples_by_name


@given("a fresh bootstrap state")
def step_fresh_state(context):
    from atelier.classify.bootstrap import BootstrapState
    context.agent_state = BootstrapState()


@given("fixture samples with the mock LLM backend")
def step_fixture_with_mock_llm(context):
    """Set up fixture samples and mock LLM backend (same as bootstrap_steps)."""
    from atelier.classify.sampler import load_fixture_samples

    samples = load_fixture_samples()
    gt = {}
    samples_by_name = {}
    column_names = []
    column_table = {}

    for ts in samples:
        for col in ts.columns:
            samples_by_name[col.name] = col
            column_names.append(col.name)
            column_table[col.name] = col.table_name or "__flat__"
            if col.ground_truth:
                gt[col.name] = col.ground_truth

    from features.agent.step_defs.bootstrap_steps import _MockLLMBackend
    context.agent_mock_backend = _MockLLMBackend(gt)
    context.agent_samples = samples_by_name
    context.agent_columns = column_names
    context.agent_column_table = column_table


@given("fixture samples with the realistic mock LLM backend")
def step_fixture_with_realistic_mock(context):
    """Set up fixture samples with a realistic (error-prone) mock backend."""
    from atelier.classify.sampler import load_fixture_samples
    from atelier.classify.mock_llm import RealisticMockLLMBackend

    samples = load_fixture_samples()
    gt = {}
    samples_by_name = {}
    column_names = []
    column_table = {}

    for ts in samples:
        for col in ts.columns:
            samples_by_name[col.name] = col
            column_names.append(col.name)
            column_table[col.name] = col.table_name or "__flat__"
            if col.ground_truth:
                gt[col.name] = col.ground_truth

    context.agent_mock_backend = RealisticMockLLMBackend(
        gt, base_accuracy=0.55, seed=42,
    )
    context.agent_samples = samples_by_name
    context.agent_columns = column_names
    context.agent_column_table = column_table


@given("initial ML validation is complete")
def step_initial_ml_validation(context):
    """Run LLM sweep + ML validation to populate state before agent loop."""
    from atelier.classify.bootstrap import (
        BootstrapState,
        BootstrapConfig,
        _llm_sweep,
        _run_ml_validation,
        _identify_disagreements,
        _mean_k,
        record_iteration_metrics,
    )

    category_set = context.category_set
    frame = _ensure_frame(context)

    state = BootstrapState()
    boot_cfg = BootstrapConfig()

    column_table = getattr(context, "agent_column_table", {})

    # Phase 1: LLM sweep
    _llm_sweep(
        state, boot_cfg, context.agent_mock_backend,
        "Classify these columns.",
        context.agent_columns, context.agent_samples, column_table,
    )

    # Phase 2: ML validation
    _run_ml_validation(
        state, boot_cfg, context.agent_columns, context.agent_samples,
        category_set, frame, has_embeddings=True,
    )

    disagreements = _identify_disagreements(state, context.agent_columns, boot_cfg)
    record_iteration_metrics(state, context.agent_columns, len(disagreements))

    context.agent_state = state
    context.agent_boot_cfg = boot_cfg

    # Track high-K columns for the revisit mock client
    context.agent_high_k_columns = [
        n for n in context.agent_columns
        if state.ml_conflict.get(n, 0) > boot_cfg.k_threshold
    ]


@given("a mock Anthropic client that inspects then declares converged")
def step_mock_client_inspect(context):
    context.agent_mock_client = _client_inspect_then_converge()


@given("a mock Anthropic client that revisits high-K columns then converges")
def step_mock_client_revisit(context):
    high_k = getattr(context, "agent_high_k_columns", ["col_0"])
    context.agent_mock_client = _client_revisit_then_converge(high_k)


# ── When steps ───────────────────────────────────────────────────


@when("I call get_conflict_report with threshold {threshold:g}")
def step_call_conflict_report(context, threshold):
    from atelier.classify.agent_loop import _handle_get_conflict_report

    context.agent_result = _handle_get_conflict_report(
        context.agent_state,
        context.agent_columns,
        context.category_set,
        k_threshold=float(threshold),
    )


@when("I call check_convergence")
def step_call_check_convergence(context):
    from atelier.classify.agent_loop import _handle_check_convergence
    from atelier.classify.bootstrap import BootstrapConfig

    boot_cfg = getattr(context, "agent_boot_cfg", BootstrapConfig())
    context.agent_result = _handle_check_convergence(
        context.agent_state,
        context.agent_columns,
        boot_cfg,
    )


@when("I call get_column_detail for a labeled column")
def step_call_column_detail(context):
    from atelier.classify.agent_loop import _handle_get_column_detail
    from atelier.classify.bootstrap import BootstrapConfig

    boot_cfg = getattr(context, "agent_boot_cfg", BootstrapConfig())
    frame = _ensure_frame(context)
    # Pick first labeled column
    col_name = next(iter(context.agent_state.labels), None)
    assert col_name, "No labeled columns in state"

    context.agent_result = _handle_get_column_detail(
        context.agent_state,
        col_name,
        context.agent_samples,
        context.category_set,
        frame,
        boot_cfg,
    )


@when('I call declare_converged with reason "{reason}"')
def step_call_declare_converged(context, reason):
    from atelier.classify.agent_loop import _handle_declare_converged

    context.agent_result = _handle_declare_converged(
        context.agent_state,
        reason,
    )


@when("I run the agent loop with the mock client")
def step_run_agent_loop(context):
    from atelier.classify.agent_loop import run_agent_loop
    from atelier.classify.bootstrap import BootstrapConfig
    from atelier.config import load_config

    cfg = load_config()
    boot_cfg = getattr(context, "agent_boot_cfg", BootstrapConfig())
    category_set = context.category_set
    frame = _ensure_frame(context)
    samples = getattr(context, "agent_samples", {})
    column_table = getattr(context, "agent_column_table", {})
    backend = getattr(context, "agent_mock_backend", None)
    state = getattr(context, "agent_state", None)

    if state is None:
        from atelier.classify.bootstrap import BootstrapState
        state = BootstrapState()
        context.agent_state = state

    # Populate state if not already done
    if not state.labels and samples:
        from atelier.classify.bootstrap import (
            _llm_sweep,
            _run_ml_validation,
            _identify_disagreements,
            record_iteration_metrics,
        )
        if backend:
            _llm_sweep(
                state, boot_cfg, backend,
                "Classify these columns.",
                context.agent_columns, samples, column_table,
            )
            _run_ml_validation(
                state, boot_cfg, context.agent_columns, samples,
                category_set, frame, has_embeddings=True,
            )
            disagreements = _identify_disagreements(
                state, context.agent_columns, boot_cfg,
            )
            record_iteration_metrics(
                state, context.agent_columns, len(disagreements),
            )

    events = []
    context.agent_converged = run_agent_loop(
        state, cfg, boot_cfg, backend,
        "Classify these columns.",
        context.agent_columns, samples, column_table,
        category_set, frame,
        has_embeddings=True,
        on_event=lambda e: events.append(e),
        client=context.agent_mock_client,
    )
    context.agent_events = events


# ── Then steps ───────────────────────────────────────────────────


@then("the conflict report should list {count:d} columns")
def step_conflict_report_count(context, count):
    result = context.agent_result
    assert result["high_k_count"] == count, (
        f"Expected {count} high-K columns, got {result['high_k_count']}"
    )


@then("each conflict column should have K greater than {threshold:g}")
def step_conflict_k_threshold(context, threshold):
    for col in context.agent_result["columns"]:
        assert col["conflict_K"] > float(threshold), (
            f"Column {col['column_name']} K={col['conflict_K']} <= {threshold}"
        )


@then("the convergence report should include {field}")
def step_convergence_field(context, field):
    assert field in context.agent_result, (
        f"Field '{field}' not in convergence report: {list(context.agent_result.keys())}"
    )


@then("the column detail should include {field}")
def step_column_detail_field(context, field):
    assert field in context.agent_result, (
        f"Field '{field}' not in column detail: {list(context.agent_result.keys())}"
    )


@then('the state agent_converged_reason should be "{reason}"')
def step_state_reason(context, reason):
    assert context.agent_state.agent_converged_reason == reason, (
        f"Expected '{reason}', got '{context.agent_state.agent_converged_reason}'"
    )


@then('the state agent_reasoning should contain "{text}"')
def step_state_reasoning_contains(context, text):
    all_reasoning = " ".join(context.agent_state.agent_reasoning)
    assert text in all_reasoning, (
        f"'{text}' not found in agent reasoning: {context.agent_state.agent_reasoning}"
    )


@then("the agent loop should complete within {max_turns:d} turns")
def step_agent_loop_turns(context, max_turns):
    assert context.agent_state.agent_turns <= max_turns, (
        f"Agent took {context.agent_state.agent_turns} turns, expected <= {max_turns}"
    )


@then("the state agent_converged_reason should not be empty")
def step_state_reason_not_empty(context):
    assert context.agent_state.agent_converged_reason, (
        "agent_converged_reason is empty"
    )


@then("the state agent_turns should be greater than {n:d}")
def step_agent_turns_gt(context, n):
    assert context.agent_state.agent_turns > n, (
        f"agent_turns={context.agent_state.agent_turns}, expected > {n}"
    )


@then("the state agent_reasoning should contain at least {n:d} entries")
def step_agent_reasoning_entries(context, n):
    count = len(context.agent_state.agent_reasoning)
    assert count >= n, (
        f"agent_reasoning has {count} entries, expected >= {n}"
    )
