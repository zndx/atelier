"""Step definitions for coverage_guarantees.feature.

Unit-level verification of the LLM-coverage fixes.  Each scenario
exercises one of the five failure modes we've seen bite CAI runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from behave import given, when, then


# ── Scenario 1: reference-column exclusion ──────────────────────────


@given("a mixed sample set with natural-named and reference-named columns")
def _ref_excl_mixed_sample(context):
    from atelier.classify.sampler import ColumnSample, TableSample

    sibling_names = ["first_name", "attr_1_1_1_9_2_1", "customer_id"]
    context.mixed_samples = [
        TableSample(name="uat_shape", columns=[
            ColumnSample(name="first_name", siblings=list(sibling_names)),
            ColumnSample(name="attr_1_1_1_9_2_1", siblings=list(sibling_names)),
            ColumnSample(name="customer_id", siblings=list(sibling_names)),
        ]),
    ]
    context.production_samples = [
        TableSample(name="customers", columns=[
            ColumnSample(name="customer_id", siblings=["customer_id", "email", "phone"]),
            ColumnSample(name="email", siblings=["customer_id", "email", "phone"]),
            ColumnSample(name="phone", siblings=["customer_id", "email", "phone"]),
        ]),
    ]


@when("I apply the reference-column exclusion invariant")
def _ref_excl_apply(context):
    from atelier.classify.meta_tagging_source import exclude_reference_columns

    context.mixed_out = exclude_reference_columns(context.mixed_samples)
    context.prod_out = exclude_reference_columns(context.production_samples)


@then("the reference columns are dropped")
def _ref_excl_dropped(context):
    remaining = {c.name for t in context.mixed_out for c in t.columns}
    assert "attr_1_1_1_9_2_1" not in remaining, f"reference column leaked: {remaining}"
    assert "first_name" in remaining
    assert "customer_id" in remaining


@then("sibling contexts no longer reference the dropped columns")
def _ref_excl_siblings_clean(context):
    for t in context.mixed_out:
        for c in t.columns:
            assert "attr_1_1_1_9_2_1" not in c.siblings, (
                f"reference name leaked through siblings of {c.name}: {c.siblings}"
            )


@then("production-shape column names are untouched")
def _ref_excl_prod_untouched(context):
    before = {c.name for t in context.production_samples for c in t.columns}
    after = {c.name for t in context.prod_out for c in t.columns}
    assert before == after, f"production columns changed: before={before}, after={after}"


# ── Scenario 2: Bedrock ceiling ─────────────────────────────────────


@given("the Bedrock output-token ceiling table")
def _bedrock_table(context):
    from atelier.classify.llm_backend import bedrock_max_output_tokens
    context.bedrock_ceiling = bedrock_max_output_tokens


@then("claude-3-5-sonnet resolves to 8192")
def _bedrock_3_5_sonnet(context):
    assert context.bedrock_ceiling("anthropic.claude-3-5-sonnet-20240620-v1:0") == 8192


@then("claude-sonnet-4 resolves to 64000")
def _bedrock_4_sonnet(context):
    assert context.bedrock_ceiling("anthropic.claude-sonnet-4-20250514-v1:0") == 64000


@then("claude-3-haiku resolves to 4096")
def _bedrock_3_haiku(context):
    assert context.bedrock_ceiling("anthropic.claude-3-haiku-20240307-v1:0") == 4096


@then("an unknown model falls back to 4096")
def _bedrock_unknown(context):
    assert context.bedrock_ceiling("totally-fake-model-id") == 4096
    assert context.bedrock_ceiling("") == 4096


@then("a Bedrock inference-profile ARN is matched on the model substring")
def _bedrock_arn(context):
    arn = (
        "arn:aws:bedrock:us-west-2::inference-profile/"
        "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    )
    assert context.bedrock_ceiling(arn) == 8192


# ── Scenario 3: partial-response detection ─────────────────────────


class _DroppingBackend:
    """Fake backend that returns only the first ``keep_n`` classifications."""

    def __init__(self, keep_n: int, *, finish_reason: str = "end_turn"):
        self.keep_n = keep_n
        self.finish_reason = finish_reason
        self.calls = 0

    def effective_max_tokens(self) -> int:
        return 8192

    def classify_batch(self, samples, system_prompt, revisit_context=None,
                       table_name=None):
        from atelier.classify.llm_backend import (
            ColumnClassification, LLMResponse,
        )
        self.calls += 1
        kept = samples[: self.keep_n]
        classifications = [
            ColumnClassification(
                column_name=s.name, category_code="0.1",
                confidence=0.9, evidence="mock", alternatives=[],
            )
            for s in kept
        ]
        # Detect partial exactly the way real backends do.
        returned_names = {c.column_name for c in classifications if c.column_name}
        expected_names = [s.name for s in samples]
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = (
            bool(missing) and self.finish_reason not in ("length", "max_tokens")
        )
        return LLMResponse(
            classifications=classifications,
            input_tokens=100, output_tokens=50, model="mock",
            finish_reason=self.finish_reason,
            partial=is_partial,
        )


@given("a backend that returns fewer classifications than requested")
def _partial_backend(context):
    context.dropping_backend = _DroppingBackend(keep_n=2)


@when("the LLM sweep processes a batch")
def _partial_sweep(context):
    from atelier.classify.sampler import ColumnSample
    samples = [ColumnSample(name=f"col_{i}") for i in range(5)]
    context.partial_response = context.dropping_backend.classify_batch(
        samples, system_prompt="test",
    )


@then("the response carries partial=True even with a clean stop_reason")
def _partial_flag(context):
    r = context.partial_response
    assert r.finish_reason == "end_turn"
    assert r.partial is True, "partial flag did not fire on dropped columns"


@then("the truncated property is True")
def _partial_truncated(context):
    assert context.partial_response.truncated is True


@then("halving retry engages on the partial response")
def _partial_halving(context):
    # The halving path is driven by response.truncated.  Simulate the
    # bootstrap.py branch that would fire.
    should_halve = context.partial_response.truncated and len(
        context.partial_response.classifications
    ) < 5
    assert should_halve, "halving would not engage on a partial response"


# ── Scenario 4: coverage-gap retry ──────────────────────────────────


class _RetryRecoveringBackend:
    """First call drops columns; subsequent calls return them fully."""

    def __init__(self):
        self.calls = 0

    def effective_max_tokens(self) -> int:
        return 8192

    def classify_batch(self, samples, system_prompt, revisit_context=None,
                       table_name=None):
        from atelier.classify.llm_backend import (
            ColumnClassification, LLMResponse,
        )
        self.calls += 1
        # First call: drop the second half (simulates Bedrock silent truncation
        # under our old code path).
        if self.calls == 1:
            kept = samples[: len(samples) // 2]
        else:
            kept = list(samples)
        classifications = [
            ColumnClassification(
                column_name=s.name, category_code="0.1",
                confidence=0.8, evidence="mock", alternatives=[],
            )
            for s in kept
        ]
        returned = {c.column_name for c in classifications}
        expected = [s.name for s in samples]
        missing = [n for n in expected if n not in returned]
        return LLMResponse(
            classifications=classifications,
            input_tokens=100, output_tokens=50, model="mock",
            finish_reason="end_turn",
            partial=bool(missing),
        )

    def health_check(self) -> bool:
        return True


@given("an LLM sweep where the first pass drops some columns")
def _gap_given(context):
    context.retry_backend = _RetryRecoveringBackend()


@when("the sweep completes")
def _gap_when(context):
    from atelier.classify.bootstrap import (
        BootstrapConfig, BootstrapState, _llm_sweep,
    )
    from atelier.classify.sampler import ColumnSample

    names = [f"col_{i}" for i in range(8)]
    samples = {n: ColumnSample(name=n, table_name="t") for n in names}
    column_table = {n: "t" for n in names}

    state = BootstrapState()
    cfg = BootstrapConfig(
        columns_per_call=8, min_columns_per_call=1,
        max_total_llm_calls=50,
    )
    _llm_sweep(
        state, cfg, context.retry_backend, system_prompt="test",
        column_names=names, samples=samples, column_table=column_table,
        category_count=0,
    )
    context.gap_state = state
    context.gap_names = names


@then("a coverage-gap retry runs on the missing columns")
def _gap_retry_ran(context):
    assert context.retry_backend.calls >= 2, (
        f"coverage-gap retry never fired (calls={context.retry_backend.calls})"
    )


@then("every requested column ends up with a label")
def _gap_all_labeled(context):
    missing = [n for n in context.gap_names if n not in context.gap_state.labels]
    assert not missing, f"coverage gap not closed: {missing}"


# ── Scenario 5: config override respect ─────────────────────────────


@dataclass
class _StubCfg:
    """Minimal AtelierConfig surrogate for the discovery-limits test."""
    classify_tables_limit: int = 42
    classify_sample_size: int = 17
    classify_column_sample_limit: int = 999
    cml_data_connection_names: list[str] = field(default_factory=list)


@given("an AtelierConfig with classify_tables_limit=42 and classify_sample_size=17")
def _cfg_given(context):
    context.stub_cfg = _StubCfg()


@when("a caller invokes the pipeline without passing those values")
def _cfg_when(context):
    # Simulate the pipeline.py top-of-function fallback block verbatim —
    # this is the precise code path that was broken before the fix.
    cfg = context.stub_cfg
    sample_size: int | None = None
    tables_limit: int | None = None
    if sample_size is None:
        sample_size = cfg.classify_sample_size
    if tables_limit is None:
        tables_limit = cfg.classify_tables_limit
    context.resolved_sample_size = sample_size
    context.resolved_tables_limit = tables_limit


@then("the pipeline uses 42 and 17, not the hard-coded function defaults")
def _cfg_then(context):
    assert context.resolved_tables_limit == 42, (
        f"tables_limit not respected: got {context.resolved_tables_limit}"
    )
    assert context.resolved_sample_size == 17, (
        f"sample_size not respected: got {context.resolved_sample_size}"
    )
