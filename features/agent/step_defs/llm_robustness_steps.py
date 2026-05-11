# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for LLM robustness scenarios."""

from behave import given, when, then


@given("a mock LLM backend that truncates at {limit:d} columns")
def step_mock_truncating_backend(context, limit):
    from dataclasses import dataclass, field as dc_field
    from atelier.classify.llm_backend import LLMResponse, ColumnClassification

    class TruncatingBackend:
        """Mock backend that truncates responses beyond a column limit."""

        def __init__(self, truncate_at):
            self._truncate_at = truncate_at

        def classify_batch(self, samples, system_prompt, **kwargs):
            # Only return classifications for first N columns
            truncated = len(samples) > self._truncate_at
            classifications = [
                ColumnClassification(
                    column_name=s.name,
                    category_code=f"TEST.{i}",
                    confidence=0.9,
                    evidence="mock",
                    alternatives=[],
                )
                for i, s in enumerate(samples[:self._truncate_at])
            ]
            return LLMResponse(
                classifications=classifications,
                input_tokens=100,
                output_tokens=50,
                model="mock",
                finish_reason="length" if truncated else "stop",
            )

    context.mock_backend = TruncatingBackend(limit)


@when("I classify a batch of {count:d} columns with retry")
def step_classify_with_retry(context, count):
    from atelier.classify.bootstrap import _classify_batch_with_retry, BootstrapState
    from atelier.classify.sampler import ColumnSample

    samples = [
        ColumnSample(
            name=f"col_{i}",
            column_type="varchar",
            values=[f"val_{i}"],
            total_count=10,
            null_count=0,
            table_name="test_table",
            database="test_db",
            siblings=[],
        )
        for i in range(count)
    ]
    state = BootstrapState()
    context.classifications = _classify_batch_with_retry(
        context.mock_backend, samples, "system prompt", state,
    )
    context.state = state


@then("all {count:d} columns receive classifications")
def step_all_columns_classified(context, count):
    classified_names = {c.column_name for c in context.classifications}
    expected = {f"col_{i}" for i in range(count)}
    assert classified_names == expected, (
        f"Missing: {expected - classified_names}, extra: {classified_names - expected}"
    )


@given("a bootstrap state with truncation tracking")
def step_bootstrap_state(context):
    from atelier.classify.bootstrap import BootstrapState
    context.state = BootstrapState()
    assert context.state.truncation_count == 0


@when("a truncated response is recorded")
def step_record_truncation(context):
    context.state.truncation_count += 1


@then("truncation_count increments")
def step_check_truncation_count(context):
    assert context.state.truncation_count > 0


@given('an LLM response with finish_reason "{reason}"')
def step_llm_response(context, reason):
    from atelier.classify.llm_backend import LLMResponse
    context.llm_response = LLMResponse(
        classifications=[],
        input_tokens=0,
        output_tokens=0,
        model="test",
        finish_reason=reason,
    )


@then("truncated is true")
def step_truncated_true(context):
    assert context.llm_response.truncated is True, (
        f"Expected truncated=True for finish_reason={context.llm_response.finish_reason!r}"
    )


@then("truncated is false")
def step_truncated_false(context):
    assert context.llm_response.truncated is False
