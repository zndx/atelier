# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for LLM backend factory BDD scenarios."""

from behave import given, when, then


@when('I create a backend with config backend="cerebras" api_key="test"')
def step_create_cerebras(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(backend="cerebras", api_key="test")
    context.backend = create_backend(config)


@when('I create a backend with config backend="bedrock" aws_access_key_id="test" aws_secret_access_key="test"')
def step_create_bedrock(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(
        backend="bedrock",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    context.backend = create_backend(config)


@when('I create a backend with config backend="bedrock_structured" aws_access_key_id="test" aws_secret_access_key="test"')
def step_create_bedrock_structured(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(
        backend="bedrock_structured",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    context.backend = create_backend(config)


@when('I create a backend with config backend="anthropic_structured" api_key="test"')
def step_create_anthropic_structured(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(backend="anthropic_structured", api_key="test")
    context.backend = create_backend(config)


@when('I attempt to create a backend with config backend="unknown"')
def step_create_unknown(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(backend="unknown")
    context.factory_error = None
    try:
        create_backend(config)
    except ValueError as e:
        context.factory_error = e


# ── Subagent model scenarios ──────────────────────────────────────


@given('an AtelierConfig with Bedrock credentials and subagent model "{model}"')
def step_config_bedrock_subagent(context, model):
    from atelier.config import AtelierConfig
    context.test_cfg = AtelierConfig(
        classify_subagent_model=model,
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )


@given('an AtelierConfig with ANTHROPIC_API_KEY and subagent model "{model}"')
def step_config_anthropic_subagent(context, model):
    from atelier.config import AtelierConfig
    context.test_cfg = AtelierConfig(
        classify_subagent_model=model,
        anthropic_api_key="sk-ant-test-key",
    )


@given("an AtelierConfig with only Bedrock credentials")
def step_config_bedrock_only(context):
    from atelier.config import AtelierConfig
    context.test_cfg = AtelierConfig(
        aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
        aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    )


@when("I call create_backend_from_cfg")
def step_call_factory(context):
    from atelier.classify.llm_backend import create_backend_from_cfg
    context.backend = create_backend_from_cfg(context.test_cfg)


@when("I attempt to call create_backend_from_cfg")
def step_attempt_call_factory(context):
    from atelier.classify.llm_backend import create_backend_from_cfg
    context.factory_error = None
    try:
        create_backend_from_cfg(context.test_cfg)
    except ValueError as e:
        context.factory_error = e


# ── Type assertions ──────────────────────────────────────────────


@then("the backend should be an OpenAICompatibleBackend")
def step_check_openai_type(context):
    from atelier.classify.llm_backend import OpenAICompatibleBackend
    assert isinstance(context.backend, OpenAICompatibleBackend)


@then('the backend base_url should be "{url}"')
def step_check_base_url(context, url):
    assert context.backend._config.base_url == url, (
        f"Expected {url}, got {context.backend._config.base_url}"
    )


@then('the backend model should be "{model}"')
def step_check_model(context, model):
    assert context.backend._config.model == model, (
        f"Expected {model}, got {context.backend._config.model}"
    )


@then("the backend should be a BedrockBackend")
def step_check_bedrock_type(context):
    from atelier.classify.llm_backend import BedrockBackend
    assert isinstance(context.backend, BedrockBackend)


@then("the backend should be a BedrockStructuredBackend")
def step_check_bedrock_structured_type(context):
    from atelier.classify.llm_backend import BedrockStructuredBackend
    assert isinstance(context.backend, BedrockStructuredBackend)


@then("the backend should be an AnthropicStructuredBackend")
def step_check_anthropic_structured_type(context):
    from atelier.classify.llm_backend import AnthropicStructuredBackend
    assert isinstance(context.backend, AnthropicStructuredBackend)


@then("the result should be a BedrockStructuredBackend")
def step_result_bedrock_structured(context):
    from atelier.classify.llm_backend import BedrockStructuredBackend
    assert isinstance(context.backend, BedrockStructuredBackend), (
        f"Expected BedrockStructuredBackend, got {type(context.backend).__name__}"
    )


@then("the result should be an AnthropicStructuredBackend")
def step_result_anthropic_structured(context):
    from atelier.classify.llm_backend import AnthropicStructuredBackend
    assert isinstance(context.backend, AnthropicStructuredBackend), (
        f"Expected AnthropicStructuredBackend, got {type(context.backend).__name__}"
    )


@then("the factory should raise ValueError")
def step_check_factory_error(context):
    assert context.factory_error is not None, "Expected ValueError but none was raised"
    assert isinstance(context.factory_error, ValueError)
