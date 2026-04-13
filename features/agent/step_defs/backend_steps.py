"""Step definitions for LLM backend factory BDD scenarios."""

from behave import when, then


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


@when('I attempt to create a backend with config backend="unknown"')
def step_create_unknown(context):
    from atelier.classify.llm_backend import LLMBackendConfig, create_backend
    config = LLMBackendConfig(backend="unknown")
    context.factory_error = None
    try:
        create_backend(config)
    except ValueError as e:
        context.factory_error = e


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


@then("the factory should raise ValueError")
def step_check_factory_error(context):
    assert context.factory_error is not None, "Expected ValueError but none was raised"
    assert isinstance(context.factory_error, ValueError)
