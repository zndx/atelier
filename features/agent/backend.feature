@tier-0
Feature: LLM backend factory
  As a developer, I need the backend factory to create the correct
  backend class for each configured provider.

  Scenario: Factory creates OpenAICompatibleBackend for cerebras
    When I create a backend with config backend="cerebras" api_key="test"
    Then the backend should be an OpenAICompatibleBackend
    And the backend base_url should be "https://api.cerebras.ai/v1"
    And the backend model should be "zai-glm-4.7"

  Scenario: Factory creates BedrockBackend for bedrock
    When I create a backend with config backend="bedrock" aws_access_key_id="test" aws_secret_access_key="test"
    Then the backend should be a BedrockBackend

  Scenario: Factory creates BedrockStructuredBackend for bedrock_structured
    When I create a backend with config backend="bedrock_structured" aws_access_key_id="test" aws_secret_access_key="test"
    Then the backend should be a BedrockStructuredBackend

  Scenario: Factory creates AnthropicStructuredBackend for anthropic_structured
    When I create a backend with config backend="anthropic_structured" api_key="test"
    Then the backend should be an AnthropicStructuredBackend

  Scenario: Factory rejects unknown backend
    When I attempt to create a backend with config backend="unknown"
    Then the factory should raise ValueError

  Scenario: Subagent model with Bedrock format selects BedrockStructuredBackend
    Given an AtelierConfig with Bedrock credentials and subagent model "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    When I call create_backend_from_cfg
    Then the result should be a BedrockStructuredBackend

  Scenario: Subagent model with plain Anthropic format selects AnthropicStructuredBackend
    Given an AtelierConfig with ANTHROPIC_API_KEY and subagent model "claude-sonnet-4-5-20250929"
    When I call create_backend_from_cfg
    Then the result should be an AnthropicStructuredBackend

  Scenario: No LLM configuration raises ValueError
    Given an AtelierConfig with only Bedrock credentials
    When I attempt to call create_backend_from_cfg
    Then the factory should raise ValueError
