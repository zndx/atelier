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

  Scenario: Factory rejects unknown backend
    When I attempt to create a backend with config backend="unknown"
    Then the factory should raise ValueError
