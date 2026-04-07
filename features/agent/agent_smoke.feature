@agent
Feature: Agent SDK smoke test
  Verify the Claude Agent SDK integration is wired end-to-end.

  @tier-0
  Scenario: Keystone agents are seeded
    Given the database is bootstrapped
    Then at least 3 agents are registered

  @tier-0
  Scenario: Bedrock credentials detected when AWS keys present
    Given config with overrides
      | field                 | value          |
      | aws_access_key_id     | AKIAEXAMPLE    |
      | aws_secret_access_key | secret         |
    Then the config has_bedrock is true
    And the config has_anthropic is false

  @tier-0
  Scenario: Anthropic credentials detected when API key present
    Given config with overrides
      | field              | value      |
      | anthropic_api_key  | sk-ant-xxx |
    Then the config has_anthropic is true
    And the config has_bedrock is false

  @tier-0
  Scenario: Both providers coexist
    Given config with overrides
      | field                 | value          |
      | anthropic_api_key     | sk-ant-xxx     |
      | aws_access_key_id     | AKIAEXAMPLE    |
      | aws_secret_access_key | secret         |
    Then the config has_anthropic is true
    And the config has_bedrock is true

  @tier-1
  Scenario: Credentials are valid
    When I validate all credentials
    Then at least one provider is valid

  @tier-1
  Scenario: SDK smoke test completes
    When I run the SDK smoke test
    Then the result contains a session_id
