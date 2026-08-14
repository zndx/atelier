@gateway @tier-0
Feature: Web Terminal Agent model catalog + selection
  Operators pick which frontier model the embedded Claude Agent SDK
  terminal session routes through. Selection is gateway-lifetime —
  set once, sticks until the process restarts.

  Background:
    Given ANTHROPIC_API_KEY is set so Anthropic rows are available
    And the terminal model override is cleared

  Scenario: Catalog exposes two Opus-class provider rows plus the gateway row
    When I GET the terminal models catalog
    Then the catalog has 3 entries
    And the catalog includes a "bedrock" provider row
    And the catalog includes an "anthropic" provider row
    And at least one "anthropic" row is available
    And every catalog entry is an Opus-class model

  Scenario: Selecting an available model sets the override
    When I POST terminal model id "anthropic-opus-4-7"
    Then the selection response is ok
    And the active terminal model id is "anthropic-opus-4-7"
    And the terminal catalog reports override_set true

  Scenario: Selecting an unavailable model is rejected
    Given Bedrock credentials are absent
    When I POST terminal model id "bedrock-opus-4-6"
    Then the selection response reports not available

  Scenario: Selecting an unknown model id is rejected
    When I POST terminal model id "not-a-real-id"
    Then the selection response reports unknown id

  Scenario: Clearing override falls back to cfg.agent_model
    When I POST terminal model id "anthropic-opus-4-7"
    And I DELETE the terminal model override
    Then the terminal catalog reports override_set false

  Scenario: Catalog stats are present and start at n=0
    When I GET the terminal models catalog
    Then every row has a stats object with n >= 0
