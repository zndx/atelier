@infra @preflight
Feature: Preflight validation

  Operators run preflight before starting services to catch
  misconfigurations early. Deny = blocking, warn = advisory.

  @tier-0
  Scenario: Preflight passes with valid config
    When I run preflight checks
    Then the result has no denies

  @tier-0
  Scenario: Missing config file is a deny
    When I run preflight with a nonexistent config path
    Then the result has a deny for "config_exists"

  @tier-0
  Scenario: Missing credentials is a warning not a deny
    Given config with no LLM credentials
    When I run preflight checks against that config
    Then the result has a warning for "credentials_configured"
    And the result has no denies
