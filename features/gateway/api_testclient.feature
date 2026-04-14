@gateway @tier-0
Feature: Gateway API shape validation (TestClient)
  Validate API response schemas without the full stack. The TestClient
  runs the FastAPI app in-process — gRPC calls fail gracefully (the
  gateway catches them and returns JSON error envelopes). These tests
  validate the gateway code itself: routing, error handling, response
  shapes. Tier-1 tests verify populated data.

  Scenario: Skills endpoint returns well-formed envelope
    When I GET /api/skills from a test client
    Then the test client response should be valid JSON
    And the test client JSON should contain "skills"
    And the test client JSON "skills" should be a list

  Scenario: Vocabulary stats returns expected keys
    When I GET /api/vocabulary/stats from a test client
    Then the test client response should be valid JSON
    And the test client JSON should contain "terms"

  Scenario: FSM status returns valid state envelope
    When I GET /api/fsm/status from a test client
    Then the test client response should be valid JSON
    And the test client JSON should contain "state"

  Scenario: Agents endpoint returns graceful response
    When I GET /api/agents from a test client
    Then the test client response should be valid JSON

  Scenario: Data sources endpoint returns graceful response
    When I GET /api/data-sources from a test client
    Then the test client response should be valid JSON

  Scenario: Health endpoint returns graceful response
    When I GET /api/health from a test client
    Then the test client response should be valid JSON

  Scenario: Datasets endpoint returns graceful response
    When I GET /api/datasets from a test client
    Then the test client response should be valid JSON
