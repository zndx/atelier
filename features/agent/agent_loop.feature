@tier-0 @gpu
Feature: Agent-driven classification convergence
  The agent loop wraps bootstrap pipeline functions as tools,
  letting Claude reason about which columns to revisit.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: get_conflict_report returns well-formed data
    Given a bootstrap state with 5 high-conflict columns
    When I call get_conflict_report with threshold 0.2
    Then the conflict report should list 5 columns
    And each conflict column should have K greater than 0.2

  Scenario: check_convergence returns metrics
    Given a bootstrap state with iteration history
    When I call check_convergence
    Then the convergence report should include mean_k
    And the convergence report should include coverage
    And the convergence report should include iteration_history

  Scenario: get_column_detail returns evidence breakdown
    Given fixture samples with a bootstrap state
    When I call get_column_detail for a labeled column
    Then the column detail should include evidence_sources
    And the column detail should include sample_values

  Scenario: declare_converged records reason in state
    Given a fresh bootstrap state
    When I call declare_converged with reason "Belief gap below threshold"
    Then the state agent_converged_reason should be "Belief gap below threshold"
    And the state agent_reasoning should contain "CONVERGED"

  Scenario: Agent loop with mock client reaches convergence
    Given fixture samples with the mock LLM backend
    And a mock Anthropic client that inspects then declares converged
    When I run the agent loop with the mock client
    Then the agent loop should complete within 5 turns
    And the state agent_converged_reason should not be empty

  @slow
  Scenario: Agent revisit reduces conflict on mock data
    Given fixture samples with the realistic mock LLM backend
    And initial ML validation is complete
    And a mock Anthropic client that revisits high-K columns then converges
    When I run the agent loop with the mock client
    Then the state agent_turns should be greater than 1
    And the state agent_reasoning should contain at least 2 entries
