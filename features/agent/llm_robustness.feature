@agent @tier-0
Feature: LLM classification robustness
  The bootstrap LLM sweep detects truncated responses and retries with
  smaller batch sizes.  Truncation metrics are tracked and exposed to
  the convergence agent.

  Scenario: Truncated response triggers retry with smaller batch
    Given a mock LLM backend that truncates at 5 columns
    When I classify a batch of 10 columns with retry
    Then all 10 columns receive classifications

  Scenario: Truncation metrics are tracked in state
    Given a bootstrap state with truncation tracking
    When a truncated response is recorded
    Then truncation_count increments

  Scenario: LLM response truncated property covers all backends
    Given an LLM response with finish_reason "length"
    Then truncated is true
    Given an LLM response with finish_reason "max_tokens"
    Then truncated is true
    Given an LLM response with finish_reason "end_turn"
    Then truncated is false
