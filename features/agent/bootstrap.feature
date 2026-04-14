@tier-0
Feature: LLM bootstrap convergence loop
  As a data steward, I need the bootstrap pipeline to iterate LLM
  and ML classification until DST conflict converges, validating
  column labels across independent evidence sources.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: LLM mass function converts prediction to belief assignment
    Given a frame of discernment from the vocabulary
    When I compute llm_to_mass for code "ICE.SENSITIVE.PID.CONTACT.EMAIL" with confidence 0.9
    Then the LLM mass should assign mass greater than 0.8 to "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And the LLM mass should assign less than 0.2 to theta

  Scenario: LLM mass function handles alternatives
    Given a frame of discernment from the vocabulary
    When I compute llm_to_mass for "ICE.SENSITIVE.PID.CONTACT.EMAIL" with confidence 0.7 and alternative "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN" at 0.2
    Then the LLM mass should have "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN" as a focal element
    And the total LLM mass should sum to 1.0

  Scenario: LLM mass function returns vacuous for unknown code
    Given a frame of discernment from the vocabulary
    When I compute llm_to_mass for code "99.99.99" with confidence 0.9
    Then the LLM mass should be vacuous

  @slow
  Scenario: Bootstrap convergence loop reaches convergence with mock LLM
    When I run the bootstrap pipeline with mock data and mock LLM
    Then the bootstrap pipeline should reach CONVERGED state
    And the bootstrap result should report total LLM calls greater than 0

  @slow
  Scenario: Bootstrap converges with realistic LLM disagreement
    When I run the bootstrap pipeline with mock data and realistic mock LLM
    Then the bootstrap pipeline should reach CONVERGED state
    And the bootstrap should have iterated more than 0 times
    And the final accuracy should exceed 0.40

  @slow
  Scenario: Revisit phase reduces disagreements
    When I run the bootstrap pipeline with mock data and realistic mock LLM
    Then the bootstrap pipeline should reach CONVERGED state
    And the number of disagreements should decrease across iterations

  @slow
  Scenario: Mean K decreases across bootstrap iterations
    When I run the bootstrap pipeline with mock data and realistic mock LLM
    Then the bootstrap pipeline should reach CONVERGED state
    And the k_convergence_rate should be negative or zero

  Scenario: K convergence tracker detects plateau
    Given a bootstrap state with metrics showing K plateau
    Then should_stop_early should return true

  Scenario: K convergence tracker allows decreasing K
    Given a bootstrap state with metrics showing K decrease
    Then should_stop_early should return false

  Scenario: FSM transitions through bootstrap states
    Given a fresh AgentFSM
    When I start a new run
    And I advance to "LOADING_VOCAB"
    And I advance to "DISCOVERING"
    And I advance to "SAMPLING"
    And I advance to "LLM_SWEEP"
    Then the state should be "LLM_SWEEP"
    When I advance to "VALIDATING"
    Then the state should be "VALIDATING"
    When I advance to "LLM_SWEEP"
    Then the state should be "LLM_SWEEP"
    When I advance to "VALIDATING"
    And I advance to "CLASSIFYING"
    Then the state should be "CLASSIFYING"
