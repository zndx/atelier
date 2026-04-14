@tier-0 @gpu
Feature: Dempster-Shafer classification pipeline
  As a data steward, I need the classification pipeline to correctly
  identify column types from metadata so that I can tag production
  tables with the controlled vocabulary.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: Belief assignment roundtrip
    Given a frame of discernment from the vocabulary
    When I create a belief assignment with mass 0.6 on "ICE.SENSITIVE.PID.CONTACT.EMAIL" and 0.4 on theta
    Then the belief for "ICE.SENSITIVE.PID.CONTACT.EMAIL" should be approximately 0.6
    And the plausibility for "ICE.SENSITIVE.PID.CONTACT.EMAIL" should be approximately 1.0
    And the uncertainty for "ICE.SENSITIVE.PID.CONTACT.EMAIL" should be approximately 0.4

  Scenario: Dempster combination reduces uncertainty
    Given a frame of discernment from the vocabulary
    And two independent evidence sources both supporting "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    When I combine them via Dempster's rule
    Then the combined belief for "ICE.SENSITIVE.PID.CONTACT.EMAIL" should exceed either source alone
    And the conflict K should be less than 0.5

  Scenario: Feature extraction produces 12 fields
    When I extract features for column "customer_email" of type "VARCHAR" with email values
    Then all 12 feature names should be present
    And the pattern signals should include "email_pattern"
    And the embedding text should contain "customer email"

  Scenario: Pattern detection fires on known patterns
    When I run pattern detection on SSN values "123-45-6789, 234-56-7890, 345-67-8901"
    Then the detected patterns should include "ssn_pattern"
    When I run pattern detection on credit card values "4111111111111111, 5500000000000004"
    Then the detected patterns should include "credit_card_pattern"

  Scenario: Name matching finds exact and abbreviation matches
    Given a frame of discernment from the vocabulary
    When I run name matching for column "email address"
    Then the name match mass function should not be vacuous
    And the top singleton should be "ICE.SENSITIVE.PID.CONTACT.EMAIL"

  @slow
  Scenario: Pipeline end-to-end with mock data produces results
    When I run the classification pipeline with mock data
    Then the pipeline should reach CONVERGED state
    And the results should contain at least 40 classified columns
    And the accuracy against ground truth should exceed 0.6
    And the micro-F1 should exceed 0.55

  Scenario: FSM state transitions are valid
    Given a fresh AgentFSM
    When I start a new run
    Then the state should be "IDLE"
    When I advance to "LOADING_VOCAB"
    Then the state should be "LOADING_VOCAB"
    When I advance to "DISCOVERING"
    Then the state should be "DISCOVERING"

  Scenario: Pignistic probability distributes Theta mass fairly
    Given a frame of discernment from the vocabulary
    When I create a belief assignment with mass 0.6 on "ICE.SENSITIVE.PID.CONTACT.EMAIL" and 0.4 on theta
    Then the pignistic probability for "ICE.SENSITIVE.PID.CONTACT.EMAIL" should exceed 0.6

  Scenario: HierarchicalClassification navigates belief at parent level
    Given a frame of discernment from the vocabulary
    And two independent evidence sources both supporting "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    When I build a HierarchicalClassification from combined evidence
    Then belief at leaf "ICE.SENSITIVE.PID.CONTACT.EMAIL" should be positive
    And belief at parent "ICE.SENSITIVE.PID.CONTACT" should be at least as high as at "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And the classification should report whether clarification is needed

  Scenario: Mock annotations map ontology to label and annotation to formal code
    Then category "ICE.SENSITIVE.PID.CONTACT.EMAIL" label should be "Email Address"
    And category "ICE.SENSITIVE.PID.CONTACT.EMAIL" abbrev should be "EMAIL"
    And category "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN" label should be "Payment Card Number"
    And category "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN" abbrev should be "PAN"

  @slow
  Scenario: Structured evaluation produces per-category metrics
    When I run the classification pipeline with mock data
    Then the pipeline should reach CONVERGED state
    And the evaluation report should contain per-category metrics
    And every category with support > 0 should have precision and recall
    And the evaluation report should contain a confusion matrix

  Scenario: Pattern map produces non-vacuous mass for email pattern
    Given a frame of discernment from the vocabulary
    When I compute pattern_to_mass for signals ["email_pattern"]
    Then the pattern mass should assign weight to "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And the pattern mass should not be vacuous

  Scenario: FSM rejects invalid transitions
    Given a fresh AgentFSM in "IDLE" state
    When I attempt to advance to "FUSING"
    Then the transition should be rejected with an error

  @slow
  Scenario: Configurable discounts affect classification confidence
    Given custom discount factors with cosine 0.5 and svm 0.4
    When I run the classification pipeline with custom discounts
    Then the pipeline should reach CONVERGED state
    And the average confidence should differ from default discounts
