# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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

  Scenario: Currency pattern rejects non-currency 3-letter codes
    When I run pattern detection on values "USA, NYC, ABC, XYZ, LAX"
    Then the detected patterns should not include "iso_currency_pattern"
    When I run pattern detection on values "USD, EUR, GBP, JPY, CAD"
    Then the detected patterns should include "iso_currency_pattern"

  Scenario: Date pattern rejects impossible dates
    When I run pattern detection on values "1234-99-7890, 2345-88-8901, 3456-77-9012"
    Then the detected patterns should not include "date_iso_pattern"
    When I run pattern detection on values "2024-01-15 10:30:00, 2024-02-28 14:45:00, 2024-03-01 09:00:00"
    Then the detected patterns should include "datetime_iso_pattern"

  Scenario: Credit card pattern rejects non-Luhn digit strings
    When I run pattern detection on values "1234567890123, 1111111111111, 2222222222222"
    Then the detected patterns should not include "credit_card_pattern"

  Scenario: IPv4 pattern rejects invalid octet ranges
    When I run pattern detection on values "999.999.999.999, 300.400.500.600, 256.1.1.1"
    Then the detected patterns should not include "ipv4_pattern"
    When I run pattern detection on values "192.168.1.1, 10.0.0.1, 172.16.0.1"
    Then the detected patterns should include "ipv4_pattern"

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
    And the accuracy against the curated reference should exceed 0.6
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

  Scenario: Quarantined patterns do not contribute mass
    # phone_pattern, date_iso_pattern, vin_pattern, license_plate_pattern,
    # and friends have been moved to _QUARANTINED_PATTERN_MAP per the
    # 2026-04-17 overwatch audit — they fire wrong 8× more than right
    # and compete with LLM evidence rather than corroborating it.
    Given a frame of discernment from the vocabulary
    When I compute pattern_to_mass for signals ["phone_pattern"]
    Then the pattern mass should be vacuous
    When I compute pattern_to_mass for signals ["date_iso_pattern"]
    Then the pattern mass should be vacuous
    When I compute pattern_to_mass for signals ["vin_pattern"]
    Then the pattern mass should be vacuous
    When I compute pattern_to_mass for signals ["license_plate_pattern"]
    Then the pattern mass should be vacuous

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

  # ── Monte Carlo Sampling ─────────────────────────────────────

  Scenario: MC config loads from HOCON with defaults
    Then MC config should load from HOCON with default values

  Scenario: MC passthrough when corpus below threshold
    Given the fixture corpus with 50 columns
    And an MC config with min_corpus_size 200
    When I run pre-classification on the corpus
    And I stratify the pre-classified columns
    And I select the MC sample
    Then the MC plan should be passthrough
    And frontier + propagation should cover all columns

  Scenario: MC activates when corpus exceeds threshold
    Given the fixture corpus with 50 columns
    And an MC config with min_corpus_size 5
    When I run pre-classification on the corpus
    And I stratify the pre-classified columns
    And I select the MC sample
    Then the MC plan should NOT be passthrough
    And frontier columns should be a subset of all columns
    And frontier + propagation should cover all columns

  Scenario: Pre-classification assigns every column
    Given the fixture corpus with 50 columns
    And an MC config with min_corpus_size 5
    When I run pre-classification on the corpus
    Then every column should have a pre-classification

  Scenario: Stratification produces meaningful strata
    Given the fixture corpus with 50 columns
    And an MC config with min_corpus_size 5
    When I run pre-classification on the corpus
    And I stratify the pre-classified columns
    Then there should be at least 2 strata

  # ── Row-Level Monte Carlo ──────────────────────────────────

  @row-mc
  Scenario: Row MC config loads from HOCON with defaults
    Then row MC config should load with default values

  @row-mc
  Scenario: Row reservoir preserves all values from sample source
    Given the OOTB sample source is loaded
    Then each column should have all_values with more entries than values

  @row-mc
  Scenario: Stratified row selection produces diverse subsets
    Given a column with 50 distinct values in its reservoir
    When I select stratified row samples for 3 iterations with k=10
    Then each iteration should produce a different value subset
    And each subset should contain 10 values

  @row-mc
  Scenario: Row MC passthrough when reservoir is small
    Given a column with 5 values in its reservoir
    When I select a row sample with k=10
    Then all 5 values should be returned unchanged

  @row-mc
  Scenario: Row-unstable columns are detected
    Given a bootstrap state with varying labels across row iterations
    Then the row stability for the unstable column should be below 0.5
    And the row stability for the stable column should be 1.0
