@tier-0
Feature: Real data classification baseline
  As a data steward, I need to measure classification performance
  against real annotated data before production deployment.

  Background:
    Given the real data directory is available

  Scenario: Parse real CSVs and extract the curated reference
    When I parse real CSVs from the data directory
    Then at least 200 target columns should be extracted
    And every curated reference code should exist in the vocabulary

  Scenario: Template-based synthetic data covers real vocabulary
    When I generate template-based synthetic training data
    Then at least 150 categories should have synthetic columns

  @slow
  Scenario: ML-only classification baseline on real data
    When I run the real data ML train-eval cycle
    Then at least 200 columns should be classified
    And the hierarchical accuracy should exceed 0.30
    And the evaluation report should be written to the work directory
