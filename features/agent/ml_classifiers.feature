@tier-0
Feature: ML classifier mass functions
  As a classification pipeline, I need CatBoost and SVM mass functions
  to convert ML predictions into DST belief assignments.

  Background:
    Given the mock annotations vocabulary is loaded
    And a frame of discernment from the vocabulary

  Scenario: CatBoost mass converts probabilities to belief assignment
    When I compute CatBoost mass from {"1.1.1.1": 0.7, "1.1.2.1": 0.2, "2.1": 0.1}
    Then the ML mass function should not be vacuous
    And the ML top singleton should be "1.1.1.1"
    And theta mass should be approximately 0.15

  Scenario: CatBoost adaptive discount increases with high variance
    When I compute CatBoost mass with variance from {"1.1.1.1": 0.7, "2.1": 0.3} and {"1.1.1.1": 0.25, "2.1": 0.25}
    Then the ML mass function should not be vacuous
    And theta mass should be greater than 0.20

  Scenario: SVM mass converts calibrated probabilities
    When I compute SVM mass from {"1.1.2.1": 0.6, "1.1.2.2": 0.3, "2.1": 0.1}
    Then the ML mass function should not be vacuous
    And the ML top singleton should be "1.1.2.1"
    And theta mass should be approximately 0.20

  Scenario: Mass functions return vacuous on empty input
    When I compute CatBoost mass for empty probabilities
    And I compute SVM mass for empty probabilities
    Then both mass functions should be vacuous
