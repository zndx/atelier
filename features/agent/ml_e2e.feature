@tier-0
Feature: ML training end-to-end cycle
  As a data steward, I need the synth-train-predict-evaluate cycle
  to produce classifiers that contribute meaningful evidence.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: Synth-train-eval cycle produces accurate classifications
    When I run the synth-train-eval cycle
    Then the cycle should complete successfully
    And both CatBoost and SVM models should be trained
    And the pipeline should use at least 4 evidence sources
    And the overall accuracy should exceed 0.70

  Scenario: CatBoost and SVM contribute non-vacuous evidence
    When I run the synth-train-eval cycle
    Then at least 50% of columns should have CatBoost evidence
    And at least 50% of columns should have SVM evidence
