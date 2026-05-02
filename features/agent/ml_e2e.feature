# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: ML training end-to-end cycle
  As a data steward, I need the synth-train-predict-evaluate cycle
  to produce classifiers that contribute meaningful evidence.

  Background:
    Given the mock annotations vocabulary is loaded

  @slow
  Scenario: Synth-train-eval cycle produces accurate classifications
    When I run the synth-train-eval cycle
    Then the cycle should complete successfully
    And both CatBoost and SVM models should be trained
    And the pipeline should use at least 4 evidence sources
    And the overall accuracy should exceed 0.70

  @slow
  Scenario: CatBoost and SVM contribute non-vacuous evidence
    When I run the synth-train-eval cycle
    Then at least 50% of columns should have CatBoost evidence
    And at least 50% of columns should have SVM evidence
