# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: Synthetic data generation
  As a data engineer, I need synthetic training data with a known
  curated reference to train CatBoost and SVM classifiers.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: Synth generates columns for all leaf categories
    When I generate synthetic data with 5 variants per category
    Then every leaf category should have at least 3 columns
    And the curated reference should map each column to a valid category code

  Scenario: Synth generation is deterministic
    When I generate synthetic data with seed 42
    And I generate synthetic data again with seed 42
    Then both outputs should be identical
