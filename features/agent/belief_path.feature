# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: Belief path epistemic evaluation
  As a data steward, I need to understand not just what category
  the system predicts, but how certain it is at each hierarchy level,
  so I can make informed decisions about classification confidence.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: Belief path traces from leaf to root
    Given a frame of discernment from the vocabulary
    And evidence strongly supporting "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    When I classify using belief path evidence
    Then the belief path should start at "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And the belief path should end at "ICE"
    And belief should increase or stay the same ascending the path
    And plausibility should increase or stay the same ascending the path

  Scenario: Cautious classification chooses deepest confident level
    Given a frame of discernment from the vocabulary
    And weak evidence supporting "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    When I classify using belief path evidence
    Then the cautious code at threshold 0.7 should be an ancestor of "ICE.SENSITIVE.PID.CONTACT.EMAIL"
    And belief at the cautious code should exceed 0.7

  Scenario: Epistemic evaluation computes per-depth metrics
    Given classification results with belief paths
    When I run epistemic_evaluation
    Then per-depth metrics should include mean_bel and mean_pl and mean_gap
    And cautious_accuracy should be a valid fraction
    And mean_commitment_depth should be a positive number
