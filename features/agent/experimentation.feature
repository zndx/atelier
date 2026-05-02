# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: User experimentation with custom taxonomy
  As a data steward, I need to bring my own domain taxonomy,
  run the DST pipeline against it, and verify that results are
  consistent with my vocabulary — the core experimentation flow.

  This exercises the critical user journey phase transition from
  "orienting and understanding" to "experimentation". Uses a mock
  customer taxonomy by default; overrides with real meta-tagging
  annotations when ATELIER_REAL_DATA_DIR is set and data exists.

  Background:
    Given the user taxonomy is loaded

  Scenario: Custom vocabulary composes with universal base
    Then the composed vocabulary should have more leaves than universal alone
    And every custom leaf should be reachable from the ICE root

  Scenario: Synth generators cover custom vocabulary categories
    When I build a generator registry for the composed vocabulary
    Then at least 80 percent of custom leaf categories should have generators

  @slow
  Scenario: Pipeline produces vocabulary-consistent classifications with custom taxonomy
    When I run the classification pipeline with the composed vocabulary and mock LLM
    Then the pipeline should reach CONVERGED state
    And every predicted code should exist in the composed vocabulary
    And the belief paths should trace to roots in the universal hierarchy
