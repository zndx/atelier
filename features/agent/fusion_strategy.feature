# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: DST fusion strategy — Dempster vs Yager
  Yager's rule is a configurable alternative to Dempster's rule that
  redirects conflict mass to ignorance (Θ) instead of normalizing by
  (1-K).  When K=0, the two rules produce identical results.

  Scenario: K=0 equivalence — Yager matches Dempster when no conflict
    Given two non-conflicting mass functions
    When I combine them with Dempster's rule
    And I combine them with Yager's rule
    Then the results have identical mass distributions

  Scenario: Yager redirects conflict to Θ
    Given two conflicting mass functions with K > 0.5
    When I combine them with Yager's rule
    Then the Θ mass in the result is at least the conflict value
    And no singleton mass exceeds its raw conjunctive product

  Scenario: Dempster normalizes conflict away
    Given two conflicting mass functions with K > 0.5
    When I combine them with Dempster's rule
    Then the singleton masses sum to more than their raw products
    And the total mass sums to 1.0

  Scenario: combine_multiple requires theta for Yager
    Given three mass functions
    When I call combine_multiple with strategy "yager" and no theta
    Then a ValueError is raised
