# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-1 @slow
Feature: SAGE feature importance analysis
  As a data scientist, I need to understand which features
  drive classification decisions.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: SAGE produces ranked importance for all 12 features
    When I run SAGE with 8 permutations on mock data
    Then the SAGE result should have 12 feature importance values
    And column_name should have non-zero importance
    And sample_values should have non-zero importance
    And column_name should be in the top 3 by absolute importance
