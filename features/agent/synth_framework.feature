# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-0
Feature: Onboarding synth framework
  As a data engineer, I need the synth framework to generate
  representative data for any vocabulary I provide, so I can
  benchmark classification before connecting real data sources.

  Background:
    Given the mock annotations vocabulary is loaded

  Scenario: Generator registry covers expanded vocabulary
    When I build the full generator registry
    Then the coverage report should show at least 80 percent total coverage

  Scenario: Vocabulary-driven generation produces valid output
    When I run generate_for_vocabulary with 5 variants per category
    Then CSV files should be created with columns for each leaf
    And reference_labels.json should map every column to a valid code
