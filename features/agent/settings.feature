# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: Runtime config overlay for the settings page
  The settings page writes into an in-memory overlay that is applied to
  AtelierConfig at the start of each pipeline run.  Values survive for
  the gateway process lifetime and reset on restart.

  Scenario: Empty overlay is a no-op
    Given the config overlay is empty
    When I apply the overlay to a loaded config
    Then the resulting config is unchanged

  Scenario: Applying a valid choice overlays the config
    Given the config overlay is empty
    When I set overlay "classify_fusion_strategy" to "yager"
    And I apply the overlay to a loaded config
    Then the resulting config has classify_fusion_strategy equal to "yager"

  Scenario: Applying a valid float overlays the config
    Given the config overlay is empty
    When I set overlay "classify_llm_discount" to 0.15
    And I apply the overlay to a loaded config
    Then the resulting config has classify_llm_discount equal to 0.15

  Scenario: Unknown keys are rejected
    Given the config overlay is empty
    When I set overlay "not_a_real_setting" to "anything"
    Then a ValueError is raised by the overlay

  Scenario: Out-of-range floats are rejected
    Given the config overlay is empty
    When I set overlay "classify_llm_discount" to 0.5
    Then a ValueError is raised by the overlay

  Scenario: Invalid choices are rejected
    Given the config overlay is empty
    When I set overlay "classify_fusion_strategy" to "unknown"
    Then a ValueError is raised by the overlay

  Scenario: Reset clears the overlay
    Given the config overlay is empty
    When I set overlay "classify_fusion_strategy" to "yager"
    And I clear the overlay
    And I apply the overlay to a loaded config
    Then the resulting config has classify_fusion_strategy equal to "dempster"
