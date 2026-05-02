# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: Adaptive Settings focus computation
  After a pipeline run completes, the Settings page shows an adaptive
  "Focus" section curated from deterministic drift rules unioned with
  an optional overwatch-emitted JSON block. The BDD scenarios below
  exercise compute_focus() directly against synthesized snapshot +
  overwatch artifacts so we don't pay pipeline cost.

  Scenario: Deterministic drift flags drifting floats and ints
    Given a settings snapshot where mc_sample_fraction is 0.35 (default 0.15)
    And the overlay resolves classify_bootstrap_max_iterations to 10 (default 5)
    When I compute focus for that run
    Then the focus source is "rules"
    And the deterministic focus includes "mc_sample_fraction"
    And the deterministic focus includes "classify_bootstrap_max_iterations"
    And the overwatch focus is empty

  Scenario: Small float drift below 10% threshold is not flagged
    Given a settings snapshot where classify_llm_discount is 0.157 (default 0.15)
    When I compute focus for that run
    Then the deterministic focus does not include "classify_llm_discount"

  Scenario: Bool and choice changes always flag regardless of magnitude
    Given a settings snapshot where row_mc_enabled is true (default false)
    And the overlay resolves classify_fusion_strategy to "yager" (default "dempster")
    When I compute focus for that run
    Then the deterministic focus includes "row_mc_enabled"
    And the deterministic focus includes "classify_fusion_strategy"

  Scenario: Overwatch focus block contributes keys
    Given a settings snapshot at defaults
    And an overwatch report with focus keys "classify_discount_name_match_exact,mc_sample_fraction"
    When I compute focus for that run
    Then the focus source is "overwatch"
    And the overwatch focus includes "classify_discount_name_match_exact"
    And the overwatch focus includes "mc_sample_fraction"

  Scenario: Hybrid merge unions rules and overwatch sources
    Given a settings snapshot where mc_sample_fraction is 0.35 (default 0.15)
    And an overwatch report with focus keys "classify_bootstrap_clarity_target"
    When I compute focus for that run
    Then the focus source is "hybrid"
    And the merged focus includes "mc_sample_fraction"
    And the merged focus includes "classify_bootstrap_clarity_target"

  Scenario: Malformed overwatch focus block is tolerated silently
    Given a settings snapshot where mc_sample_fraction is 0.35 (default 0.15)
    And an overwatch report with a malformed focus block
    When I compute focus for that run
    Then the focus source is "rules"
    And the overwatch focus is empty
    And the deterministic focus includes "mc_sample_fraction"

  Scenario: Unknown keys from overwatch are discarded
    Given a settings snapshot at defaults
    And an overwatch report with focus keys "not_a_real_key,classify_llm_discount"
    When I compute focus for that run
    Then the overwatch focus includes "classify_llm_discount"
    And the overwatch focus does not include "not_a_real_key"

  Scenario: Missing snapshot yields a degraded but stable response
    Given a results directory with no settings_snapshot.json
    When I compute focus for that run
    Then the focus source is "no_snapshot"
    And the merged focus is empty
