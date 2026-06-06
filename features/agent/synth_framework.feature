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
