@tier-0
Feature: SHAP per-item explanations
  As a data steward, I need to understand why each column
  was classified the way it was.

  Background:
    Given the mock annotations vocabulary is loaded

  @slow
  Scenario: CatBoost TreeSHAP produces per-item explanations
    When I run the synth-train-eval cycle with SHAP enabled
    Then each classification should have shap_top1_name and shap_top1_value
    And shap values should be non-zero for at least 80% of columns

  @tier-1
  Scenario: Embedding PermutationSHAP produces per-item explanations
    When I run SHAP analysis with embedding permutation method on mock data
    Then the SHAP result should have explanations for all classified columns
    And each item should have top-3 feature attributions
    And column_name or sample_values should appear in most top-3 lists
