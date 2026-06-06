@tier-1 @gpu
Feature: Extend Classification — streamlined inference pipeline
  As a governance engineer with a freshly trained ML artifact set, I
  need to apply it to new columns or new tables WITHOUT re-running the
  full classify pipeline (LLM sweep + DST iteration + agent loop).
  The Extend pipeline reuses CatBoost+SVM+UMAP from a saved bundle and
  produces a new Dataset visible in the Embeddings UI.

  Background:
    Given the mock annotations vocabulary is loaded
    And an existing classify-run artifact set is registered

  Scenario: Extend run produces a new dataset with run_kind=extend
    Given the artifact set is the active artifact set
    When I run Extend Classification against the OOTB Sample source
    Then the Extend run should complete with state CONVERGED
    And the new dataset row should have run_kind "extend"
    And the new dataset row should reference the consumed artifact set

  Scenario: Extend run never invokes the LLM backend
    Given an LLM call counter starting at zero
    And the artifact set is the active artifact set
    When I run Extend Classification against the OOTB Sample source
    Then the Extend run should complete with state CONVERGED
    And the LLM call counter should still be zero

  Scenario: Extend run surfaces vocabulary compatibility status
    Given the artifact set is the active artifact set
    When I run Extend Classification against the OOTB Sample source
    Then the run summary should include a vocab_compatibility status
    And the vocab_compatibility status should be one of "ok|superset|partial|disjoint"

  Scenario: Extend writes atlas-compatible parquet
    Given the artifact set is the active artifact set
    When I run Extend Classification against the OOTB Sample source
    Then the run dir should contain "atelier_embeddings.parquet"
    And the run dir should contain "classifications.json"
    And the run dir should contain "evaluation_report.json"
    And the run dir should contain "settings_snapshot.json"
