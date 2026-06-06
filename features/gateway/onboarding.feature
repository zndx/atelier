@gateway @tier-1 @slow
Feature: New-user onboarding flow
  Exercises the out-of-box experience: archive existing data for a
  clean slate, run the classification pipeline on OOTB sample data,
  and verify the full data lifecycle from empty state through to
  embeddings readiness. Uses archive/unarchive to simulate fresh
  installation without destructive data operations.

  Background:
    Given all data sources are archived for a clean slate

  Scenario: Fresh classification pipeline produces viewable embeddings
    # Verify the clean slate
    When I GET "/api/data-sources" from the gateway
    Then the response status should be 200
    And the response JSON "sources" should be an empty list
    When I GET "/api/datasets" from the gateway
    Then the response status should be 200
    And the response JSON "datasets" should be an empty list

    # Restore just the OOTB sample source (simulates first-boot seeding)
    When I POST "/api/data-sources/ootb-sample/unarchive" to the gateway
    Then the response status should be 200
    And the response JSON "ok" should be true

    # Verify the source is now visible
    When I GET "/api/data-sources" from the gateway
    Then the response status should be 200
    And the response JSON "sources" should be a non-empty list
    And a source with display_name containing "sample" should exist

    # Run the classification pipeline
    When I POST "/api/fsm/start?source_id=ootb-sample" to the gateway
    Then the response status should be 200
    And the response JSON should contain "started"
    When I poll "/api/fsm/status" until state is terminal
    Then the FSM state should be "CONVERGED"

    # Verify dataset was created with embeddings data
    When I GET "/api/datasets?source_id=ootb-sample" from the gateway
    Then the response status should be 200
    And the response JSON "datasets" should be a non-empty list
    And the latest dataset should have row_count greater than 100
    When I fetch the latest dataset's parquet data
    Then the parquet response status should be 200

  Scenario: Archived items remain accessible via include_archived
    When I GET "/api/data-sources?include_archived=true" from the gateway
    Then the response status should be 200
    And the response JSON "sources" should be a non-empty list

    When I GET "/api/datasets?include_archived=true" from the gateway
    Then the response status should be 200
