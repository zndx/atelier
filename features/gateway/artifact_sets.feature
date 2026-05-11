# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@tier-1 @gpu
Feature: ML Artifact Sets gateway endpoints
  Operators interact with artifact sets through a small REST surface
  exposed by the FastAPI gateway: list, activate, archive, check
  vocab compatibility, and start an Extend Classification run.  This
  feature exercises the endpoints end-to-end against a real DB so a
  contract regression is caught before it reaches the React UI.

  Background:
    Given the FastAPI gateway is reachable
    And an artifact set is registered for the OOTB Sample source

  Scenario: GET /api/artifact-sets returns the registered rows
    When I call GET "/api/artifact-sets"
    Then the response status should be 200
    And the response body should have field "artifact_sets"
    And the artifact_sets list should contain at least 1 row

  Scenario: GET /api/artifact-sets/{id} returns the requested row
    When I call GET "/api/artifact-sets/{seeded_artifact_set_id}"
    Then the response status should be 200
    And the response body should have field "catboost_path"
    And the response body should have field "vocab_signature"

  Scenario: POST .../activate flips is_active for the targeted row
    When I call POST "/api/artifact-sets/{seeded_artifact_set_id}/activate"
    Then the response status should be 200
    And the response body should have field "ok" with value true

  Scenario: GET .../compatibility surfaces the vocab status
    When I call GET "/api/artifact-sets/{seeded_artifact_set_id}/compatibility?source_id=ootb-sample"
    Then the response status should be 200
    And the response body should have field "status"

  Scenario: POST /api/fsm/extend rejects a missing artifact_set_id with 400
    When I call POST "/api/fsm/extend" with body {"source_id": "ootb-sample"}
    Then the response status should be 400

  Scenario: POST /api/fsm/extend with a nonexistent artifact set returns 404
    When I call POST "/api/fsm/extend" with body {"source_id": "ootb-sample", "artifact_set_id": "does-not-exist-xyz"}
    Then the response status should be 404

  Scenario: GET /api/artifact-sets/{id} returns 404 for a missing row
    When I call GET "/api/artifact-sets/does-not-exist-xyz"
    Then the response status should be 404
