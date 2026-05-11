# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@gateway @tier-1
Feature: Gateway API endpoint validation
  The gateway bridges REST to gRPC. With the stack healthy and
  migrations applied, every endpoint returns substantive data
  from seeded agents, skills, vocabulary, and data sources.

  Scenario: Agents list returns all 5 keystone agents
    When I GET "/api/agents" from the gateway
    Then the response status should be 200
    And the response JSON "agents" should have at least 5 items
    And every agent should have "id" and "name" and "role"
    And the agent roles should include "sampler" and "classifier"

  Scenario: Skills endpoint returns skill definitions
    When I GET "/api/skills" from the gateway
    Then the response status should be 200
    And the response JSON "skills" should be a non-empty list
    And every skill should have "id" and "title" and "content"

  Scenario: Data sources includes OOTB sample
    When I GET "/api/data-sources" from the gateway
    Then the response status should be 200
    And the response JSON "sources" should be a non-empty list
    And a source with display_name containing "sample" should exist

  Scenario: Vocabulary stats reports 300+ terms for sample source
    When I GET "/api/vocabulary/stats?source_id=ootb-sample" from the gateway
    Then the response status should be 200
    And the response JSON "terms" should be at least 300

  Scenario: FSM status returns idle or valid state
    When I GET "/api/fsm/status" from the gateway
    Then the response status should be 200
    And the response JSON should contain "state"
    And the response JSON "state" should be a known FSM state

  Scenario: Health endpoint confirms gRPC connection
    When I GET "/api/health" from the gateway
    Then the response status should be 200
    And the response JSON should contain "status"

  Scenario: Vocabulary endpoint returns universal base terms
    When I GET "/api/vocabulary/stats" from the gateway
    Then the response status should be 200
    And the response JSON "terms" should be at least 25

  Scenario: Status endpoint reports all infrastructure probes
    When I GET "/api/status" from the gateway
    Then the response status should be 200
    And the response JSON should contain "grpc"
    And the response JSON should contain "postgres"
    And the response JSON should contain "qdrant"
    And the response JSON should contain "connected"
