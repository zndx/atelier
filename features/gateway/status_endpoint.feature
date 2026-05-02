# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@gateway @status
Feature: Status endpoint
  The GET /api/status endpoint aggregates infrastructure health probes
  and configuration state for the operator dashboard.

  @tier-0
  Scenario: Status endpoint returns expected shape
    When I GET /api/status from a test client
    Then the status response contains "grpc"
    And the status response contains "postgres"
    And the status response contains "qdrant"
    And the status response contains "config"
    And the status response contains "connected"

  @tier-0
  Scenario: Status endpoint reports config state
    When I GET /api/status from a test client
    Then the config section contains "agent_model"
    And the config section contains "has_anthropic"
    And the config section contains "has_bedrock"

  @tier-0
  Scenario: Status endpoint does not leak credentials
    When I GET /api/status from a test client
    Then the status response does not contain "anthropic_api_key"
    And the status response does not contain "aws_secret_access_key"
    And the status response does not contain "aws_access_key_id"

  @tier-1
  Scenario: Status endpoint reports connected when stack is healthy
    When I GET /api/status from the running gateway
    Then the status response "connected" is true
    And the status response "grpc.ok" is true
    And the status response "postgres.ok" is true
    And the status response "qdrant.ok" is true
