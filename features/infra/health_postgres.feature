# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@infra @health @postgres
Feature: PostgreSQL health

  @tier-1
  Scenario: Connect to PostgreSQL
    When I connect to the atelier database
    Then the connection succeeds
    And extension "vector" is loaded

  @tier-1
  Scenario: Migrations are applied
    When I connect to the atelier database
    Then table "schema_migrations" exists
