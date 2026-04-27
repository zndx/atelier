# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@deployment @runtime-profile
Feature: CAI Runtime Profile
  Validates deployment readiness without requiring a live CAI environment.
  Catches import errors, missing scripts, config issues before pushing.

  @tier-0
  Scenario: Core package is importable
    When I import "atelier"
    Then no ImportError is raised
    And atelier.__version__ is defined

  @tier-0
  Scenario: All entry points are importable
    When I import "atelier.server"
    And I import "atelier.gateway"
    And I import "atelier.config"
    And I import "atelier.db.bootstrap"
    Then no ImportError is raised

  @tier-0
  Scenario: Proto stubs are generated and importable
    When I import "atelier.proto.atelier_pb2"
    And I import "atelier.proto.atelier_pb2_grpc"
    Then no ImportError is raised

  @tier-0
  Scenario: Required scripts exist and are executable
    Then the file "scripts/install_deps.py" exists
    And the file "scripts/startup_app.py" exists
    And the file "scripts/install_node.sh" is executable
    And the file "scripts/install_qdrant.sh" is executable
    And the file "bin/start-app.sh" is executable

  @tier-0
  Scenario: HOCON config resolves without errors
    When I load the config with no overrides
    Then no exception is raised
    And the config has grpc_port > 0
    And the config has gateway_port > 0

  @tier-0
  Scenario: Database migrations are parseable
    Given migration files exist in "db/migrations/"
    When I parse each migration for UP/DOWN blocks
    Then every migration has a valid UP block

  @tier-cai
  Scenario: Hive annotation sources are auto-discovered on startup
    Given ATELIER_DATA_CONNECTIONS includes a valid connection
    When the gateway starts
    Then a Hive data source should be registered for each connection with annotations
