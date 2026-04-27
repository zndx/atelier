# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@infra @health @pglite
Feature: PGlite Node.js process

  @tier-0
  Scenario: pglite-server.mjs script exists
    Then the file "scripts/pglite-server.mjs" exists

  @tier-0
  Scenario: PGlite npm dependencies are declared
    Given the file "scripts/package.json" exists
    Then it declares dependency "@electric-sql/pglite"
    And it declares dependency "@electric-sql/pglite-socket"
