# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@infra @preflight
Feature: Preflight validation

  Operators run preflight before starting services to catch
  misconfigurations early. Deny = blocking, warn = advisory.

  @tier-0
  Scenario: Preflight passes with valid config
    When I run preflight checks
    Then the result has no denies

  @tier-0
  Scenario: Missing config file is a deny
    When I run preflight with a nonexistent config path
    Then the result has a deny for "config_exists"

  @tier-0
  Scenario: Missing credentials is a warning not a deny
    Given config with no LLM credentials
    When I run preflight checks against that config
    Then the result has a warning for "credentials_configured"
    And the result has no denies
