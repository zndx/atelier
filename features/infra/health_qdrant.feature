# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@infra @health @qdrant
Feature: Qdrant health

  @tier-1
  Scenario: Qdrant health endpoint responds
    When I check the Qdrant health endpoint
    Then the response status is 200
