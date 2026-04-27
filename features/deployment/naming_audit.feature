# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@deployment @naming
Feature: Naming conventions — no Apache Atlas confusion
  Ensures user-facing surfaces use "Embeddings" (not "Atlas Viewer")
  to avoid confusion with Apache Atlas in the Cloudera ecosystem.

  @tier-0
  Scenario: Landing page uses "Embeddings" label
    When I search ui/src/ for "Atlas Viewer"
    Then no matches are found
    When I search ui/src/ for "Embeddings"
    Then at least one match is found

  @tier-0
  Scenario: Documentation uses "Embeddings"
    When I search docs/src/ for "Atlas Viewer"
    Then no matches are found
