# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@gateway @tier-1
Feature: SPA route serving
  The gateway serves the React SPA for all non-API routes.
  Each client-side route must return 200 with HTML content.
  This catches misconfigured static file serving or build issues.

  Scenario Outline: SPA route serves HTML
    When I GET "<route>" from the gateway
    Then the response status should be 200
    And the response should contain "text/html"

    Examples:
      | route        |
      | /            |
      | /agents      |
      | /workflows   |
      | /embeddings  |
