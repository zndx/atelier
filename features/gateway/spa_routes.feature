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
