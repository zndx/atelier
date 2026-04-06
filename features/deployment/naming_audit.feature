@deployment @naming
Feature: Naming conventions — no Apache Atlas confusion
  Ensures user-facing surfaces use "Embeddings Viewer" (not "Atlas Viewer")
  to avoid confusion with Apache Atlas in the Cloudera ecosystem.

  @tier-0
  Scenario: Landing page uses "Embeddings Viewer" label
    When I search ui/src/ for "Atlas Viewer"
    Then no matches are found
    When I search ui/src/ for "Embeddings Viewer"
    Then at least one match is found

  @tier-0
  Scenario: Documentation uses "Embeddings Viewer"
    When I search docs/src/ for "Atlas Viewer"
    Then no matches are found
