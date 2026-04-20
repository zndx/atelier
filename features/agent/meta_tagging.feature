@tier-0
Feature: Meta-tagging blended overlay validation
  As a data steward, I need to validate our ontology against
  real enterprise annotations to ensure our hierarchy captures
  the concepts that matter in production.

  Scenario: Code mapping translates known meta-tagging codes
    Given a curated reference dict with meta-tagging codes
    When I translate the curated reference to ICE codes
    Then every translated code should be a valid ICE code
    And the translation should cover all provided codes

  Scenario: Blended vocabulary preserves hierarchy consistency
    Given the mock annotations vocabulary is loaded
    When I build a blended vocabulary from the base
    Then every leaf should have a valid parent chain to ICE root
