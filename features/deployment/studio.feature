@deployment @studio
Feature: CAI Studio modality (future)
  A Studio is a pre-built Docker image with IS_COMPOSABLE=true.
  Atelier runs as an embedded service within the Studio container.

  @tier-0
  Scenario: install_deps.py handles IS_COMPOSABLE root path
    When I set IS_COMPOSABLE to "true"
    And I parse scripts/install_deps.py for root_dir
    Then root_dir is "/home/cdsw/atelier"

  @tier-0
  Scenario: install_deps.py uses default root without IS_COMPOSABLE
    When IS_COMPOSABLE is not set
    And I parse scripts/install_deps.py for root_dir
    Then root_dir is "/home/cdsw"
