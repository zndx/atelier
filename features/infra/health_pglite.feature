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
