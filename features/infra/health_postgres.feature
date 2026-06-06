@infra @health @postgres
Feature: PostgreSQL health

  @tier-1
  Scenario: Connect to PostgreSQL
    When I connect to the atelier database
    Then the connection succeeds
    And extension "vector" is loaded

  @tier-1
  Scenario: Migrations are applied
    When I connect to the atelier database
    Then table "schema_migrations" exists
