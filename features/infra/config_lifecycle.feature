@infra @config
Feature: Pipeline configuration lifecycle

  @tier-0
  Scenario: Load default configuration from base.conf
    When I load the config with no overrides
    Then the config has grpc_port = 50051
    And the config has gateway_port = 8090
    And the config has db_url starting with "postgresql+psycopg://"

  @tier-0
  Scenario: CLI overrides take precedence over defaults
    When I load the config with overrides
      | field     | value |
      | grpc_port | 9999  |
    Then the config has grpc_port = 9999

  @tier-0
  Scenario: Materialize and validate config
    Given I load the config with no overrides
    When I materialize the config to a temporary path
    And I validate the materialized config
    Then validation returns no errors

  @tier-0
  Scenario: Data connection names are parsed from HOCON config
    When I load the config with ATELIER_DATA_CONNECTIONS set to "conn-a, conn-b"
    Then the parsed connection names should be ["conn-a", "conn-b"]
