@gateway @tier-0
Feature: Classification pipeline auto-start on deploy
  ATELIER_CLASSIFY_AUTO_START gates the lifespan auto-dispatch. When
  the deploy sets AUTO_START=true AND a CONNECTION + DATABASE, the
  gateway calls fsm_start(source_id=classify-{conn}-{db}) during the
  seed phase so the operator lands on a usable run without any clicks.

  Scenario: Auto-start disabled — no fsm_start dispatch
    Given ATELIER_CLASSIFY_AUTO_START is "false"
    And ATELIER_CLASSIFY_CONNECTION is "hive-poc"
    And ATELIER_CLASSIFY_DATABASE is "default"
    When _maybe_auto_start_classify runs
    Then fsm_start is not called

  Scenario: Auto-start enabled with CONNECTION + DATABASE — dispatch fires
    Given ATELIER_CLASSIFY_AUTO_START is "true"
    And ATELIER_CLASSIFY_CONNECTION is "hive-poc"
    And ATELIER_CLASSIFY_DATABASE is "default"
    When _maybe_auto_start_classify runs
    Then fsm_start is called with source_id "classify-hive-poc-default"

  Scenario: Auto-start enabled but CONNECTION missing — skipped with warning
    Given ATELIER_CLASSIFY_AUTO_START is "true"
    And ATELIER_CLASSIFY_CONNECTION is ""
    And ATELIER_CLASSIFY_DATABASE is "default"
    When _maybe_auto_start_classify runs
    Then fsm_start is not called
