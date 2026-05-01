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

  Scenario: Auto-start prefers the user's last-selected source over env defaults
    # When the user picks a different source via the Status / Data
    # Platform UI, every manual /api/fsm/start records a FSM run with
    # that source_id.  An AMP restart that re-fires auto-start should
    # honor the user's last expressed intent — most-recent FSM run's
    # source_id — rather than dredging up the deployment-time
    # ATELIER_CLASSIFY_* env defaults.
    Given ATELIER_CLASSIFY_AUTO_START is "true"
    And ATELIER_CLASSIFY_CONNECTION is "hive-poc"
    And ATELIER_CLASSIFY_DATABASE is "default"
    And the most recent FSM run has source_id "reference_corpus/annotations"
    When _maybe_auto_start_classify runs
    Then fsm_start is called with source_id "reference_corpus/annotations"

  Scenario: Auto-start falls back to env defaults when no prior runs exist
    # Initial deploy: no FSM run history yet, _last_user_selected_source_id
    # returns None, fall through to the env-driven default.
    Given ATELIER_CLASSIFY_AUTO_START is "true"
    And ATELIER_CLASSIFY_CONNECTION is "hive-poc"
    And ATELIER_CLASSIFY_DATABASE is "default"
    And there are no prior FSM runs
    When _maybe_auto_start_classify runs
    Then fsm_start is called with source_id "hive-poc/default"
