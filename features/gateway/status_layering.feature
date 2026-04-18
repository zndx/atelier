@gateway @tier-0
Feature: Classify source layering — env vars, UI-saved rows, per-run overrides
  Precedence ladder (low → high):
    base.conf < .env.cai.enc < ATELIER_CLASSIFY_* env < data_sources row
    < fsm_start source_id.  Setting connection + database at deploy
  time seeds a data_source row so the env-driven default is visible +
  editable in the UI; the pipeline FSM start path uses cfg defaults
  when no source_id is passed.

  Scenario: fsm_start without source_id falls back to cfg classify defaults
    Given ATELIER_CLASSIFY_CONNECTION is "prod-hive" in config
    And ATELIER_CLASSIFY_DATABASE is "default" in config
    When I inspect the fsm_start resolution source
    Then the resolution reads cfg.classify_connection_name as default
    And the resolution reads cfg.classify_database as default
    And the resolution comment documents the precedence ladder

  Scenario: load_annotations_from_hive rejects unsafe database identifiers
    When I call load_annotations_from_hive with database "; DROP TABLE--"
    Then a ValueError is raised mentioning unsafe

  Scenario: load_annotations_from_hive accepts the whitelisted identifier shape
    When I call load_annotations_from_hive with database "hive_poc"
    Then no whitelist-violation error is raised before the cml import

  Scenario: _load_vocabulary falls through to Hive when vocab_uri is empty
    Given _load_vocabulary receives connection_name "prod-hive" and database "default" with no vocab_uri
    Then the env-default Hive branch is present in the function source
    And load_annotations_from_hive is called before the universal fallback

  Scenario: Seeded classify source appears in the data sources list
    Given ATELIER_CLASSIFY_CONNECTION is "test-conn" in config
    And ATELIER_CLASSIFY_DATABASE is "hive_poc" in config
    When _seed_classify_data_source runs against the in-process gateway
    Then a data_source row exists with id "classify-test-conn-hive_poc"
    And its source_uri is "test-conn/hive_poc"
    And its vocab_uri is "hive_poc.annotations"
    And its metadata carries seeded_from_env true

  Scenario: Seeder is idempotent — re-running preserves operator edits
    Given a data_source row with id "classify-test-conn-hive_poc" and a custom vocab_uri
    When _seed_classify_data_source runs a second time
    Then the row's custom vocab_uri is preserved
