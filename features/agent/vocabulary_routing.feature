@agent @tier-0
Feature: Vocabulary routing per source type
  The pipeline loads different vocabularies depending on the data source.
  OOTB sample uses the 316-leaf ICE ontology; hive/synth sources use the
  customer's domain annotations directly.  Domain codes are the
  classification targets — the LLM reads labels and descriptions and
  classifies into hierarchical dot-codes.

  Scenario: OOTB sample uses 316-category ICE vocabulary
    When I resolve vocabulary for source "ootb-sample"
    Then the vocabulary has 316 categories
    And all codes start with "ICE."

  Scenario: Domain annotations are used directly as classification target
    Given a hive source with vocab_uri "meta.annotations"
    And load_annotations_from_hive is stubbed to return a non-empty CategorySet
    When I resolve vocabulary for the hive source
    Then load_annotations_from_hive was called with database "meta"
    And the resolved vocabulary matches the stub's return

  Scenario: Malformed vocab_uri fails the run instead of falling back
    Given a hive source with vocab_uri "nonexistent.table"
    When I resolve vocabulary for the hive source
    Then a RuntimeError is raised

  Scenario: Domain code hierarchy enables belief-path fall-up
    Given domain annotations with hierarchical dot-codes
    When I build a DST frame from the domain vocabulary
    Then internal nodes exist for parent codes

  Scenario: Adaptive batch sizing reduces for large vocabularies
    Given a vocabulary with 290 categories
    Then the estimated safe batch size is less than 50
    Given a vocabulary with 16 categories
    Then the estimated safe batch size is 50

  Scenario: User-selected vocab_uri routes to the chosen database
    Given a hive source with vocab_uri "reference_corpus.annotations"
    And ATELIER_CLASSIFY_DATABASE is "default"
    And load_annotations_from_hive is stubbed to return a non-empty CategorySet
    When I resolve vocabulary for the hive source
    Then load_annotations_from_hive was called with database "reference_corpus"
    And load_annotations_from_hive was not called with database "default"
    And the universal fallback was not used

  Scenario Outline: vocab_uri parser rejects malformed values
    When I parse the vocab_uri "<uri>"
    Then a ValueError is raised mentioning "<expected>"

    Examples:
      | uri                                 | expected                    |
      | reference_corpus                   | expected "{db}.annotations" |
      | reference_corpus.other_table       | table must be 'annotations' |
      | bad-name.annotations                | unsafe database identifier  |
      | .annotations                        | expected "{db}.annotations" |
      | reference_corpus.annotations.extra | expected "{db}.annotations" |

  Scenario: Annotations cache key is per-(connection, database)
    Given the annotations cache directory is empty
    And load_annotations_from_hive is stubbed to return a non-empty CategorySet
    When I resolve vocabulary for a hive source with vocab_uri "db_a.annotations" on connection "conn1"
    And I resolve vocabulary for a hive source with vocab_uri "db_b.annotations" on connection "conn1"
    Then the annotations cache contains "conn1__db_a.json"
    And the annotations cache contains "conn1__db_b.json"
    And load_annotations_from_hive was called twice

  Scenario: Hive load failure surfaces the underlying cause
    Given a hive source with vocab_uri "reference_corpus.annotations"
    And load_annotations_from_hive is stubbed to raise RuntimeError "Hive: table not found"
    When I resolve vocabulary for the hive source
    Then a RuntimeError is raised whose message mentions "reference_corpus.annotations"
    And the raised exception's __cause__ message contains "Hive: table not found"
    And the universal fallback was not used
