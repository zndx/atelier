@tier-0 @agent @coverage
Feature: LLM sweep must classify every column the operator asked for
  As an operator who set 100% coverage, I expect every column to reach
  the LLM.  Four failure modes have burned us in production:
    1. Reference columns leaking into the Hive-sampled corpus;
    2. Bedrock silently capping max_tokens below the configured value;
    3. Backends returning fewer classifications than requested with a
       clean stop_reason, masking the loss;
    4. Hard-coded tables_limit/sample_size function defaults winning
       over operator-configured HOCON/env values.

  Scenario: Reference columns are excluded regardless of loader
    Given a mixed sample set with natural-named and reference-named columns
    When I apply the reference-column exclusion invariant
    Then the reference columns are dropped
    And sibling contexts no longer reference the dropped columns
    And production-shape column names are untouched

  Scenario: Bedrock per-model max_tokens ceilings are enforced
    Given the Bedrock output-token ceiling table
    Then claude-3-5-sonnet resolves to 8192
    And claude-sonnet-4 resolves to 64000
    And claude-3-haiku resolves to 4096
    And an unknown model falls back to 4096
    And a Bedrock inference-profile ARN is matched on the model substring

  Scenario: Partial LLM responses force halving retry
    Given a backend that returns fewer classifications than requested
    When the LLM sweep processes a batch
    Then the response carries partial=True even with a clean stop_reason
    And the truncated property is True
    And halving retry engages on the partial response

  Scenario: Coverage gap triggers targeted retry
    Given an LLM sweep where the first pass drops some columns
    When the sweep completes
    Then a coverage-gap retry runs on the missing columns
    And every requested column ends up with a label

  Scenario: Pipeline respects HOCON-configured discovery limits
    Given an AtelierConfig with classify_tables_limit=42 and classify_sample_size=17
    When a caller invokes the pipeline without passing those values
    Then the pipeline uses 42 and 17, not the hard-coded function defaults

  Scenario: No prediction path regex-decodes reference column names
    # Defends against the "UAT renames reference columns, our numbers
    # collapse" accusation.  The reference regex must ONLY be used to
    # (a) filter the sample set, (b) build the curated reference
    # (answer key for evaluation).  It must NOT appear in the LLM
    # sweep, cosine, pattern, mass-function, or DST fusion paths.
    Given the set of modules on the classifier prediction path
    When I scan each for uses of the reference-column regex
    Then no prediction-path module imports or matches the regex

  Scenario: All-columns parquet does not fabricate predictions from names
    # The UAT-coverage parquet ships reference-column rows so row-audit
    # scripts find every source column, but those rows must not carry
    # predictions decoded from the column name — that would pass our
    # synth corpus and silently fail any renamed-column validation set.
    Given the bundle's atelier_predictions_all_columns.parquet
    When I inspect the reference-column rows
    Then predicted_code is empty on every reference-column row
    And matches_reference is None on every reference-column row
    And the evidence field explains the row is excluded by configuration

  Scenario Outline: Reference-column regex filters every known prefix
    # Pin the prefix list so someone adding a new one has to come
    # through this test.  Shape is <prefix>_<digit>(_<digit>)*, anchored,
    # case-sensitive — deliberately strict so production naming
    # (e.g. ``product_id_classified``, ``account_coded``) doesn't
    # accidentally match.
    Given a column name "<name>"
    Then the reference-column regex match is <expected>

    Examples: synth answer-key prefixes (all must filter)
      | name           | expected |
      | attr_1_1_1     | True     |
      | code_1_4_2     | True     |
      | col_1_2        | True     |
      | data_1         | True     |
      | field_1_2_3    | True     |
      | item_1_5_2_1   | True     |
      | key_1          | True     |
      | ref_1_2        | True     |
      | val_1_7_1      | True     |
      | var_1          | True     |

    Examples: production-style names (must NOT filter)
      | name                   | expected |
      | first_name             | False    |
      | customer_id            | False    |
      | product_id_classified  | False    |
      | account_number_coded   | False    |
      | ATTR_1_1               | False    |
      | attr_name              | False    |
      | attr_1x                | False    |
