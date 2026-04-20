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
