@agent @tier-0
Feature: Governance Cost Model in the LLM system prompt
  Atelier biases the LLM's first-pass classification toward Type-II
  error aversion: in data governance, missing sensitive data and
  labeling it non-sensitive (false negative) is far more costly than
  over-classifying (false positive).  Cost-sensitive classification
  per Elkan 2001 (*The Foundations of Cost-Sensitive Learning*) and
  privacy-regime convention (GDPR Art. 25, HIPAA Safe Harbor, PCI
  DSS).  See docs/src/architecture/dst-evidence-independence.md.

  The vocabulary-aware sensitivity map activates only on Atelier's
  own publicly-grounded ICE conventions (per
  src/atelier/classify/fixtures/PROVENANCE.md).  For every other
  vocabulary shape, the framework deliberately makes no assumptions
  about sensitivity encoding — the LLM gets the markdown category
  table, per-column ontology priors, and the fixed cost-asymmetry
  preamble, which is sufficient to navigate any taxonomy without
  the framework guessing at its structure.

  Scenario: System prompt carries cost-asymmetry language
    # The fixed Governance Cost Model preamble is present in every
    # system prompt regardless of vocabulary.  Operators / auditors
    # can grep the system prompt for the cost-asymmetry markers and
    # confirm the LLM was instructed under the Type-II-aversion
    # principle.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "cost(FN) ≫ cost(FP)"
    And the system prompt contains "Type II"
    And the system prompt contains "Honest confidence calibration"
    And the system prompt contains "ontology priors"
    And the system prompt contains "Guard against over-classification"

  Scenario: Sensitivity map activates on the universal vocab (ICE conventions)
    # Branch B: when the loaded vocabulary uses Atelier's own
    # publicly-grounded ICE.SENSITIVE / ICE.NONSENSITIVE
    # conventions, the prompt names them so the LLM has an explicit
    # runtime reference for "where to land sensitive picks."  Every
    # term named here traces to a public source per
    # fixtures/PROVENANCE.md.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "ICE.SENSITIVE"
    And the system prompt contains "ICE.NONSENSITIVE"
    And the system prompt mentions at least one of "SSN", "PAN", or "EMAIL"

  Scenario: Sensitivity map omitted on a fictitious non-ICE vocabulary
    # Branch C: a vocabulary that does NOT use ICE conventions
    # gets only the fixed cost-asymmetry preamble — no sensitivity
    # map is fabricated from unfamiliar schema.  The framework
    # makes no assumptions about arbitrary taxonomies; future
    # vocabularies with different sensitivity encodings, scales,
    # or no encoding at all all degrade to this branch.
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    When I build the system prompt against that fictitious vocabulary
    Then the system prompt contains "cost(FN) ≫ cost(FP)"
    And the system prompt does not contain "Vocabulary sensitivity map"
    And the system prompt does not contain "ICE.SENSITIVE"

  Scenario: Helper returns empty for non-ICE vocabularies
    # Direct unit-level assertion that _sensitive_subtree_summary
    # returns "" when the vocabulary does not use ICE conventions,
    # regardless of whether it carries other sensitivity metadata.
    # The helper deliberately avoids inferring structure from
    # arbitrary schemas.
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    Then _sensitive_subtree_summary returns the empty string for that vocabulary
