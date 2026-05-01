@agent @tier-0
Feature: Governance Cost Model in the LLM system prompt
  Atelier biases the LLM's first-pass classification toward Type-II
  error aversion: in data governance, missing sensitive data and
  labeling it non-sensitive (false negative) is far more costly than
  over-classifying (false positive).  Cost-sensitive classification
  per Elkan 2001 (*The Foundations of Cost-Sensitive Learning*) and
  privacy-regime convention (GDPR Art. 25, HIPAA Safe Harbor, PCI
  DSS).  See docs/src/architecture/dst-evidence-independence.md.

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

  Scenario: System prompt names the high-sensitivity subtree on the universal vocab
    # When the vocab carries ICE.SENSITIVE / ICE.NONSENSITIVE path
    # conventions, the prompt's vocabulary-sensitivity-map section
    # names them so the LLM has a runtime reference for "where do I
    # land sensitive picks."
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "ICE.SENSITIVE"
    And the system prompt contains "ICE.NONSENSITIVE"
    And the system prompt mentions at least one of "SSN", "PAN", or "EMAIL"

  Scenario: System prompt degrades gracefully on a vocab with neither signal
    # No sensitivity ratings AND no ICE.* paths — the
    # vocabulary-specific summary block is omitted entirely; only
    # the fixed preamble stands.  The LLM never sees fabricated
    # codes from a runtime helper that couldn't infer structure.
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    When I build the system prompt against that fictitious vocabulary
    Then the system prompt contains "cost(FN) ≫ cost(FP)"
    And the system prompt does not contain "Vocabulary sensitivity map"
    And the system prompt does not contain "ICE.SENSITIVE"

  Scenario: Helper tiers a category by min sensitivity rating across roles
    # _category_min_rating ignores "N/A" values and returns the
    # numeric minimum across the role-keyed dict.  A category with
    # at least one role at rating 1 belongs to the high tier.
    Given a category with sensitivity ratings non_corp=1 and corp=N/A
    Then the category's min sensitivity rating is 1
    And the category tiers as high

  Scenario: Helper exemplars prefer codes with abbrevs
    # Exemplar selection sorts by abbrev-presence first, then
    # label-length ascending.  Codes with abbrevs surface ahead of
    # unabbreviated members so the LLM sees recognisable mnemonics
    # in the prompt.
    Given high-tier members "PAN(abbrev=PAN), Card-Last-4(no abbrev), CVV2(abbrev=CVV2)"
    Then the exemplar list contains "PAN" before "Card-Last-4"
    And the exemplar list contains "CVV2" before "Card-Last-4"
