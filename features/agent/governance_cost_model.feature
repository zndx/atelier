@agent @tier-0
Feature: Sensitivity-classification perspective in the LLM system prompt
  Atelier nudges the LLM toward Type-II error aversion as a
  governance principle by invoking its existing knowledge of BFO/CCO
  and the privacy regimes (GDPR, HIPAA, PCI DSS) those ontologies
  overlap with — rather than re-teaching the model in prescriptive
  rule form.  See docs/src/architecture/dst-evidence-independence.md.

  The vocabulary-aware sensitivity map activates only on Atelier's
  own publicly-grounded ICE conventions (per
  src/atelier/classify/fixtures/PROVENANCE.md).  For every other
  vocabulary shape, the framework deliberately makes no assumptions
  about sensitivity encoding — the LLM gets the markdown category
  table, per-column ontology priors, and the perspective preamble.

  Scenario: System prompt invokes the LLM's ontology and privacy-regime knowledge
    # The perspective preamble is present in every system prompt
    # regardless of vocabulary.  Operators / auditors can grep the
    # system prompt for the ontology-grounding markers and confirm
    # the LLM was framed within the public knowledge base it already
    # has training on.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "BFO and CCO"
    And the system prompt contains "GDPR"
    And the system prompt contains "ontology prior"
    And the system prompt contains "non-sensitive is a larger error"
    And the system prompt contains "Calibrate confidence"

  Scenario: Sensitivity map activates on the universal vocab (ICE conventions)
    # When the loaded vocabulary uses Atelier's own publicly-grounded
    # ICE.SENSITIVE / ICE.NONSENSITIVE conventions, the prompt names
    # them so the LLM has an explicit runtime reference for "where to
    # land sensitive picks."  Every term named here traces to a
    # public source per fixtures/PROVENANCE.md.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "ICE.SENSITIVE"
    And the system prompt contains "ICE.NONSENSITIVE"
    And the system prompt mentions at least one of "SSN", "PAN", or "EMAIL"

  Scenario: Sensitivity map omitted on a fictitious non-ICE vocabulary
    # A vocabulary that does NOT use ICE conventions gets only the
    # perspective preamble — no sensitivity map is fabricated from
    # unfamiliar schema.  The framework makes no assumptions about
    # arbitrary taxonomies; future vocabularies with different
    # sensitivity encodings, scales, or no encoding at all all
    # degrade to this branch.
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    When I build the system prompt against that fictitious vocabulary
    Then the system prompt contains "BFO and CCO"
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
