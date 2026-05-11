# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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

  Scenario: Sensitivity map omitted on a non-ICE vocabulary
    # A vocabulary outside Atelier's own ICE namespace (e.g. a CCO
    # domain extension, a DPV-based governance vocabulary, or a
    # legacy customer encoding) gets only the perspective preamble —
    # no sensitivity map is fabricated from unfamiliar schema.  The
    # framework deliberately makes no assumption about arbitrary
    # taxonomies' sensitivity encodings, scales, or absence of
    # encoding; any such vocabulary degrades to this branch.
    Given a non-ICE vocabulary with abbrev "TXNAMT" at code "cco:MoneyTransfer"
    When I build the system prompt against that non-ICE vocabulary
    Then the system prompt contains "BFO and CCO"
    And the system prompt does not contain "Vocabulary sensitivity map"
    And the system prompt does not contain "ICE.SENSITIVE"

  Scenario: Helper returns empty for non-ICE vocabularies
    # Direct unit-level assertion that _sensitive_subtree_summary
    # returns "" when the vocabulary does not use ICE conventions,
    # regardless of whether it carries other sensitivity metadata.
    # The helper deliberately avoids inferring structure from
    # arbitrary schemas.
    Given a non-ICE vocabulary with abbrev "TXNAMT" at code "cco:MoneyTransfer"
    Then _sensitive_subtree_summary returns the empty string for that vocabulary

  Scenario: Taxonomy renders parents as first-class rows alongside leaves
    # Atlas-style governance treats every node as a valid tagging
    # target.  ``build_category_tree`` renders the full hierarchy
    # so the LLM can vote at the level its evidence supports —
    # parent or leaf — instead of being forced into an over-
    # committed leaf pick.  The customer's per-row metadata
    # (abbrev / sensitivity / aliases) appears for parents too.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "## Taxonomy"
    And the system prompt contains "▸"
    And the system prompt contains "ICE.SENSITIVE"
    And the system prompt does not contain "exactly ONE leaf category"
    And the system prompt contains "most specific level you can defend"
    And the system prompt contains "name the parent"

  Scenario: Response-format example shows both leaf and parent picks
    # The contract demonstrates that parent-level votes at moderate
    # confidence are valid output — the LLM should not feel
    # obligated to manufacture a leaf pick when the evidence only
    # supports a parent.
    When I build the system prompt against the universal vocabulary
    Then the system prompt contains "amount_field"
    And the system prompt contains "parent-level"
