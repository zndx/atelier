# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# PROPOSED SPEC — companion guard for the test-gittables fixture.
# Moves to features/agent/fixture_provenance.feature with step defs.
#
# This file contains material proprietary to Cloudera, Inc. ...

@agent @tier-0
Feature: The test-gittables fixture is strictly public, with no UAT leakage
  The modernBERT+NHSVM critical-path test trains a head + builds a Qdrant
  collection on a committed sample of the GitTables semantic-type taxonomy.
  GitTables is public (CC BY 4.0; DBpedia/Schema.org-grounded), chosen
  precisely so the test corpus is a NON-target domain — no customer /
  UAT data, conventions, or answer keys may enter git through it.

  This is the release-blocker guard. It is @tier-0 (inspects committed
  fixture files only — no head, no Qdrant, no pipeline run) so it runs on
  every `just behave`, even when the @slow efficacy tier is skipped. It
  exists to catch the contamination class unwound on 2026-04-30, when a
  mock_annotations.json fixture carried customer conventions into the
  universal layer. See src/atelier/classify/fixtures/PROVENANCE.md for the
  attribution convention this extends, and meta_tagging_source.py's
  reference-column ("answer key") exclusion invariant.

  Background:
    Given the committed test-gittables fixture under src/atelier/classify/fixtures/test-gittables/

  Scenario: Every term traces to a public ontology, not a customer taxonomy
    When I load the test-gittables taxonomy
    Then every code resolves to a public namespace (gittables / dbpedia / schema.org)
    And no code resolves to the customer ICE.* or Hive-loaded domain namespaces
    And test-gittables/PROVENANCE.md lists a public source for every leaf term

  Scenario: No customer naming conventions leak through codes, labels, or abbrevs
    # Same invariants the universal vocabulary already enforces, applied
    # to the fixture: the C_ class-prefix convention and SKOS notation
    # codes are customer-supplied surfaces that must never appear here.
    When I load the test-gittables taxonomy
    Then no abbrev value begins with "C_"
    And the notation field is empty for every entry
    And no label or description contains a customer-internal abbreviation

  Scenario: No reference/answer-key columns hide in the fixture entities
    # Reference columns (^(attr|code|col|...)_\d+...) are synth answer
    # keys, never classifiable inputs. The fixture's target-domain
    # entities must be natural-named columns only — no answer keys
    # committed alongside them.
    When I load the test-gittables held-out entities
    Then no entity column name matches the reference-column answer-key pattern
    And every entity value traces to GitTables public table provenance

  Scenario: The fixture build surface cannot reach the customer corpus
    # The hermetic guarantee: `just optimize svm --fixture` must refuse
    # the customer input surface entirely, not merely skip the agent.
    When I resolve the input paths for `just optimize svm --fixture`
    Then the reference database is unset (no Hive reference connection)
    And no resolved input path lives under build/data/agent_mediated/
    And no resolved input path lives under the customer synth corpus
    And resolving any customer path in fixture mode raises a fail-closed error
