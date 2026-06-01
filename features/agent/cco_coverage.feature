# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc. ...

@agent @tier-0
Feature: The test-gittables taxonomy realizes CCO module coverage
  An operator builds the public test-gittables fixture to exercise the
  classification critical path. These scenarios are the observations that
  convince them the taxonomy genuinely realizes CCO coverage — measured
  against the canonical 11 modules, with the one residual module positively
  represented as absence rather than silently dropped. tier-0: they inspect
  the committed fixture + the pure-Python epistemic machinery, no stack.
  See docs/src/architecture/cco-coverage.md.

  Background:
    Given the test-gittables taxonomy is loaded

  Scenario: Coverage is measured against the canonical 11 CCO modules
    Then the taxonomy data-covers 10 of the 11 canonical CCO modules
    And the only module without data leaves is "UNIT"
    And "UNIT" is positively represented in the CCO manifest, not dropped

  Scenario: Extended Relation is realized as relation data, not just types
    # CPA gives the data face CTA cannot reach: a column annotated by the
    # relation it expresses, distinct from its entity-type twin.
    Then the "REL" module has leaves grounded in DBpedia relation properties
    And the relation leaf "SDG.REL.CURRENCY" is distinct from the type leaf "SDG.CUR.CURRENCY"

  Scenario: Leaves are grounded in SDG terms over verified CCO IRIs
    # The fixture is a test-scoped subset of SDG, not a one-off namespace:
    # each leaf is a proposed SDG term expressed via an SDG property, and
    # every ICE class resolves to a CCO IRI verified against the published
    # ontology. The build emits the requirements artifact Aegir consumes.
    Then every leaf is an SDG term via "hasValueType" or "describesProperty"
    And every ICE class resolves to a verified CCO IRI
    And the fixture emits an SDG requirements artifact for Aegir

  Scenario: A measurement classification surfaces its unit absence
    When I classify a Quality leaf of the taxonomy
    Then it surfaces the semantic absence "unit" as "UNRESOLVED"
    And the unit absence is grounded in the CCO property "has_token_unit"

  Scenario: Taxonomy terms are grounded in CCO ExtendedRelation annotations
    Then every leaf carries the CCO annotations "acronym" and "definition_source"
    And those annotations key only to canonical ExtendedRelationOntology IRIs

  Scenario: Module coverage does not silence a per-read relation absence
    # Even though Extended Relation is now covered (the vocabulary exists), a
    # relational read whose relation is unresolved still surfaces the absence.
    Given the Extended Relation module is covered
    When a relational read leaves its relation unresolved
    Then the relation axis is surfaced as "UNRESOLVED"
