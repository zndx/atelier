# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: Vocabulary routing per source type
  The pipeline loads different vocabularies depending on the data source.
  OOTB sample uses the 316-leaf ICE ontology; hive/synth sources use
  the customer's domain annotations directly.  Domain codes are the
  classification targets — the LLM reads labels and descriptions and
  classifies into hierarchical dot-codes.

  Scenario: OOTB sample uses 316-leaf ICE vocabulary
    When I resolve vocabulary for source "ootb-sample"
    Then the vocabulary has 316 leaves
    And all leaf codes start with "ICE."

  Scenario: Domain annotations are used directly as classification target
    Given domain annotations with 50 leaf codes
    And a hive source with vocab_uri "meta.annotations"
    When I resolve vocabulary for the hive source
    Then the vocabulary has 50 leaves

  Scenario: Hive source requires annotations
    Given a hive source with no vocab_uri
    When I attempt to resolve vocabulary with vocab_uri
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
