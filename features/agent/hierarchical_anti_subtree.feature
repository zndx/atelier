# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

@agent @tier-0
Feature: Hierarchical anti-example carves the descendant subtree

  When negative-channel (anti-example) evidence in late-interaction
  cosine targets an internal-node tag X whose descendant leaf set is
  D(X), the mass function must allocate to the complement focal
  element Θ \ D(X) — *the truth is not anywhere in this subtree*.
  The previous singleton subtraction (Θ \ {X}) is a silent no-op when
  X is not itself a leaf, and would have collapsed the negative
  channel's contribution to zero in any hierarchical taxonomy that
  uses internal-node tags as first-class classification targets.

  This is the load-bearing semantic for the every-node-is-a-tag
  policy that ``HierarchicalClassification.from_combined_evidence``
  already embodies (see ``belief.py`` "Atlas-style 'every node is a
  tag'") and that the late-interaction cosine source must honour
  consistently.

  These scenarios use an abstract test taxonomy with arbitrary codes
  whose hierarchy is conveyed *only* through the ``parent_code``
  column.  No mnemonic, no dotted-number notation, no relationship
  to any specific ontology — the structural property is namespace-
  agnostic and applies whether tags are arranged as filesystem
  directories, HDF5 groups, RDF subClassOf chains, or any other
  parent-child structure.

  Background:
    Given an abstract hierarchical taxonomy
      | code   | parent |
      | node_a |        |
      | node_b |        |
      | node_c |        |
      | node_d | node_a |
      | node_e | node_a |
    # Resulting leaf set: {node_b, node_c, node_d, node_e}
    # Internal nodes:      {node_a} with descendants {node_d, node_e}

  Scenario: Anti-example targeting an internal node allocates mass to its descendant complement
    # The structural correctness test: a high-confidence anti-example
    # against ``node_a`` must produce evidence "the answer is not in
    # the {node_d, node_e} subtree", not the broken "the answer is
    # not the literal string node_a" (which is vacuous because node_a
    # is never a candidate at the leaf level in the first place).
    Given a late-interaction tag score for "node_a" with positive 0.30 and negative 0.85
    And a late-interaction tag score for "node_b" with positive 0.60 and negative 0.00
    And a late-interaction tag score for "node_c" with positive 0.10 and negative 0.00
    When I compute the late-interaction mass function over those scores
    Then the mass function should contain a focal element exactly covering codes {node_b, node_c}
    And the focal element covering codes {node_b, node_c} should carry strictly positive mass
    And the mass function should not contain a focal element covering all leaf codes minus the singleton "node_a"

  Scenario: Anti-example targeting a leaf produces a singleton complement (leaf-case regression)
    # When the anti-example target is itself a leaf, the descendant
    # set is the singleton.  The previous behaviour is preserved: a
    # leaf-targeted negative produces a complement focal element with
    # all leaves except that one.
    Given a late-interaction tag score for "node_d" with positive 0.20 and negative 0.85
    And a late-interaction tag score for "node_b" with positive 0.60 and negative 0.00
    And a late-interaction tag score for "node_c" with positive 0.10 and negative 0.00
    And a late-interaction tag score for "node_e" with positive 0.10 and negative 0.00
    When I compute the late-interaction mass function over those scores
    Then the mass function should contain a focal element exactly covering codes {node_b, node_c, node_e}

  Scenario: Channel conflict K arises when positive and negative target the same internal node
    # Both channels carry strong evidence pointing at node_a (positive
    # saying "yes node_a", negative saying "not in node_a's subtree").
    # The DST machinery must surface this as Dempster conflict K
    # rather than silently cancelling — the channel-decomposed
    # construction's load-bearing property.
    Given a late-interaction tag score for "node_a" with positive 0.80 and negative 0.85
    And a late-interaction tag score for "node_b" with positive 0.20 and negative 0.00
    And a late-interaction tag score for "node_c" with positive 0.15 and negative 0.00
    When I compute the late-interaction mass function over those scores
    Then the mass function should contain a focal element exactly covering codes {node_d, node_e}
    And the focal element covering codes {node_d, node_e} should carry strictly positive mass
    And the focal element covering codes {node_b, node_c} should carry strictly positive mass

  Scenario: Internal-node tag is a first-class prediction target
    # Positive late-interaction evidence supporting an internal-node
    # tag must land mass on that node's focal element directly — not
    # silently disappear via a singletons-only filter and not get
    # diffusely smeared across its descendant leaves.
    Given a late-interaction tag score for "node_a" with positive 0.85 and negative 0.00
    And a late-interaction tag score for "node_b" with positive 0.10 and negative 0.00
    And a late-interaction tag score for "node_c" with positive 0.10 and negative 0.00
    When I compute the late-interaction mass function over those scores
    Then the mass function should contain a focal element exactly covering codes {node_d, node_e}
    And the focal element covering codes {node_d, node_e} should be the top non-Θ focal element by mass
