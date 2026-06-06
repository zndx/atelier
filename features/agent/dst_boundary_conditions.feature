@agent @tier-0
Feature: DST boundary conditions across observed classification failure modes

  Each scenario captures a user-observable failure mode at the boundary
  where a small hyperparameter change flips the verdict.  The point is
  not to enumerate every permutation but to lock in the *transit*
  behaviour — what happens just below and just above the threshold — so
  future tuning has empirical anchors.  Each scenario pairs a
  tuned-knob assertion with a non-regression check confirming the
  tuning does not silently break easy classifications.

  As elsewhere, the test taxonomy is abstract: parent-child links are
  conveyed *only* through the ``parent`` column.  No positional
  notation, no real-world mnemonics.  The taxonomy is a branching tree
  (no single-child chains) so descendant-set inclusion gives a
  well-defined specificity ordering independent of any code-string
  convention.

  Background:
    Given an abstract multi-depth taxonomy
      | code          | parent        |
      | left_root     |               |
      | left_branch_a | left_root     |
      | left_branch_b | left_root     |
      | left_leaf_1   | left_branch_a |
      | left_leaf_2   | left_branch_a |
      | left_leaf_3   | left_branch_b |
      | left_leaf_4   | left_branch_b |
      | right_root    |               |
      | right_leaf_a  | right_root    |
      | right_leaf_b  | right_root    |
      | peer_leaf     |               |
    # Resulting frame:
    #   leaves:         {left_leaf_1, left_leaf_2, left_leaf_3, left_leaf_4,
    #                    right_leaf_a, right_leaf_b, peer_leaf}
    #   internal nodes: {left_root, left_branch_a, left_branch_b, right_root}
    # Descendant sets are distinct per internal node — specificity is
    # well-defined.

  # ────────────────────────────────────────────────────────────────────
  # Failure mode 1 — cross-subtree disagreement at different depths.
  # LLM picks a deep leaf in one subtree; the independent sources
  # (cosine + pattern + name_match) carry mass for a tag in a
  # competing subtree.  At the default revisit-mass threshold the
  # disagreement is silently absorbed; lowering the threshold surfaces
  # it.  The non-regression case asserts the lowering does not invent
  # disagreements when the indep tier supports the LLM's call.
  # ────────────────────────────────────────────────────────────────────

  Scenario: Lowering the indep-tier revisit threshold surfaces a deep-vs-shallow cross-subtree conflict
    Given the following per-source classification votes
      | source     | code         | mass |
      | llm        | left_leaf_1  | 0.85 |
      | cosine     | right_root   | 0.20 |
      | pattern    | right_leaf_a | 0.18 |
      | name_match | right_leaf_a | 0.15 |
    # Indep-tier fused top-1 should be right_leaf_a at mass ≈ 0.30 —
    # below 0.45 (silent absorption), above 0.30 (gate fires).
    When the indep_revisit_mass_threshold is 0.45
    Then the bootstrap revisit gate should not fire for the column
    When the indep_revisit_mass_threshold is lowered to 0.25
    Then the bootstrap revisit gate should fire for the column
    # Non-regression: indep tier supports the LLM's call — no revisit
    # should fire even under the lowered threshold.
    When the per-source votes are
      | source     | code        | mass |
      | llm        | left_leaf_1 | 0.90 |
      | cosine     | left_leaf_1 | 0.55 |
      | pattern    | left_leaf_1 | 0.40 |
      | name_match | left_leaf_1 | 0.50 |
    And the indep_revisit_mass_threshold is 0.25
    Then the bootstrap revisit gate should not fire for the column

  # ────────────────────────────────────────────────────────────────────
  # Failure mode 2 — leaf prediction is unsupported, parent would be
  # the epistemically honest answer (Smets 1993 least-commitment).
  # Lowering commit_threshold causes the operator-facing cautious code
  # to promote up the hierarchy.  Non-regression: a confident leaf
  # (Bel ≥ commit_threshold) is not over-coarsened.
  # ────────────────────────────────────────────────────────────────────

  Scenario: Lowering cautious commit threshold promotes the prediction to the supported parent
    Given fused mass on focal elements
      | focal_element_codes            | mass |
      | {left_leaf_1}                  | 0.20 |
      | {left_leaf_1, left_leaf_2}     | 0.50 |
      | Θ                              | 0.30 |
    # Bel(left_leaf_1)   = 0.20
    # Bel(left_branch_a) = 0.20 + 0.50 = 0.70   (covers {left_leaf_1, left_leaf_2})
    # Bel(left_root)     = 0.20 + 0.50 = 0.70   (covers {1,2,3,4}; but only 0.70 mass intersects)
    When the cautious commit_threshold is 0.55
    # At 0.55: leaf is 0.20 (below); left_branch_a is 0.70 (above);
    # left_root is 0.70 (above).  Most specific by descendant-set
    # size: left_branch_a (2 leaves) over left_root (4 leaves).
    Then cautious_promoted_code should equal "left_branch_a"
    When the cautious commit_threshold is lowered to 0.15
    # At 0.15: leaf (0.20) now qualifies and is the most-specific code.
    Then cautious_promoted_code should equal "left_leaf_1"
    # Non-regression: a confident leaf at moderate threshold should
    # remain the cautious code (not promoted up).
    Given fused mass on focal elements
      | focal_element_codes | mass |
      | {right_leaf_a}      | 0.78 |
      | Θ                   | 0.22 |
    When the cautious commit_threshold is 0.55
    Then cautious_promoted_code should equal "right_leaf_a"

  # ────────────────────────────────────────────────────────────────────
  # Failure mode 3 — late-interaction anti-example fires on a parent
  # internal node, and the negative-channel must carve out the entire
  # descendant subtree (Θ \ descendants(parent), not the broken
  # singleton subtraction).  Without the hierarchical fix the carve-out
  # silently collapsed and the competing subtree never gained mass.
  # ────────────────────────────────────────────────────────────────────

  Scenario: Anti-example on an internal node carves the descendant subtree
    Given a late-interaction tag score for "left_branch_a" with positive 0.30 and negative 0.85
    And a late-interaction tag score for "right_leaf_a" with positive 0.60 and negative 0.00
    And a late-interaction tag score for "peer_leaf" with positive 0.20 and negative 0.00
    When I compute the late-interaction mass function over those scores
    # descendants(left_branch_a) = {left_leaf_1, left_leaf_2}
    # complement = frame_leaves \ {left_leaf_1, left_leaf_2}
    #            = {left_leaf_3, left_leaf_4, right_leaf_a, right_leaf_b, peer_leaf}
    Then the mass function should contain a focal element exactly covering codes {left_leaf_3, left_leaf_4, right_leaf_a, right_leaf_b, peer_leaf}
    And the focal element covering codes {left_leaf_3, left_leaf_4, right_leaf_a, right_leaf_b, peer_leaf} should carry strictly positive mass

  # ────────────────────────────────────────────────────────────────────
  # Failure mode 4 — within-subtree sibling-leaf disagreement.  LLM
  # picks leaf_1, ML picks leaf_2; both in left_branch_a.  Hierarchical
  # aggregation routes the mass to the shared parent, and stepping
  # back from a contested leaf to the supported parent is both
  # epistemically honest (Smets least-commitment) *and* the
  # operator-intuitive outcome — an operator-facing prediction at the
  # parent level is more useful than an arbitrary leaf pick when the
  # leaves disagree at comparable mass.
  # ────────────────────────────────────────────────────────────────────

  Scenario: Sibling-leaf disagreement within a subtree promotes the shared parent at moderate cautious threshold
    Given fused mass on focal elements
      | focal_element_codes        | mass |
      | {left_leaf_1}              | 0.25 |
      | {left_leaf_2}              | 0.25 |
      | {left_leaf_1, left_leaf_2} | 0.30 |
      | Θ                          | 0.20 |
    # Bel(left_leaf_1)   = 0.25
    # Bel(left_leaf_2)   = 0.25
    # Bel(left_branch_a) = 0.25 + 0.25 + 0.30 = 0.80
    # Bel(left_root)     = 0.25 + 0.25 + 0.30 = 0.80  (covers a superset)
    When the cautious commit_threshold is 0.55
    # At 0.55: neither leaf (0.25) clears; left_branch_a (0.80) clears;
    # left_root (0.80) clears.  Most-specific is left_branch_a.
    Then cautious_promoted_code should equal "left_branch_a"

  # ────────────────────────────────────────────────────────────────────
  # Failure mode 5 — generic-vs-specific at the same hierarchical depth.
  # The dominant observed pattern in the running bel × gap sweep:
  # ~24% of all wrong_subtree errors trace to the LLM-derivative
  # cluster (LLM + CatBoost-on-LLM-labels) consolidating on a *generic*
  # catch-all leaf under uncertainty, while an independent late-
  # interaction signal (prototype-value MaxSim plus an explicit
  # anti-example on the generic bucket) carries the *specific*
  # domain leaf at the same depth.  Channel-decomposed Dempster +
  # indep-tier revisit gate jointly surface the disagreement.
  #
  # Operational frame in this fixture:
  #   right_leaf_a — the "generic bucket" the dependent-source
  #                  cluster gravitates to under low signal
  #   right_leaf_b — the "specific domain" leaf the operator-
  #                  intended classification should land on
  # Both at depth 2 in the same parent subtree (right_root) — the
  # *structural* property the scenario tests is "same-depth disagreement
  # surfaces through independent channels"; the within-subtree
  # placement here is incidental and the test would behave the same
  # way under any same-depth sibling pair across subtrees.
  # ────────────────────────────────────────────────────────────────────

  Scenario: Independent late-interaction signal surfaces the specific code over an LLM-favoured generic bucket at the same depth
    # Architectural property — late-interaction mass alone.
    # The specific code carries strong positive prototype-value
    # support; the generic bucket carries weak positive evidence plus
    # strong anti-example evidence.  The channel-decomposed Dempster
    # combination produces conflict K against the generic bucket's
    # singleton mass and leaves the specific code's mass intact.
    Given a late-interaction tag score for "right_leaf_a" with positive 0.40 and negative 0.65
    And a late-interaction tag score for "right_leaf_b" with positive 0.70 and negative 0.00
    And a late-interaction tag score for "peer_leaf" with positive 0.10 and negative 0.00
    When I compute the late-interaction mass function over those scores
    # Post-channel-Dempster, the singleton focal element for the
    # specific code should be the top non-Θ focal by mass — the
    # late-interaction channel-decomposed view, on its own, votes the
    # specific code.
    Then the focal element covering codes {right_leaf_b} should be the top non-Θ focal element by mass

    # Operational property — full multi-source fusion.
    # When the LLM-derivative cluster (LLM + CatBoost-on-LLM) votes
    # the generic bucket while the truly independent sources (cosine,
    # pattern, name_match) unanimously vote the specific domain leaf,
    # the indep-tier revisit gate fires at its default threshold —
    # the wrong-subtree call is surfaced for re-evaluation rather than
    # silently absorbed.
    Given the following per-source classification votes
      | source     | code         | mass |
      | llm        | right_leaf_a | 0.75 |
      | catboost   | right_leaf_a | 0.55 |
      | cosine     | right_leaf_b | 0.45 |
      | pattern    | right_leaf_b | 0.40 |
      | name_match | right_leaf_b | 0.35 |
    When the indep_revisit_mass_threshold is 0.45
    Then the bootstrap revisit gate should fire for the column

    # Non-regression: when the LLM-derivative cluster *and* the
    # independent sources all agree on the same code (the easy case),
    # the revisit gate must not fire — no false positives under the
    # same default threshold.
    When the per-source votes are
      | source     | code         | mass |
      | llm        | right_leaf_b | 0.80 |
      | catboost   | right_leaf_b | 0.50 |
      | cosine     | right_leaf_b | 0.55 |
      | pattern    | right_leaf_b | 0.45 |
      | name_match | right_leaf_b | 0.40 |
    And the indep_revisit_mass_threshold is 0.45
    Then the bootstrap revisit gate should not fire for the column
