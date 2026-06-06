@agent @tier-0
Feature: DST evidence-source independence
  Atelier's classification fusion treats LLM-derivative sources
  (CatBoost fit-to-LLM, the excised M9 SVM-on-LLM-labels retrain) as
  non-distinct evidence per Denoeux 2008.  These scenarios validate
  the three soundness mechanisms: independent-tier consensus
  driving the revisit gate, raised reliability discount on
  derivative sources (Shafer 1976 §11.3), and the pattern-target
  alias resolver that prevents the entire pattern source from
  silently disappearing when a vocabulary uses non-ICE codes.
  See docs/src/architecture/dst-evidence-independence.md.

  # Scenario codes below are drawn from publicly-grounded ontologies:
  # the Atelier ``ICE.*`` namespace (Information Content Entity, per
  # IAO_0000030; see ``src/atelier/classify/fixtures/PROVENANCE.md``)
  # for sensitivity-bearing exemplars, and the Common Core Ontologies
  # (``cco:*``) namespace where a non-ICE example is required to
  # exercise the resolver's namespace-agnostic path.  The mechanisms
  # under test (revisit gate, resolver, provenance) are
  # namespace-agnostic — any code system exercises the same paths.

  Scenario: Revisit fires on independent-tier disagreement when fused prediction matches LLM
    # LLM and the LLM-derivative ML cluster agree on a code, so the
    # legacy gate (llm_code != fused_code) cannot fire.  The
    # cosine + pattern + name_match consensus disagrees at high
    # mass — the soundness condition the indep-tier branch is for.
    Given a BootstrapState with LLM "ICE.NONSENSITIVE" and indep-tier consensus "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT" at mass 0.75
    And the fused ml_prediction also equals "ICE.NONSENSITIVE" with conflict K=0.20
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should appear in the disagreements list

  Scenario: Indep-tier branch is gated by mass threshold
    Given a BootstrapState with LLM "ICE.NONSENSITIVE" and indep-tier consensus "ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT" at mass 0.30
    And the fused ml_prediction also equals "ICE.NONSENSITIVE" with conflict K=0.20
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should not appear in the disagreements list

  Scenario: High-K branch still fires as safety net
    # No indep-tier consensus available — legacy high-K branch carries.
    Given a BootstrapState with LLM "ICE.NONSENSITIVE" and no indep-tier consensus
    And the fused ml_prediction equals "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN" with conflict K=0.85
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should appear in the disagreements list

  Scenario: Pattern map resolves through abbrev when target code is in a non-ICE namespace
    # When a vocabulary adopts ontology classes from outside Atelier's
    # own ICE namespace as governance types (e.g. Common Core
    # Ontologies' ``cco:MoneyTransfer`` published as an Apache Atlas
    # classification type), the resolver maps the canonical pattern
    # to the leaf via its mnemonic abbrev rather than its IRI.  The
    # mechanism is namespace-agnostic — BFO, CCO, DPV, or domain
    # extensions all reach the leaf the same way.
    Given a vocabulary with abbrev "TXNAMT" at code "cco:MoneyTransfer"
    When I resolve the default pattern map against that vocabulary
    Then the resolved map binds "monetary_pattern" to "cco:MoneyTransfer"

  Scenario: Pattern map resolver logs misses without raising
    Given a vocabulary with abbrev "TXNAMT" at code "cco:MoneyTransfer"
    When I resolve the default pattern map against that vocabulary
    Then the resolved map omits patterns whose target abbrev is not in the vocabulary

  Scenario: Ontology priors thread canonical ICE.* metadata through embedding text
    # When a pattern fires, the canonical universal-vocabulary label,
    # description, common-names aliases, and full ontological path are
    # injected into the column embedding text so cosine similarity is
    # anchored to publicly-grounded ontology terms (recognizable to
    # any leading embedding model from training) rather than just
    # the regex name.  See mass_functions.lookup_pattern_ontology and
    # docs/src/architecture/dst-evidence-independence.md.
    Given a column whose values match the monetary pattern
    When I extract features from that column
    Then the ontology_priors list contains a "Transaction Amount" entry
    And the embedding text contains "Transaction Amount"
    And the embedding text contains the alias "amount, payment, price"
    And ablating the "ontology_priors" feature removes those tokens from the embedding text

  Scenario: Ontology priors reach the first-pass LLM prompt
    # Publicly-grounded canonical metadata is fed to the LLM on every
    # batch so it has a translation anchor when the user vocabulary
    # doesn't carry an exact equivalent of a detected pattern (He et
    # al. 2023, ontology alignment via LLMs).  The LLM is explicitly
    # told to translate to the closest fit in the candidate
    # vocabulary; the canonical ICE.* code is never a valid
    # classification target.
    Given a column whose values match the monetary pattern
    When I build the LLM batch user prompt for that column
    Then the prompt contains "Pattern-detected ontology priors"
    And the prompt contains "Transaction Amount"
    And the prompt contains "translate to the closest fit"

  Scenario Outline: Cosine reliability shaping concentrates mass on a clear top-1
    # Haenni & Hartmann 2006 — source reliability α derived from
    # observable quality signals (top-1 absolute similarity, top-1/
    # top-2 margin), with margin-aware allocation that concentrates
    # mass on top-1 instead of diluting it through softmax over
    # hundreds of siblings.  Replaces the static discount=0.30
    # behavior whose softmax compression made cosine unable to
    # carry mass on large vocabularies.
    Given a frame with <vocab_size> singletons
    And cosine similarities with top-1 "<top1>" at <sim1> and top-2 at <sim2>
    When I convert similarities to mass
    Then the top-1 singleton mass is <expected_band>
    And the Theta mass is <theta_band>

    Examples: regimes
      | vocab_size | top1   | sim1 | sim2 | expected_band      | theta_band         |
      | 300        | sharp  | 0.70 | 0.50 | at least 0.55      | at most 0.35       |
      | 300        | clear  | 0.45 | 0.20 | at least 0.55      | at most 0.35       |
      | 300        | ambig  | 0.45 | 0.44 | at most 0.20       | at least 0.45      |
      | 300        | noise  | 0.23 | 0.23 | at most 0.05       | at least 0.85      |

  Scenario: Hierarchical cosine mass aggregates at common-ancestor internal node
    # When cosine top-K candidates cluster within a subtree but no
    # single leaf is decisive, mass should aggregate at the
    # subtree's internal-node focal element rather than diluting
    # across leaves (Shafer 1976 §3, Smets 1990 §6 on refinement).
    # Without this aggregation, cosine's localized signal gets lost
    # in Dempster fusion against a confident LLM vote in a
    # different subtree.
    Given a hierarchy with a "Financial Data" parent over four leaves and an "Internal Non-Sensitive" sibling
    And cosine top-1 is "Salary" at sim 0.50 with three siblings within "Financial Data" at sim 0.46-0.48
    When I convert similarities to mass with hierarchical aggregation
    Then the "Financial Data" internal node carries non-zero mass
    And belief at "Financial Data" exceeds belief at "Internal Non-Sensitive"

  Scenario: cautious_code walks the full hierarchy across subtrees
    # When the LLM votes a leaf in subtree A but cosine evidence
    # localizes to subtree B at sufficient belief, cautious_code
    # must be able to return the subtree-B internal node — not be
    # confined to the predicted leaf's ancestor chain.
    Given a HierarchicalClassification where the LLM voted "0.1" but cosine localizes to "Financial Data"
    When I compute cautious_code at threshold 0.50
    And I list cross_subtree_belief at threshold 0.10
    Then cross_subtree_belief includes "Financial Data" as an internal-node entry
    And cross_subtree_belief includes "Internal Non-Sensitive" as a leaf entry

  Scenario: Evidence string surfaces per-source top-1 codes
    # The evidence summary previously showed only mass values per
    # source (e.g. "cosine=0.65, llm=0.77") which hid leaf-level
    # disagreement.  Now shows source→code(mass) so operators see
    # which code each source voted, making cross-source conflict
    # visible at a glance.
    Given a HierarchicalClassification where the LLM voted "0.1" but cosine localizes to "Financial Data"
    Then the evidence string contains "cosine→"
    And the evidence string contains "llm→"
    And the evidence string contains "→ Internal Non-Sensitive"

  Scenario: Evidence string appends competing-subtree summary when present
    Given a HierarchicalClassification where the LLM voted "0.1" but cosine localizes to "Financial Data"
    Then the evidence string contains "competing"

  Scenario: cross_subtree_belief surfaces top-per-subtree alternatives below the threshold
    # The structured cross-subtree view always includes the highest-
    # belief leaf and internal node from each top-level subtree
    # regardless of the absolute threshold, so operators see the
    # competing subtree even when Dempster fusion compresses its
    # mass below the headline bar.
    Given a HierarchicalClassification where the LLM voted "0.1" but cosine localizes to "Financial Data"
    When I list cross_subtree_belief at threshold 0.50
    Then cross_subtree_belief includes a code from the "1." subtree

  Scenario: LLM vote at an internal-node code lands on the internal-node focal element
    # Apache Atlas treats every node in a classification hierarchy as
    # a first-class tagging target.  When the LLM votes an
    # internal-node (parent) code such as
    # ``ICE.SENSITIVE.PID.CONTACT`` — the publicly-grounded
    # Information Content Entity for "Contact Information" — the
    # ``llm_to_mass`` converter attaches evidence mass to the
    # internal-node focal element directly rather than coercing to
    # a child leaf.  Under Dempster's rule a parent focal element
    # is the union of its leaf descendants (Shafer 1976 §3); the
    # downstream ``cross_subtree_belief`` and ``cautious_promoted_code``
    # selectors then surface the parent belief without any post-hoc
    # aggregation.  See docs/src/architecture/dst-evidence-independence.md.
    Given a hierarchical frame with internal node "ICE.SENSITIVE.PID.CONTACT" over leaves "ICE.SENSITIVE.PID.CONTACT.EMAIL" and "ICE.SENSITIVE.PID.CONTACT.PHONE"
    When the LLM votes internal node "ICE.SENSITIVE.PID.CONTACT" at confidence 0.70
    Then the resulting mass function carries non-zero mass on the internal-node focal element for "ICE.SENSITIVE.PID.CONTACT"
    And no mass is allocated to either leaf singleton from that vote

  Scenario: LLM vote at a degenerate internal node (single leaf descendant) coerces to the leaf
    # When an internal node has exactly one leaf descendant in the
    # frame, the parent and its leaf are extensionally identical
    # under DST (the focal element is the same singleton set).
    # ``llm_to_mass`` collapses the vote to the singleton in this
    # degenerate case to keep the focal-element granularity minimal
    # and avoid spurious internal-node bookkeeping.
    Given a hierarchical frame with internal node "ICE.SENSITIVE.TECHNICAL" over a single leaf "ICE.SENSITIVE.TECHNICAL.IPADDR"
    When the LLM votes internal node "ICE.SENSITIVE.TECHNICAL" at confidence 0.80
    Then the resulting mass function carries singleton mass on "ICE.SENSITIVE.TECHNICAL.IPADDR"

  Scenario: Headline picker selects internal-node parent over a wrong-subtree leaf
    # Atlas-style governance treats every node as a first-class
    # tagging target, including for the *headline* prediction.  When
    # the LLM votes a parent in subtree A (e.g. ``Financial Data``)
    # at high confidence and cosine localizes to a leaf in subtree B
    # (e.g. ``Internal Non-Sensitive``), the parent and the leaf
    # occupy *different* focal elements with their own committed
    # masses.  The picker iterates both singletons and internal-node
    # focal elements, ranks by direct mass m(A) (Smets' least-
    # commitment principle: commit at the level evidence directly
    # supports — Bel(A) would systematically promote ancestors), and
    # returns the parent when its committed mass exceeds the
    # competing leaf's.  Without this, the leaf-only argmax would
    # silently pick the wrong-subtree leaf and bury the parent in
    # ``cross_subtree_belief`` where downstream consumers may not
    # see it.
    Given a frame from the loan hierarchy
    And the LLM votes internal node "1.1.1.1" at confidence 0.90 with discount 0.15
    And cosine localizes to wrong-subtree leaf "0.1" at mass 0.55
    When I fuse those sources into a HierarchicalClassification
    Then the headline predicted_code is the internal node "1.1.1.1"
    And the headline category label is "Financial Data"

  Scenario: Headline picker selects leaf when leaf has its own committed mass within the parent
    # Sanity check — the parent-aware picker must not over-promote.
    # When cosine votes a leaf inside the LLM-voted parent (say
    # ``Salary`` inside ``Financial Data``), Dempster's rule
    # concentrates mass on the leaf via the intersection
    # parent ∩ {leaf} = {leaf}.  The leaf ends up with materially
    # higher direct mass than the parent's residual, and the picker
    # correctly returns the leaf — Smets least-commitment "go as
    # specific as the evidence supports".
    Given a frame from the loan hierarchy
    And the LLM votes internal node "1.1.1.1" at confidence 0.90 with discount 0.15
    And cosine localizes to in-subtree leaf "1.1.1.1.1" at mass 0.65
    When I fuse those sources into a HierarchicalClassification
    Then the headline predicted_code is the leaf "1.1.1.1.1"

  Scenario: Headline picker excludes Theta and depth-0 root from candidacy
    # Theta (frame of total ignorance) and the depth-0 root carry
    # mass under vacuous fusion but are never legitimate tags.  When
    # every source is vacuous (all mass on Theta), the picker falls
    # back to the leaf-pignistic argmax so the headline is still a
    # real category — preserving prior behavior in the no-evidence
    # case while keeping Theta out of the result.
    Given a frame from the loan hierarchy
    And every source is vacuous
    When I fuse those sources into a HierarchicalClassification
    Then the headline predicted_code is a depth-1-or-deeper code
    And the headline predicted_code is not the empty string

  Scenario: cautious_promoted_code retains predicted leaf when belief is sufficient
    # Smets' least-commitment principle: promote only when the
    # predicted leaf's belief is below the commit threshold AND the
    # system flags needs_clarification.
    Given a HierarchicalClassification where the LLM voted "0.1" but cosine localizes to "Financial Data"
    When I compute cautious_promoted_code at commit threshold 0.55
    Then promoted_from is null
    And the rationale mentions "commit_threshold"

  Scenario: Unified residual norm reduces under successful refinement
    # The bootstrap loop is iterative refinement on a belief-
    # assignment vector.  Per Saad 2003 §4.1 the headline diagnostic
    # is the residual norm ‖r(B)‖ + the contraction factor ρ_n =
    # ‖r_{n+1}‖ / ‖r_n‖.  Successful iterations should drop the
    # residual; ρ < 1 indicates a contractive regime; ρ → 1
    # indicates a stalled iteration that warrants strategy change.
    Given a BootstrapState with high gap, conflict, and indep-tier disagreement
    When I record iteration 0 metrics
    And the state improves on the next iteration
    And I record iteration 1 metrics
    Then the residual_norm decreases between iterations
    And the contraction_rate at iteration 1 is below 1.0

  Scenario: Stalled iteration surfaces ρ → 1
    Given a BootstrapState with high gap, conflict, and indep-tier disagreement
    When I record iteration 0 metrics
    And the state does not change
    And I record iteration 1 metrics
    Then the contraction_rate at iteration 1 is approximately 1.0

  Scenario: Per-column trajectory grows with each recorded iteration
    # Per-column residual history complements the corpus-wide
    # IterationMetrics with a column-major view: one snapshot per
    # labeled column per iteration, captured by
    # record_iteration_metrics.  Foundation for per-column ρ,
    # bandit-style revisit ordering, and Aitken Δ² early-stop
    # (Phase B — gated on this trajectory data).
    Given a BootstrapState with three labeled columns
    When I record three successive iteration metrics
    Then state.column_history has an entry for each labeled column
    And each column's snapshot sequence has length 3
    And each column's iteration sequence is contiguous starting at 0

  Scenario: Revisit flag propagates into the trajectory snapshot
    Given a BootstrapState with three labeled columns
    When I record iteration 0 with no revisits
    And I record iteration 1 with column "a" revisited
    Then column "a" snapshot at iteration 1 has revisited=true
    And column "b" snapshot at iteration 1 has revisited=false

  Scenario: Per-column ρ computes from trajectory
    Given a BootstrapState with one column whose gap sequence is 0.40, 0.20, 0.10
    Then column_contraction for that column is approximately 0.5

  Scenario: Per-column ρ returns None when fewer than two snapshots
    Given a BootstrapState with one column and a single iteration recorded
    Then column_contraction for that column is None

  Scenario Outline: Shipped vocabularies carry no customer-derived naming conventions
    # Atelier ships universal_vocabulary.json (BFO/IAO-grounded base)
    # and data/sample/ontology.json (300+ leaf OOTB-sample expansion
    # built by scripts/expand_vocabulary.py).  Both must be free of
    # customer-internal abbrev conventions and numeric encoding —
    # contamination unwound 2026-04-30 after an audit identified that
    # a deprecated mock_annotations.json fixture had carried customer
    # conventions into the universal layer.  See
    # src/atelier/classify/fixtures/PROVENANCE.md for public-source
    # attribution per leaf abbrev.
    When I load the "<vocab>" vocabulary fixture
    Then no abbrev value begins with "C_"
    And the notation field is empty for every entry

    Examples:
      | vocab     |
      | universal |
      | sample    |
