@agent @tier-0
Feature: DST evidence-source independence
  Atelier's classification fusion treats LLM-derivative sources
  (CatBoost fit-to-LLM, frontier SVM trained on LLM labels) as
  non-distinct evidence per Denoeux 2008.  These scenarios validate
  the three soundness mechanisms: independent-tier consensus
  driving the revisit gate, raised reliability discount on
  derivative sources (Shafer 1976 §11.3), and the pattern-target
  alias resolver that prevents the entire pattern source from
  silently disappearing when a vocabulary uses non-ICE codes.
  See docs/src/architecture/dst-evidence-independence.md.

  # Scenario codes below use a fictitious ``acme.*`` namespace so the
  # BDD source carries no customer-derived encoding (audit
  # 2026-04-30).  The mechanisms under test (revisit gate, resolver,
  # provenance) are namespace-agnostic — any non-ICE code system
  # exercises the same paths.

  Scenario: Revisit fires on independent-tier disagreement when fused prediction matches LLM
    # LLM and the LLM-derivative ML cluster agree on a code, so the
    # legacy gate (llm_code != fused_code) cannot fire.  The
    # cosine + pattern + name_match consensus disagrees at high
    # mass — the soundness condition the indep-tier branch is for.
    Given a BootstrapState with LLM "acme.misc" and indep-tier consensus "acme.fin.txn" at mass 0.75
    And the fused ml_prediction also equals "acme.misc" with conflict K=0.20
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should appear in the disagreements list

  Scenario: Indep-tier branch is gated by mass threshold
    Given a BootstrapState with LLM "acme.misc" and indep-tier consensus "acme.fin.txn" at mass 0.30
    And the fused ml_prediction also equals "acme.misc" with conflict K=0.20
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should not appear in the disagreements list

  Scenario: High-K branch still fires as safety net
    # No indep-tier consensus available — legacy high-K branch carries.
    Given a BootstrapState with LLM "acme.misc" and no indep-tier consensus
    And the fused ml_prediction equals "acme.account.balance" with conflict K=0.85
    When I call _identify_disagreements with k_threshold 0.30 and indep_revisit_mass_threshold 0.45
    Then the column should appear in the disagreements list

  Scenario: Pattern map resolves through abbrev when target code differs from default
    # When a vocabulary uses a code system that doesn't match the
    # static ICE.* pattern targets (e.g. dotted-string customer codes,
    # numeric encodings, domain-specific schemes), the resolver maps
    # through the leaf mnemonic.  The fictitious ``acme.fin.txn`` here
    # exercises the same path that any non-ICE customer vocabulary
    # would — the resolver is namespace-agnostic.
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    When I resolve the default pattern map against that vocabulary
    Then the resolved map binds "monetary_pattern" to "acme.fin.txn"

  Scenario: Pattern map resolver logs misses without raising
    Given a fictitious vocabulary with abbrev "TXNAMT" at code "acme.fin.txn"
    When I resolve the default pattern map against that vocabulary
    Then the resolved map omits patterns whose target abbrev is not in the vocabulary

  Scenario: Ontology priors thread canonical ICE.* metadata through embedding text
    # When a pattern fires, the canonical universal-vocabulary label,
    # description, common-names aliases, and full ontological path are
    # injected into the column embedding text so cosine similarity is
    # anchored to publicly-grounded ontology terms (recognizable to
    # any frontier embedding model from training) rather than just
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
