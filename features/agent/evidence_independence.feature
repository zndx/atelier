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
