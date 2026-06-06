# PROPOSED SPEC — not yet wired. Lives in docs/notes/ until the step defs,
# the `just optimize svm --fixture --skip-agent` fast mode, and the
# committed GitTables fixture assets exist; then it moves to
# features/agent/optimize_svm_critical_path.feature.

@agent @tier-1 @slow @optimize
Feature: just optimize produces a trustworthy NHSVM critical path
  The SVM (modernBERT+NHSVM) channel is a required member of the DST
  evidence mix (CatBoost + MaxSim + modernBERT-NHSVM) and defaults to
  fail-fast: a pipeline run before `just optimize` has trained and
  promoted a head errors loudly rather than degrading
  (see feedback_required_critical_path_defaults).

  These scenarios are the operator's confidence sequence for trusting
  that `just optimize` executed properly — each Then is an observation a
  user would make to be convinced, not a mechanism check (see
  feedback_bdd_efficacy_as_operator_confidence). Read top to bottom they
  tell the efficacy story: promotion happened → MaxSim covers the known
  head of the distribution → NHSVM carries the tail MaxSim misses → the
  channel is required, not optional.

  Everything is exercised on a controlled, PUBLIC, non-target-domain
  corpus — a committed sample of the GitTables semantic-type taxonomy
  (DBpedia/Schema.org-grounded; see fixtures/test-gittables/PROVENANCE.md
  for per-type attribution). Because the taxa and the target-domain
  entities are fixture-controlled, covered terms have robust committed
  enrichment and the MaxSim-weak tail is deliberately constructed.

  Isolation: the head, the taxonomy_registry row, and the Qdrant
  collection are all keyed by taxonomy_id "test-gittables" and cannot
  collide with the production "default" namespace. Threshold discipline:
  fixed-vector MaxSim retrieval over committed enrichment is
  deterministic → absolute floors; the trained NHSVM head carries bounded
  nondeterminism → relative/differential assertions only.

  Background:
    Given the committed GitTables fixture for taxonomy "test-gittables"
    And the operator has run `just optimize svm --fixture --skip-agent`

  Scenario: Promotion is observable — a current head exists for the test taxonomy
    # First link in the confidence chain: the operator confirms the run
    # actually promoted a head, isolated to the test taxonomy.
    When the operator lists the nhsvm_head_registry for taxonomy "test-gittables"
    Then exactly one head has status "current"
    And its encoder is "answerdotai/ModernBERT-base"
    And no head was promoted for taxonomy "default"

  Scenario: Known domain entities retrieve their correct term in MaxSim top-3
    # The confidence test for the agent-mediated reference + augmented
    # Qdrant entries: for entities the taxonomy covers, late-interaction
    # retrieval alone returns the right term in the top-3. Read directly
    # from the per-column maxsim_attribution.top_k in classifications.json.
    # Absolute floor: retrieval over committed enrichment is deterministic.
    When the operator classifies the held-out fixture entities for "test-gittables"
    Then for every entity whose true term is in the covered subset,
         the correct term appears in the MaxSim top-3
    And recall@3 over the covered subset is at least 0.90

  Scenario: NHSVM carries the tail — meaningful mass where MaxSim is weak
    # The reason NHSVM is an independent DST source at all: it must pull
    # weight exactly where retrieval is weak. Partition the held-out
    # entities by MaxSim top-3 strength (read from maxsim_attribution),
    # then read each source's contribution from
    # HierarchicalClassification.source_masses. Differential assertion —
    # tests the complementarity property, not a magic number, so it
    # survives head retraining. This scenario is the standing guard on the
    # MaxSim/NHSVM independence claim (Denoeux 2008): if the common-mode
    # LLM-enrichment dependency ever collapses NHSVM into a MaxSim echo,
    # this goes red.
    When the operator classifies the held-out fixture entities for "test-gittables"
    And partitions them into MaxSim-weak and MaxSim-strong by top-3 strength
    Then on the MaxSim-weak partition, NHSVM source_mass is non-trivial for each entity
    And NHSVM's mean source_mass on the MaxSim-weak partition
        exceeds its mean source_mass on the MaxSim-strong partition
    And the fused classification recovers at least 3 MaxSim-weak entities
        that MaxSim-alone misranked

  Scenario: The critical path is required, not optional
    # Encodes the fail-fast default: an absent head is a loud error with
    # remediation guidance, never a silent degrade to the legacy TF-IDF
    # path or a missing source.
    Given no promoted head exists for taxonomy "test-gittables"
    When the operator triggers classification with classify.svm.source "registered"
    Then the run fails fast
    And the error names the missing head and instructs running `just optimize`
