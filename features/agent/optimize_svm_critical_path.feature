@agent @tier-1 @slow @optimize
Feature: just optimize svm --fixture produces a trustworthy critical path
  The operator-confidence sequence for the SVM/maxsim critical path on the
  public test-gittables fixture. Read top to bottom it is the efficacy story:
  promotion happened → maxsim covers the known head of the distribution →
  NHSVM carries the tail maxsim misses → the channel is required, not optional.

  Tier-1 / slow: needs the devenv stack (registry DB + ModernBERT + Qdrant)
  and the hermetic `just optimize svm --fixture` build. See
  docs/src/architecture/cco-coverage.md and the fixture PROVENANCE.md.
  Thresholds are calibrated on the first stack run (conservative floors here).

  Background:
    Given the operator has run `just optimize svm --fixture` for "test-gittables"

  Scenario: Promotion is observable — head and collection are current
    Then a current NHSVM head exists for "test-gittables"
    And the maxsim collection for "test-gittables" is current

  Scenario: Known entities retrieve their correct term in maxsim top-3
    # The agent-mediated enrichment is robust enough that covered entities
    # retrieve correctly; absolute floor (fixed-vector retrieval).
    When the held-out fixture entities are classified
    Then recall@3 over the covered subset is at least 0.40

  Scenario: NHSVM carries the tail where maxsim is weak
    # The reason NHSVM is an independent source: it pulls weight exactly where
    # retrieval is weak. Differential assertion (survives retraining).
    When the held-out fixture entities are classified
    Then NHSVM contributes non-trivial mass on the maxsim-weak partition
    And NHSVM mean mass on the maxsim-weak partition is at least its maxsim-strong mass

  Scenario: The critical path is required, not optional
    # The fail-fast default: an absent head is a loud error, never a silent
    # degrade to the legacy TF-IDF path.
    Given no promoted NHSVM head exists for "test-gittables"
    When classification is triggered with svm source "registered"
    Then it fails fast naming the missing head and `just optimize`
