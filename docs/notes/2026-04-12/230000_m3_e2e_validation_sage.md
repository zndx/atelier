# M3: E2E Validation + SAGE Feature Importance

Completed all 5 phases of the M3 milestone.

## What Changed

### Phase 1: Evaluation Framework
- `evaluation.py` — Structured `EvaluationReport` dataclass with per-category P/R/F1,
  confusion matrix, hierarchical accuracy (ancestor/descendant tolerance),
  aggregate DST metrics (mean belief, plausibility, conflict)
- Wired into both `pipeline.py` and `bootstrap.py` — writes `evaluation_report.json`
  alongside `classifications.json`
- Added `iteration_metrics` tracking to bootstrap loop (per-iteration K, disagreements, coverage)
- sklearn imported lazily with pure-Python fallback for environments without it

### Phase 2: Synth-Train-Eval Cycle
- `train_eval_cycle.py` — End-to-end orchestrator: generate synth → train CatBoost + SVM →
  classify mock data → evaluate. Used by tier-0 BDD tests.
- `ml_inference.py` — Added `configure_paths()` to override default model file locations
  for testing. `reset()` now clears configured paths too.
- BDD: 2 scenarios verify cycle completion, model training, evidence source coverage,
  accuracy >0.70, CatBoost/SVM evidence on >50% of columns

### Phase 3: Realistic Mock LLM + Bootstrap Convergence
- `mock_llm.py` — `RealisticMockLLMBackend` with configurable accuracy, confusable pairs
  (Email↔URL, Phone↔BankAccount, SSN↔DOB, etc.), revisit correction, seeded RNG
- BDD: 2 scenarios — bootstrap converges with realistic LLM (55% accuracy forcing
  disagreements), revisit phase reduces disagreements across iterations

### Phase 4: SAGE Feature Importance
- `sage.py` — `FeatureMaskModel` wrapping cosine classifier for SAGE, `run_sage_analysis()`
  using `sage-importance` library's `MarginalImputer` + `PermutationEstimator`
- LRU embedding cache for within-batch dedup + repeated text efficiency
- BDD: 1 scenario (tier-1 due to ~13min runtime on CPU) verifying 12 importance values,
  column_name and sample_values non-zero, column_name in top-3

### Phase 5: Accuracy Bars + Wiring + Docs
- Raised pipeline accuracy bar from 0.3 → 0.6, added micro-F1 > 0.55 check
- Consolidated ALL optional dependency groups into core `dependencies` in pyproject.toml
  (agents, llm, bedrock, ml, viz → all core)
- Added `sage-importance>=0.0.6` to core dependencies
- Updated `docs/src/architecture/classification.md` — M3 done, new modules listed,
  evaluation_report.json in build directory

## Test Results
- 17 features passed, 0 failed, 3 skipped
- 63 scenarios passed, 0 failed (59 existing + 4 new tier-0 + 1 tier-1)
- 260 steps passed, 0 failed

## New Files (8)
- `src/atelier/classify/evaluation.py`
- `src/atelier/classify/train_eval_cycle.py`
- `src/atelier/classify/mock_llm.py`
- `src/atelier/classify/sage.py`
- `features/agent/ml_e2e.feature`
- `features/agent/sage.feature`
- `features/agent/step_defs/ml_e2e_steps.py`
- `features/agent/step_defs/sage_steps.py`

## Modified Files (11)
- `src/atelier/classify/pipeline.py` — Wire evaluate_classifications()
- `src/atelier/classify/bootstrap.py` — Wire evaluation + iteration_metrics
- `src/atelier/classify/ml_inference.py` — Add configure_paths()
- `src/atelier/classify/__init__.py` — Export evaluate_classifications
- `features/agent/classification.feature` — Raised accuracy bar, added eval + F1 scenarios
- `features/agent/bootstrap.feature` — Added realistic mock LLM scenarios
- `features/agent/step_defs/classification_steps.py` — Eval + F1 step definitions
- `features/agent/step_defs/bootstrap_steps.py` — RealisticMockLLMBackend steps
- `features/steps/__init__.py` — Added ml_e2e_steps, sage_steps re-exports
- `pyproject.toml` — Consolidated deps, added sage-importance
- `docs/src/architecture/classification.md` — M3 updates

## Next: M4
- SHAP explanations for per-column feature attribution
- Adaptive discounting based on ensemble variance
- Production scaling (async pipeline, Qdrant vector index)
- GPU acceleration for SAGE (leverage local CUDA GPUs)
