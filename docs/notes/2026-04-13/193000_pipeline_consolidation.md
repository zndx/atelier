# Pipeline Consolidation: Bootstrap Convergence as the One True Pipeline

## Summary

Consolidated the two parallel classification pipelines into a single entry
point (`run_classification_pipeline`) that always uses the bootstrap
convergence loop.  The LLM is a required evidence source — the pipeline
refuses to start without LLM credentials unless `use_mock=True`.

## Changes

### Core Pipeline (`src/atelier/classify/pipeline.py`)
- Rewrote `run_classification_pipeline()` to incorporate the full
  convergence loop: LLM_SWEEP → VALIDATING → targeted revisit loop
- Added `llm_backend` parameter for test injection
- Pipeline resolves LLM backend early (before FSM state creation)
  and raises ValueError when no credentials are configured
- Mock mode: when `use_mock=True`, creates `RealisticMockLLMBackend`
  from ground truth labels in the sample data
- Convergence criteria: coverage >= 0.95 AND mean K < 0.2
- Added bootstrap metrics to result summary (iterations, LLM calls,
  tokens, mean_k, coverage, iteration_metrics)

### Bootstrap Helpers (`src/atelier/classify/bootstrap.py`)
- Deleted `run_bootstrap_pipeline()` — all orchestration now in pipeline.py
- Deleted `BootstrapResult` dataclass (unused)
- Cleaned imports — module now only exports phase helpers and state types
- `_run_ml_validation` uses deferred import of `_classify_column` to
  avoid circular import with pipeline.py

### Unified `_classify_column` (from previous task)
- Single function with 6 evidence sources (name, pattern, cosine, LLM,
  CatBoost, SVM)
- `llm_code=None` only for offline seed preparation — docstring reflects
  that the pipeline always supplies LLM evidence

### Feature Analysis (`_run_feature_analysis`)
- Extracted from inline SHAP block into standalone function
- Added SAGE (global feature importance) alongside SHAP (per-item)
- Both gated by config: `classify_shap_enabled`, `classify_sage_enabled`

### Config (`config/base.conf`, `src/atelier/config.py`)
- Added `classify.sage.enabled` and `classify.sage.permutations`
- `has_classify_llm` property already existed

### Gateway (`src/atelier/gateway.py`)
- `/api/fsm/start`: auto-detects mock mode based on `cfg.has_classify_llm`;
  removed double-run bug (no longer calls `fsm.start_run()` before pipeline)
- `/api/fsm/start-bootstrap`: deprecated — redirects to `fsm_start()`
- `/api/status`: added `has_classify_llm` to config section

### Service (`src/atelier/service.py`)
- Fixed double-run bug in `StartClassification` handler
- Pipeline owns run creation — service no longer calls `fsm.start_run()`

### `__init__.py`
- `run_bootstrap()` emits DeprecationWarning, delegates to `run_pipeline()`
- Removed `run_bootstrap_pipeline` import

### UI (`ui/src/pages/Status.tsx`)
- Added `has_classify_llm` to ConfigInfo interface
- Classification Pipeline card shows "LLM Backend: Configured/Mock mode"
- Added convergence metrics: LLM Labeled, Mean K, Disagreements, Iteration,
  Coverage, LLM Calls
- Added LLM_SWEEP and VALIDATING to FSM state color map
- Configuration card shows "Classify LLM: Configured/Not set (mock mode)"

### Embeddings (`ui/src/pages/Embeddings.tsx`)
- `defaultChartsConfig` sets `predicted_label` as default color column

### Seed Preparation (`scripts/prepare_gittables_sample.py`)
- Added `--classify` flag (default: True) that runs atelier's own ML-only
  classification to produce `predicted_label` values
- `--no-classify` falls back to copying from tag_label/gt_code
- Labels come from atelier's evidence fusion (honest baseline)

### BDD + Train/Eval
- Fixed all `run_bootstrap_pipeline` references in step definitions
  and `train_eval_cycle.py` to use `run_classification_pipeline`
- Fixed `category_set_override` → `category_set` parameter name

## Verification
- UI builds cleanly (`pnpm build`)
- BDD tier-0: 0 import errors, 0 assertion failures
- All non-slow tests pass; slow ML train-eval tests running (unrelated)
