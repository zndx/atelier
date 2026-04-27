<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# M4: SHAP Explanations, Configurable Discounts, Production Hardening

**Date**: 2026-04-12
**Milestone**: M4
**Status**: Complete — 65 tier-0 scenarios passing (3m51s)

## What Changed

### Phase 1: Configurable Discounts

All 12 DST discount factors are now configurable via HOCON (`classify.discounts.*`).
Previously hardcoded magic numbers in mass function calls.

- **`config/base.conf`**: New `classify.discounts` section with 12 parameters
- **`config.py`**: 14 new HOCON map entries + AtelierConfig fields
- **`mass_functions.py`**: `DiscountConfig` frozen dataclass with `from_cfg()` factory
- **`pipeline.py`**, **`bootstrap.py`**: Thread `DiscountConfig` through all mass function calls

### Phase 2: Config Wiring + Thread Safety

- **`pipeline.py`**, **`bootstrap.py`**: Call `ml_inference.configure_paths(cfg)` before classify loops
- **`embedding.py`**: `threading.Lock` with double-checked locking for lazy model init
- **`ml_inference.py`**: Same pattern for CatBoost and SVM model loading

### Phase 3: SHAP Explanations

Two methods, ported from `signals/src/sigint/shap_analysis.py`:

| Method | Speed | Use Case |
|--------|-------|----------|
| CatBoost TreeSHAP | 0.1s / 50 items | Auto mode (when model loaded) |
| Embedding PermutationSHAP | ~50s/item | Tier-1 explicit only |

- **`shap_explanations.py`**: New (~250 lines) — `ShapResult`, `run_catboost_shap()`, `run_embedding_shap()`, `run_shap_analysis()`
- **Pipeline integration**: SHAP columns (`shap_top{1,2,3}_{name,value}`) added to JSON + parquet output
- **Auto mode only uses TreeSHAP** — PermutationSHAP is too slow for default pipeline runs

### Phase 4: BDD Scenarios

- **`shap.feature`**: Tier-0 CatBoost TreeSHAP scenario (via synth-train-eval), tier-1 PermutationSHAP
- **`classification.feature`**: Configurable discounts scenario
- New step files wired into `features/steps/__init__.py`

## Key Design Decision

**Auto SHAP only uses TreeSHAP**: Initially `method="auto"` fell back to
PermutationSHAP when no CatBoost model was loaded. This caused all pipeline
scenarios to run ~40 minutes. Fixed by making auto mode skip PermutationSHAP —
it's only invoked when explicitly requested (tier-1 or `method="embedding_permutation"`).

## Files Changed

### New (3)
- `src/atelier/classify/shap_explanations.py`
- `features/agent/shap.feature`
- `features/agent/step_defs/shap_steps.py`

### Modified (12)
- `config/base.conf` — discounts + shap sections
- `src/atelier/config.py` — HOCON map + AtelierConfig fields
- `src/atelier/classify/mass_functions.py` — DiscountConfig + parameterized mass functions
- `src/atelier/classify/pipeline.py` — config wiring + SHAP + discounts
- `src/atelier/classify/bootstrap.py` — config wiring + discounts
- `src/atelier/classify/train_eval_cycle.py` — pass discounts
- `src/atelier/classify/embedding.py` — thread safety
- `src/atelier/classify/ml_inference.py` — thread safety
- `src/atelier/classify/__init__.py` — export run_shap_analysis
- `features/agent/classification.feature` — discount scenario
- `features/agent/step_defs/classification_steps.py` — discount steps
- `features/steps/__init__.py` — shap_steps re-export
- `pyproject.toml` — shap>=0.42.0
- `docs/src/architecture/classification.md` — M4 docs
