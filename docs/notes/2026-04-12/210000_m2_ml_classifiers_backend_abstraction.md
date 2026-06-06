# M2: ML Classifiers + Backend Abstraction

Completed all 5 phases of the M2 milestone.

## What Changed

### Phase 1: Backend Abstraction
- `llm_backend.py` — Added `BedrockBackend` (boto3 Converse API) and Cerebras support
  (OpenAI-compatible with preset `base_url=https://api.cerebras.ai/v1`, model `zai-glm-4.7`)
- Extended `LLMBackendConfig` with AWS credential fields
- Updated `create_backend()` factory to route 4 backends: anthropic, openai_compatible, cerebras, bedrock
- `pyproject.toml` — Added `bedrock` optional dependency group (boto3>=1.34.0)

### Phase 2: Synthetic Data Generation
- `synth.py` — Full rewrite with 17 value generators for all leaf categories
- Generators: email, phone, address, full_name, dob, ssn, credit_card, bank_account,
  amount, ipv4, uuid, url, timestamp, record_id, status, generic_string, internal_code
- Semantic + opaque column name strategies, deterministic with seeded RNG
- Output: CSV files + ground_truth.json (10 tables, 470 columns with seed=42)

### Phase 3: ML Classifiers
- `svm_classifier.py` — Dual TF-IDF (char 3-6 + word 1-2 n-grams) → LinearSVC → CalibratedClassifierCV
- `catboost_classifier.py` — CatBoost with `posterior_sampling=True` for virtual ensemble uncertainty
- `ml_train.py` — Training orchestrator: load synth data → extract features → train → save
- `ml_inference.py` — Lazy-loading inference wrappers with graceful degradation when models absent
- `scripts/train_classifiers.py` — CLI entry point

### Phase 4: Mass Functions + Pipeline Integration
- `mass_functions.py` — Replaced catboost_to_mass and svm_to_mass stubs with real implementations
  - CatBoost: adaptive discount from virtual ensemble variance (0.1 + avg_var * 1.6, capped at 0.5)
  - SVM: fixed discount 0.20
- `pipeline.py` — Added CatBoost (source 4) and SVM (source 5) after cosine
- `bootstrap.py` — Added CatBoost (source 5) and SVM (source 6) after LLM
- `config/base.conf` + `config.py` — Added model path config fields

### Phase 5: BDD + Wiring + Docs
- 9 new BDD scenarios across 3 feature files:
  - `backend.feature` (3): factory routing for cerebras, bedrock, unknown
  - `synth.feature` (2): leaf coverage, deterministic generation
  - `ml_classifiers.feature` (4): CatBoost mass, adaptive discount, SVM mass, vacuous
- Updated `features/steps/__init__.py` with re-exports
- Updated `docs/src/architecture/classification.md` — evidence table, module structure, milestones
- `pyproject.toml` — Added `ml` optional dependency group

## Test Results
- 16 features passed, 0 failed
- 58 scenarios passed, 0 failed (49 existing + 9 new)
- 234 steps passed, 0 failed

## New Files (11)
- `src/atelier/classify/svm_classifier.py`
- `src/atelier/classify/catboost_classifier.py`
- `src/atelier/classify/ml_train.py`
- `src/atelier/classify/ml_inference.py`
- `scripts/train_classifiers.py`
- `features/agent/backend.feature`
- `features/agent/synth.feature`
- `features/agent/ml_classifiers.feature`
- `features/agent/step_defs/backend_steps.py`
- `features/agent/step_defs/synth_steps.py`
- `features/agent/step_defs/ml_steps.py`

## Modified Files (11)
- `src/atelier/classify/llm_backend.py`
- `src/atelier/classify/mass_functions.py`
- `src/atelier/classify/synth.py`
- `src/atelier/classify/pipeline.py`
- `src/atelier/classify/bootstrap.py`
- `src/atelier/classify/__init__.py`
- `config/base.conf`
- `src/atelier/config.py`
- `pyproject.toml`
- `features/steps/__init__.py`
- `docs/src/architecture/classification.md`

## Next: M3
- SAGE feature importance
- SHAP explanations
- Adaptive discounting
- Then deploy to CAI and validate convergence loop with Bedrock + real metadata
