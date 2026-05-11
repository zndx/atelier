<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# SVM Classifier: Signals Wholesale Adoption

## What Changed

Replaced atelier's SVM classifier implementation with the signals project's
version of record — the implementation presented to colleagues as a genuinely
independent DST evidence source.

### `src/atelier/classify/svm_classifier.py`

**Before (atelier divergent port):**
- Manual `scipy.sparse.hstack` of TF-IDF matrices
- Dict-based pipeline state management
- No safety guard for small class counts in calibration CV
- No feature importance extraction
- Save/load serialized a dict of components

**After (signals version of record):**
- `sklearn.pipeline.Pipeline` + `FeatureUnion` — proper composition
- `_min_class_count()` — prevents `CalibratedClassifierCV` crash when any
  class has fewer samples than CV folds
- `feature_importances(top_n=20)` — navigates CalibratedClassifierCV →
  LinearSVC to extract `coef_`, averages absolute coefficients across classes,
  cross-references with `FeatureUnion.get_feature_names_out()`
- `is_fitted` property
- Save/load serializes a proper Pipeline object via joblib

### `src/atelier/classify/mass_functions.py`

Updated `svm_to_mass()` docstring to document architectural independence:

> The SVM operates on sparse lexical features (character/word n-grams),
> making it architecturally independent from the dense sentence-transformer
> embedding shared by cosine and CatBoost sources.

No functional change — the mass function itself already matched signals.

## Why

The independence claim is central to the DST evidence fusion architecture.
Atelier's prior SVM was a partial port that diverged in implementation quality
but not in mathematical behavior. Adopting signals wholesale ensures the
codebase matches the version of record.

## Downstream Compatibility

API surface is identical — no changes needed in:
- `ml_inference.py` — `SVMClassifier.load(path)`, `.predict_proba_single(text)`
- `ml_train.py` — `SVMClassifier()`, `.fit()`, `.save()`
- `pipeline.py` — calls through `ml_inference`
- BDD step definitions — test mass functions only

## Verification

- 97 BDD scenarios passed, 0 failed (tier-0)
- Round-trip verified: fit → predict → save → load → predict

## Deferred: Opus-Label SVM Training (Phase 2)

The MC architecture's frontier/subagent model split enables a clean
independence-preserving training path:

- Train SVM on **frontier model (Opus) labels** from stratified importance sampling
- Combine SVM in DST with **subagent model (Sonnet/Haiku)** predictions
- Different models at training time vs. fusion time → genuinely independent

This requires changes to `pipeline.py` and `ml_train.py` — future work.
