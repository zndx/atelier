<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Synthetic Data & Training

The classification pipeline includes two ML evidence sources — CatBoost and
SVM — that require training data. Atelier generates synthetic training data
from the controlled vocabulary, trains both classifiers, and uses them as
independent evidence sources in DST fusion.

## Synth Generators

`synth_generators.py` is the single source of truth for 316+ hand-coded
value generators shared across the synth framework, sample source generation,
and the registry.

Each generator is a callable `(rng: random.Random) -> str` that produces
realistic values for a category. Examples:

- `EMAIL` → `"j.smith@example.com"`, `"alice.chen@corp.net"`
- `SSN` → `"123-45-6789"` (formatted US Social Security Number)
- `LATITUDE` → `"41.8781"` (valid geographic coordinate)
- `CURRENCY_CODE` → `"USD"`, `"EUR"`, `"JPY"`

## Three-Layer Generator Registry

`synth_registry.py` builds a complete generator set for any vocabulary
through a priority-based registry:

| Priority | Source | Description |
|----------|--------|-------------|
| 1 (highest) | **Hand-coded** | From `GENERATORS` dict in synth_generators.py |
| 2 | **Template** | Real sample values with mild perturbation (±10% numeric jitter, character substitution) |
| 3 (lowest) | **Inferred** | Regex pattern matching on category metadata (description, common_names) |

```python
registry = GeneratorRegistry.from_vocabulary(category_set)
# registry.coverage_summary() → {"hand-coded": 250, "template": 40, "inferred": 26}
```

The registry provides `coverage_report()` and `coverage_summary()` to
identify categories without generators — important for vocabulary expansion.

## Column Name Generation

Synthetic training data deliberately uses diverse column names to prevent
classifiers from relying on name heuristics:

- **Semantic names**: `email_address`, `emailAddress`, `EMAIL_ADDR`
  (snake_case, camelCase, uppercase variants, synonym-based)
- **Opaque names**: `field_42`, `col_abc`, `v_123` (~25% of columns)

This forces the ML models to learn from value patterns and context,
not just column naming conventions.

## ML Training Pipeline

`ml_train.py` orchestrates training for both classifiers:

```
synth_*.csv + reference_labels.json
        ↓
   _load_synth_data()
        ↓
   ┌────┴────┐
   ↓         ↓
  SVM     CatBoost
   ↓         ↓
 svm.pkl  catboost.cbm
```

### SVM Path (Signals Architecture)

The SVM classifier uses the `Pipeline` + `FeatureUnion` composition adopted
wholesale from the [Signals](https://github.com/zndx/signals) project:

1. Build short text from column name + type + sample values via `build_svm_text()`
2. `FeatureUnion` extracts dual TF-IDF features:
   - Character n-grams (3-6, `char_wb` analyzer) — captures subword patterns
   - Word n-grams (1-2) — captures multi-word patterns
3. `CalibratedClassifierCV(LinearSVC, method="sigmoid")` — Platt scaling
   for calibrated probability estimates
4. `_min_class_count()` guard prevents calibration CV crash on small classes
5. Save to `.pkl` + `.classes.json` via joblib

The SVM operates on **sparse lexical features** — architecturally independent
from the dense sentence-transformer embedding used by cosine and CatBoost.
See [Classification Pipeline](./classification.md#evidence-independence) for
the full independence analysis.

### CatBoost Path (GPU-accelerated)

1. Extract 12 features per column via `features.extract_features()`
2. Compute sentence-transformer embeddings (384-dim, GPU batch encoding)
3. Fit `CatBoostColumnClassifier` with:
   - `loss_function="MultiClass"`
   - `posterior_sampling=True` (virtual ensemble uncertainty)
   - `auto_class_weights="Balanced"` (handle imbalanced categories)
4. Save to `.cbm` + `.classes.json`

### Virtual Ensemble Uncertainty

CatBoost's `posterior_sampling=True` enables Bayesian uncertainty
quantification via virtual ensembles. The classifier produces not just
class probabilities but per-class variance estimates. High variance
translates to a higher DST discount factor — uncertain ML predictions
carry less evidential weight in the fusion.

## Incremental SVM Training (M9)

> The SVM trained here is the *incremental SVM* in active-learning
> nomenclature — retrained as new oracle labels accumulate.  The
> labels themselves come from the *frontier-tier* (Opus-class) LLM,
> so "frontier" appears below as a label-source qualifier.  See
> [Pareto Capability Evolution](./pareto-capability-evolution.md) for
> the broader context of why we reserved "frontier" as a noun for the
> Pareto sense going forward.

After the bootstrap LLM sweep, the pipeline has high-quality
frontier-tier labels from the Opus-class model.
`train_svm_on_frontier_labels()` **blends** these with synthetic data
and retrains the incremental SVM progressively:

```
synth_*.csv + frontier-tier LLM labels
        ↓
  train_svm_on_frontier_labels()
        ↓
  ┌──────────────────────────────────────────┐
  │  Synth texts  +  Frontier-tier texts     │
  │  (vocabulary   (corpus-specific          │
  │   coverage)     signal)                  │
  └──────────────┬───────────────────────────┘
                 ↓
         SVMClassifier.fit()
                 ↓
         svm_frontier.pkl    ← filename retained for backward compat
```

### Three-Phase Progressive Retraining

1. **Post-sweep** (always): After the first LLM sweep labels frontier columns,
   retrain immediately so the SVM carries corpus-specific signal into the
   first ML validation pass.

2. **Mid-loop** (during convergence): In the programmatic loop, retrain
   after each revisit iteration that adds ≥10 new frontier-tier labels.
   In the agent-driven loop, the agent calls `retrain_svm` when it judges
   enough new labels have accumulated.

3. **Final** (only if not converged): Last-resort retrain with all accumulated
   labels before the final classification pass. Skipped when already converged
   (the last iteration's model is already in use).

### Why Blend Synth + Frontier-tier Labels

- **Synth data**: Covers all vocabulary categories — ensures the SVM can
  classify categories not present in the live sample
- **Frontier-tier labels**: Corpus-specific patterns — column names, value
  formats, and type distributions that synth generators can't capture
- **Together**: Breadth from synth, depth from frontier-tier signal

### Hot-Swap Mechanism

After retraining, the incremental SVM is hot-swapped via:
1. `ml_inference.reset()` — clears cached models and paths
2. `ml_inference.configure_paths(svm_path=..., catboost_path=...)` — points
   the lazy-loader at the freshly-retrained model

The model file lives at `results_dir/svm_frontier.pkl` (run-specific),
preserving `build/models/svm.pkl` as the synth-trained fallback.

## Train-Eval Cycle

`train_eval_cycle.py` orchestrates the full loop:

1. **Generate** synthetic data from vocabulary
2. **Train** CatBoost + SVM models
3. **Classify** using the trained models
4. **Evaluate** against the curated reference

This runs as part of the classification pipeline when models don't
exist yet, or can be triggered explicitly for experimentation.

## SAGE Feature Importance

`sage.py` computes global feature importance via permutation-based
SAGE values. Each of the 12 discrete features is ablated and the
classification accuracy impact measured:

- High SAGE value = feature is critical for classification
- Low SAGE value = feature adds little discriminative power

SAGE runs on the frontier sample when MC sampling is active
(representative subset), reducing computation at scale.

## SHAP Per-Item Attribution

`shap_explanations.py` provides per-column explanations for why each
column was classified as it was:

| Method | Algorithm | Speed | When Used |
|--------|-----------|-------|-----------|
| CatBoost TreeSHAP | Exact O(TLD) built-in | ~0.1s for 50 items | Auto when CatBoost loaded |
| PermutationSHAP | `shap.PermutationExplainer` | ~50s/item | Explicit request only |

Each classification gains 6 SHAP columns:
`shap_top1_name`, `shap_top1_value`, `shap_top2_name`, `shap_top2_value`,
`shap_top3_name`, `shap_top3_value`.

### Background SHAP

For large corpora, SHAP can run in a background thread while the pipeline
proceeds to EVALUATING. Controlled by the HOCON flag:

```hocon
classify {
  background_analysis = true
  background_analysis = ${?ATELIER_BACKGROUND_ANALYSIS}
}
```

Set to `false` on CAI if background threads cause runtime issues.

## Key Files

| File | Role |
|------|------|
| `synth_generators.py` | 316+ hand-coded value generators |
| `synth_registry.py` | Three-layer registry: hand-coded > template > inferred |
| `synth.py` | Synthetic data generation with diverse column names |
| `ml_train.py` | Training orchestrator: synth-only CatBoost + incremental SVM (synth + frontier-tier labels) |
| `catboost_classifier.py` | CatBoost with virtual ensemble uncertainty |
| `svm_classifier.py` | Pipeline+FeatureUnion: dual TF-IDF + LinearSVC + Platt scaling (signals) |
| `train_eval_cycle.py` | Generate → train → classify → evaluate loop |
| `sage.py` | Global SAGE feature importance |
| `shap_explanations.py` | Per-item SHAP attribution |
