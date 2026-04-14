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
synth_*.csv + ground_truth.json
        ↓
   _load_synth_data()
        ↓
   ┌────┴────┐
   ↓         ↓
  SVM     CatBoost
   ↓         ↓
 svm.pkl  catboost.cbm
```

### SVM Path

1. Build TF-IDF text from column name + sample values
2. Fit `LinearSVC` with Platt scaling for probability estimates
3. Save to `.pkl`

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

## Train-Eval Cycle

`train_eval_cycle.py` orchestrates the full loop:

1. **Generate** synthetic data from vocabulary
2. **Train** CatBoost + SVM models
3. **Classify** using the trained models
4. **Evaluate** against ground truth

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
| `ml_train.py` | Training orchestrator (SVM + CatBoost) |
| `catboost_classifier.py` | CatBoost with virtual ensemble uncertainty |
| `svm_classifier.py` | TF-IDF + LinearSVC + Platt scaling |
| `train_eval_cycle.py` | Generate → train → classify → evaluate loop |
| `sage.py` | Global SAGE feature importance |
| `shap_explanations.py` | Per-item SHAP attribution |
