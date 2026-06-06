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

## SVM Training (synth-only, with vocab alignment at inference)

The SVM is trained **once** on the synthetic corpus
(`scripts/generate_synth_source.py` → `ml_train.train_svm`), with
TF-IDF char-3-6gram + word-1-2gram features and labels keyed on the
bundled-ontology ICE.* leaves from `synth_generators.GENERATORS`.
At pipeline runtime, the ICE.* predictions are translated into the
user's taxonomy via the cached LLM-mediated alignment in
`atelier.classify.ontology_alignment` (one LLM call per
(vocabulary, model) tuple; result cached on disk under
`build/cache/alignment/`).

```
data/synth/*.csv  +  ICE.* reference labels
        ↓
   train_svm()  (sklearn LinearSVC + TfidfVectorizer)
        ↓
   build/models/svm.pkl   (label space: ICE.* leaves)

────────  pipeline runtime  ──────────────────────

   svm.predict_proba(text)  →  {ICE.X: p, ICE.Y: q, ...}
        ↓
   translate_proba(proba, alignment)   ← from ontology_alignment
        ↓
   {user_code_A: p+q, user_code_B: r, ...}
        ↓
   svm_to_mass(...)  →  BeliefAssignment in user-taxonomy frame
```

> **Historical note** — earlier revisions of this design ran a
> mid-loop `train_svm_on_frontier_labels` (historical function name)
> that retrained the SVM on live LLM labels and hot-swapped the
> result into the active model slot.  That path was excised on 2026-05-04 (commits 8627c2c,
> 5199379, cc59d01) for the source-independence reasons documented
> in `ontology_alignment.py`.  The current design preserves the
> SVM's TF-IDF independence at the feature and label level; the
> only LLM dependency is the per-vocabulary alignment table, which
> is vocabulary-level rather than column-level shared error.  See
> `ontology_alignment.py` module docstring for the full independence
> argument and the BM25-reranker future-work plan.

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

SAGE runs on the directly-LLM-classified sampled subset when MC
sampling is active (representative by stratification design),
reducing computation at scale.

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
| `ml_train.py` | Training orchestrator: synth-only CatBoost + synth-only SVM (ICE.* labels) |
| `catboost_classifier.py` | CatBoost with virtual ensemble uncertainty |
| `svm_classifier.py` | Pipeline+FeatureUnion: dual TF-IDF + LinearSVC + Platt scaling (signals) |
| `train_eval_cycle.py` | Generate → train → classify → evaluate loop |
| `sage.py` | Global SAGE feature importance |
| `shap_explanations.py` | Per-item SHAP attribution |
