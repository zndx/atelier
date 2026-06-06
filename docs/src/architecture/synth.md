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
| 3 (lowest) | **Inferred** | Regex keyword-matching on category metadata (description, common_names) to *select an existing value generator* — offline synth only; not a classification detector, never enters the runtime pattern library |

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

### SVM Path (factorized fully-hierarchical NHSVM)

The SVM evidence source is a **ModernBERT mean-pool → factorized
fully-hierarchical NHSVM** (Choi et al. 2015). It is the *shipped* SVM
channel: `classify.svm.source = "registered"` is the default, and the
pipeline loads a promoted head from the NHSVM head registry
(`_ensure_registered_svm_head` → `registry/nhsvm_head.py`) or **fails loud**
if none is current (no silent degradation).

1. Encode short column text (name + type + sample values) with
   **`answerdotai/ModernBERT-base`**, mean-pooled to a 768-dim **dense**
   embedding (`factorized_nhsvm.py`).
2. Score with the factorized NHSVM head: one learnable weight vector
   `W_n ∈ ℝ^d` **per hierarchy node** plus a frozen per-node `alpha_n`
   scalar. The path score for a candidate code `y` sums node scores over
   its root-to-leaf ancestors:
   `γ(x, y) = Σ_{n ∈ A_y} alpha_n · (W_nᵀ x)`. Implemented as a single
   `(n_nodes, d)` linear layer with the structural prior baked into a
   precomputed `M_alpha` (path-indicator × diag(alpha)) matrix.
3. **Non-leaf nodes are first-class prediction targets** — an authentic
   fully-hierarchical classifier, not a flat leaf classifier with a
   hierarchy post-hoc.
4. A calibrated softmax temperature (`classify.svm.nhsvm_temperature`)
   produces calibrated path probabilities.
5. The head is trained **offline on the synthetic corpus** and emits the
   **user's taxonomy codes natively** (no runtime ICE→user alignment step).
   Heads are promoted into the registry via
   `atelier.registry.nhsvm_head.promote_to_current` and saved as
   `svm.pkl` + `svm.classes.json`.

Why dense + factorized (the motivation): the *original* NHSVM formulation
materialized the explicit Kronecker product `φ(x, y) = √alpha_y · (x ⊗ e_y)`
and fit a LinearSVC over it. That form works on sparse TF-IDF features but
**catastrophically fails on dense embeddings** — measured **98.93% top-1
on TF-IDF vs 4.26% on naïve dense** ModernBERT (`factorized_nhsvm.py`
docstring). The factorized per-node form exists precisely to make dense
ModernBERT embeddings viable as the hierarchical SVM evidence source.

> **Legacy / baseline path (`per_vocab_legacy`).** The earlier SVM —
> sparse dual TF-IDF (char 3-6 `char_wb` + word 1-2) `FeatureUnion` →
> `CalibratedClassifierCV(LinearSVC, method="sigmoid")` (Platt scaling) +
> SVD, adopted from the [Signals](https://github.com/zndx/signals)
> project (`svm_classifier.py`, `build_svm_text()`) — survives **only**
> as a DEPRECATED emergency-rollback knob (`classify.svm.source =
> "per_vocab_legacy"`, and the `auto` fallback). It is **not** the
> current SVM source and is slated for removal. See
> [Classification Pipeline](./classification.md#evidence-independence) for
> the full independence analysis; DST source independence is enforced via
> per-source discounts (Denoeux 2008), not feature-space orthogonality.

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

## SVM Training (synth-only, native user codes)

The registered NHSVM head is trained **once, offline** on the synthetic
corpus, with **ModernBERT mean-pool dense embeddings** as features and
labels keyed directly on the **user's taxonomy nodes** (leaves *and*
non-leaf nodes). It emits user codes natively, so there is **no runtime
ICE→user alignment step**: at pipeline runtime the head's calibrated path
probabilities flow straight into `nhsvm_to_mass`, which applies the
Choi et al. (2015) tree-distance reweighting before the standard
`svm_to_mass` conversion and produces a `BeliefAssignment` in the user's
taxonomy frame.

```
data/synth/*.csv  +  user-taxonomy reference labels
        ↓
   train (offline) → factorized NHSVM head
   (ModernBERT-base mean-pool → per-node W + alpha; M_alpha path prior)
        ↓
   promote_to_current()  →  nhsvm_head_registry
        ↓
   build/models/svm.pkl  +  svm.classes.json   (label space: user codes)

────────  pipeline runtime  ──────────────────────

   source="registered" → _ensure_registered_svm_head() loads the head
        ↓
   head.predict_proba(text)  →  {user_code_A: p, user_code_B: q, ...}
        ↓
   nhsvm_to_mass(proba, alphas, temperature, distance_matrix)
        ↓
   BeliefAssignment in user-taxonomy frame
```

> **Legacy / baseline training path.** The deprecated `per_vocab_legacy`
> mode retrains a fresh per-vocabulary **TF-IDF LinearSVC** model each run
> from enrichment payloads (`sklearn LinearSVC + TfidfVectorizer`, ICE.*
> labels translated through `ontology_alignment.translate_proba`). It is
> an emergency-rollback knob only — not the shipped source — and is slated
> for removal once the TF-IDF path is retired.

> **Historical note** — earlier revisions of this design ran a
> mid-loop `train_svm_on_frontier_labels` (historical function name)
> that retrained the SVM on live LLM labels and hot-swapped the
> result into the active model slot.  That path was excised on 2026-05-04 (commits 8627c2c,
> 5199379, cc59d01) for the source-independence reasons documented
> in `ontology_alignment.py`.  Per-source independence is now enforced
> by per-source DST discounts (Denoeux 2008): the registered NHSVM is a
> distinct ModernBERT-embedding-fed model whose only shared dependency on
> the LLM channel is the offline synthetic training corpus, not
> column-level labels.

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
| `ml_train.py` | Training orchestrator: synth-only CatBoost + synth-only SVM (native user codes) |
| `catboost_classifier.py` | CatBoost with virtual ensemble uncertainty |
| `factorized_nhsvm.py` | Shipped SVM source: ModernBERT mean-pool → factorized fully-hierarchical NHSVM (per-node W + alpha, path scores, non-leaf nodes first-class) |
| `registry/nhsvm_head.py` | NHSVM head registry (`get_current` / `promote_to_current`) backing `classify.svm.source="registered"` |
| `svm_classifier.py` | **Legacy / baseline** (`per_vocab_legacy`): dual TF-IDF + LinearSVC + Platt scaling (signals); deprecated, slated for removal |
| `train_eval_cycle.py` | Generate → train → classify → evaluate loop |
| `sage.py` | Global SAGE feature importance |
| `shap_explanations.py` | Per-item SHAP attribution |
