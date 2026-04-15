# Classification Pipeline

Atelier's core objective: agent-mediated metadata classification using
Dempster-Shafer Theory (DST) to produce belief intervals instead of flat
confidence scores, exposing epistemic uncertainty and source disagreement.

## Methodology

### Why Dempster-Shafer?

Traditional classifiers output a single confidence score (e.g., "85% email
address"). This hides two distinct types of uncertainty:

- **Aleatoric uncertainty**: inherent randomness in the data
- **Epistemic uncertainty**: ignorance due to insufficient evidence

DST separates these via **belief intervals** `[Bel(A), Pl(A)]`:

- `Bel(A)` = committed evidence supporting A (lower bound)
- `Pl(A)` = evidence that cannot rule out A (upper bound)
- `Pl(A) - Bel(A)` = unresolved ambiguity

When `Bel(A) = 0.8` and `Pl(A) = 0.85`, we have high confidence with low
ambiguity. When `Bel(A) = 0.3` and `Pl(A) = 0.9`, we know something
supports A but much remains uncertain — a signal to gather more evidence.

### Evidence Sources

Each source independently produces a **mass function** (Basic Probability
Assignment) that distributes belief across the frame of discernment:

| Source | Type | Discount | Configurable | Status |
|--------|------|----------|--------------|--------|
| Cosine similarity | Sentence-transformer (all-MiniLM-L6-v2) | 0.30 | `classify.discounts.cosine` | M0 |
| Pattern detection | 15 regex detectors | 0.10 | `classify.discounts.pattern_theta` | M0 |
| Name matching | Column name ↔ label/abbrev/common_names | varies | `classify.discounts.name_match_*` | M0 |
| LLM | OpenAI-compatible / Anthropic / Bedrock / Cerebras | 0.10 | `classify.llm.discount` | M1 |
| CatBoost | Gradient boosted trees (virtual ensembles) | adaptive | `classify.discounts.catboost_*` | M2 |
| SVM | Dual TF-IDF (char+word n-grams) + LinearSVC (Platt scaling) | 0.20 | `classify.discounts.svm` | M2 |

The **discount** controls how much mass goes to Θ (total ignorance). Higher
discount = more conservative = wider belief intervals.

### Evidence Independence

Dempster's rule of combination requires **cognitively independent** evidence
sources (Shafer 1976) — each mass function must reflect information not derived
from the other sources being combined. Atelier achieves this through
architectural separation of feature spaces and training signals:

| Source | Feature Space | Training Signal | Independence Basis |
|--------|---------------|-----------------|-------------------|
| Name match | String/lexical | None (deterministic) | Symbolic matching only |
| Pattern | Regex | None (deterministic) | Hand-crafted rules only |
| Cosine | Dense embedding (384-dim) | Pre-trained sentence-transformer | Learned semantic similarity |
| LLM | Semantic (frontier or subagent model) | Pre-trained weights | In-context classification |
| CatBoost | Dense embedding + 12 features | Synthetic data generators | Gradient-boosted ensemble |
| **SVM** | **Sparse TF-IDF (char 3-6 + word 1-2 n-grams)** | **Synthetic data generators** | **Lexical surface patterns** |

The SVM is architecturally the most important independence guarantee. While
cosine similarity and CatBoost both operate on the same dense
sentence-transformer embedding (384 dimensions from `all-MiniLM-L6-v2`), the
SVM operates on an entirely orthogonal feature representation: **sparse TF-IDF
character and word n-grams** extracted by `sklearn.pipeline.Pipeline` +
`FeatureUnion`. This means the SVM captures lexical surface patterns
(abbreviations, digit sequences, camelCase fragments) that the dense embedding
may collapse — providing genuine corrective signal in DST fusion.

#### SVM Architecture (adopted from Signals)

The SVM classifier follows the `Pipeline` + `FeatureUnion` composition pattern
from the [Signals](https://github.com/zndx/signals) project — the version of
record presented as an independent fifth DST evidence source:

```
Column metadata text ("email_addr | user@example.com")
        │
        ▼
    FeatureUnion
    ├── TfidfVectorizer(analyzer="char_wb", ngram_range=(3,6))
    │   → captures subword patterns, abbreviations, digit sequences
    └── TfidfVectorizer(analyzer="word", ngram_range=(1,2))
        → captures multi-word patterns ("email address", "zip code")
        │
        ▼
    Sparse feature matrix (up to 100K dimensions)
        │
        ▼
    CalibratedClassifierCV(LinearSVC, method="sigmoid")
        │
        ▼
    Calibrated probability distribution {code: probability}
```

Key implementation details:

- **`_min_class_count()`** — prevents `CalibratedClassifierCV` crash when any
  class has fewer samples than CV folds
- **`feature_importances(top_n)`** — navigates `CalibratedClassifierCV` →
  `LinearSVC` to extract `coef_`, averages absolute coefficients across classes,
  cross-references with `FeatureUnion.get_feature_names_out()` for named
  feature importance
- **`is_fitted`** property for safe state checking before prediction

#### Frontier-Label SVM Training (M9)

The Monte Carlo sampling architecture enables a stronger training signal for
the SVM without breaking independence. After the bootstrap LLM sweep, the
SVM is **retrained on blended synth + frontier labels** — high-quality
classifications from the Opus-tier model on the stratified importance sample.

```
_llm_sweep() → frontier columns get Opus labels
     ↓
  RETRAIN #1: Blend synth data + frontier labels
  SVM hot-swapped before first ML validation
     ↓
_run_ml_validation() — uses frontier-trained SVM
     ↓
  Convergence loop:
    Agent path: agent calls retrain_svm tool when it judges
                enough new labels have accumulated
    Programmatic path: retrain after each revisit iteration
                       that adds ≥10 new frontier labels
     ↓
  RETRAIN #3 (final): Only if NOT converged
     ↓
  CLASSIFYING — final pass uses best available SVM
```

**Blending** ensures categories not in the frontier sample still have
coverage from synth data (broad vocabulary), while corpus-specific patterns
dominate via frontier signal (depth).

**Independence is preserved** because:
- Training signal: Opus (frontier model, used in LLM sweep)
- Bulk LLM source in DST fusion: Sonnet/Haiku (subagent model)
- SVM feature space: sparse TF-IDF (orthogonal to all other sources)

The three independence axes:
1. Different models at training time (Opus) vs. fusion time (Sonnet/Haiku)
2. Different feature spaces (sparse TF-IDF vs. semantic LLM reasoning)
3. Different inductive biases (maximum-margin classifier vs. autoregressive LM)

The SVM becomes the **transmission mechanism** for frontier-quality signal —
MC sampling bounds the Opus cost; the SVM amortizes Opus's accuracy across
the entire table-space.

##### Configuration

```hocon
classify.bootstrap {
  frontier_svm_retrain = true    # Enable/disable frontier retraining
  frontier_svm_min_labels = 20   # Minimum frontier labels to trigger retrain
}
```

##### Implementation

- `train_svm_on_frontier_labels()` in `ml_train.py` — collects frontier
  labels (`label_source in ("llm", "llm_revisit")`), blends with synth data,
  trains `SVMClassifier`, saves to `results_dir/svm_frontier.pkl`
- `_maybe_retrain_svm()` in `pipeline.py` — encapsulates retrain + hot-swap
  via `ml_inference.reset()` + `configure_paths()`
- Three call sites in pipeline: post-sweep, iterative, final (if not converged)
- Agent tool `retrain_svm` for agent-driven convergence path

### Dempster's Rule of Combination

Sources are fused via the conjunctive combination rule:

```
m₁₂(C) = Σ{m₁(A)·m₂(B) : A∩B=C} / (1 - K)
```

where `K = Σ{m₁(A)·m₂(B) : A∩B=∅}` is the **conflict** between sources.

High K means the sources disagree — a valuable diagnostic signal.

### 12 Discrete Features

Each column produces 12 SAGE-ablatable features:

1. `column_name` — humanized column name
2. `column_type` — SQL type (suppresses uninformative STRING/VARCHAR)
3. `sample_values` — first 5 non-null values as text
4. `cardinality` — distinct value count
5. `null_ratio` — fraction of NULL values
6. `value_entropy` — Shannon entropy of value lengths
7. `pattern_signals` — matched regex patterns
8. `avg_value_length` — mean string length
9. `numeric_ratio` — fraction parseable as numbers
10. `sibling_context` — other column names in the same table
11. `source_table` — table name
12. `value_description` — auto-generated natural language description

## Architecture

### AgentFSM

The classification pipeline runs as a background Finite State Machine:

```
ML-only path:
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → CLASSIFYING → FUSING → EVALUATING → CONVERGED → IDLE

Bootstrap path (programmatic):
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → LLM_SWEEP → VALIDATING ──┐
                                                    ▲                     │
                                                    └─── (disagreements) ─┘
                                                          (converged) ────► CLASSIFYING → FUSING → EVALUATING → CONVERGED → IDLE

Agent-driven path:
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → LLM_SWEEP → VALIDATING
                                                    ▲           │
                                                    └── Agent convergence loop (5 tools)
                                                          Claude reasons about which columns to revisit
                                                          (converged) ────► CLASSIFYING → FUSING → EVALUATING → CONVERGED → IDLE

MC sampling (when corpus > 200 columns):
SAMPLING includes pre-classify → stratify → select MC sample
LLM_SWEEP classifies frontier columns only → propagate labels to remainder
```

State transitions are persisted to PostgreSQL. The Status page polls
`/api/fsm/status` for live progress updates.

### Module Structure

```
src/atelier/classify/
├── __init__.py          # Public API: run_pipeline(), run_bootstrap(), get_fsm_status()
├── belief.py            # DST core: BeliefAssignment, FocalElement, dempster_combine()
├── mass_functions.py    # Evidence→mass converters (6 active)
├── features.py          # 12 features + 8 pattern detectors
├── taxonomy.py          # ReferenceCategory, HierarchicalCategorySet
├── embedding.py         # Sentence-transformer cosine classifier
├── llm_backend.py       # LLM backend factory (Anthropic, OpenAI-compat, Bedrock, Cerebras)
├── bootstrap.py         # Bootstrap convergence loop (LLM sweep + ML validation)
├── agent_loop.py        # Agent-driven convergence (5 Claude tools)
├── monte_carlo.py       # MC stratified sampling for scale (pre-classify, stratify, select, propagate)
├── gpu.py               # GPU detection + NVIDIA driver symlink (nix+CUDA)
├── sampler.py           # Hive metadata sampling + fixture data loading
├── synth.py             # Synthetic data generation
├── synth_generators.py  # 316+ hand-coded value generators (shared module)
├── synth_registry.py    # Three-layer generator registry (hand-coded > template > inferred)
├── meta_tagging_overlay.py # 130+ META_TO_ICE mappings for meta-tagging alignment
├── svm_classifier.py    # Pipeline+FeatureUnion: dual TF-IDF + LinearSVC + Platt scaling (signals)
├── catboost_classifier.py # CatBoost with virtual ensemble uncertainty
├── ml_train.py          # Training orchestrator (synth → models)
├── ml_inference.py      # Lazy-loading inference wrappers
├── evaluation.py        # Structured evaluation (per-category P/R/F1, confusion matrix)
├── train_eval_cycle.py  # Synth → train → classify → evaluate orchestrator
├── mock_llm.py          # Realistic mock LLM (confusable pairs, seeded mistakes)
├── sage.py              # SAGE feature importance (permutation-based, GPU-aware)
├── shap_explanations.py # Per-item SHAP feature attribution (TreeSHAP + PermutationSHAP)
├── pipeline.py          # Full pipeline orchestration (6 sources + MC + background SHAP)
├── fsm.py               # AgentFSM state machine
├── fixtures/
│   ├── universal_vocabulary.json  # BFO-grounded universal vocabulary (16 leaves)
│   └── fixture_tables.json        # 8 tables, 50 columns with ground truth
data/sample/
└── ontology.json                  # Expanded vocabulary (300 leaves, 25 internal)
└── ontology/
    ├── atelier-vocab.ttl          # CCO-mediated BFO alignment (59 mapped terms)
    ├── sparql/unmapped-terms.rq   # Totality validation query
    └── README.md                  # Mapping methodology and usage
```

### Build Directory

Artifacts are written to `build/` (gitignored) to separate reproducible
code from potentially sensitive intermediate data:

```
build/
├── data/annotations/    # Cached vocabulary from hive
├── data/samples/        # Sampled metadata
├── data/synth/          # Synthetic training data
├── models/              # Trained CatBoost + SVM models, embedding caches
└── results/{run_id}/
    ├── classifications.json           # Per-column DST results (+ SHAP columns when enabled)
    ├── evaluation_report.json         # Per-category P/R/F1, confusion matrix
    └── atelier_embeddings.parquet     # For embedding-atlas (+ shap_top{1,2,3}_{name,value})
```

### Controlled Vocabulary

Loaded from hive `default.annotations` (11 columns):

| Column | Maps to | Purpose |
|--------|---------|---------|
| `id` | `code` | Hierarchical dot-notation identifier |
| `ontology` | `label` | Human-readable category name |
| `annotation` | `abbrev` | Formal code / mnemonic |
| `definition` | `description` | Human-readable definition text |
| `common_names` | `common_names` | Pipe/comma-separated aliases |
| `specifics` | (embedding text) | Examples and context |
| `non_corp`, `emp_contractor`, `individual`, `corp` | `sensitivity` | Per-role ratings (0-4) |
| `deprecated` | (filter) | "yes" = exclude |

## API

### REST Endpoints

- `GET /api/fsm/status` — Current pipeline state + progress
- `POST /api/fsm/start` — Start a single-pass ML classification run
- `POST /api/fsm/start-bootstrap` — Start bootstrap convergence loop (LLM + ML)
- `GET /api/fsm/runs` — List past runs

### gRPC RPCs

- `GetFSMStatus()` → FSMStatusResponse
- `StartClassification()` → StartClassificationResponse

## HierarchicalClassification

The pipeline wraps each column result in a `HierarchicalClassification` object
(ported from signals) that enables post-hoc hierarchy navigation:

- `belief_at(code)` — query Bel at any hierarchy level (leaf or internal)
- `plausibility_at(code)` — query Pl at any level
- `interval_at(code)` — `(Bel, Pl)` tuple
- `uncertainty_gap` — `Pl - Bel` for the predicted category
- `needs_clarification` — True when `uncertainty_gap > 0.3` or `conflict > 0.2`
- `from_combined_evidence()` — factory method: filters vacuous sources, combines
  via Dempster's rule, ranks by pignistic probability

Confidence is **pignistic probability** `BetP(singleton)`, the decision-theoretic
transform that distributes multi-element focal set mass equally among members.

## Bootstrap Convergence Loop

The bootstrap pipeline wraps the single-pass ML pipeline in an iterative
LLM↔ML convergence loop. It adds a 4th evidence source (LLM) and repeats
until DST conflict K converges across all columns.

### Three Phases

1. **LLM Sweep** (`LLM_SWEEP`): Batch-classify all columns via the configured
   LLM backend (GLM-4.7 on vLLM, Claude, or any OpenAI-compatible endpoint).
   Columns are sent in table-aware batches with sibling context.

2. **ML Validation** (`VALIDATING`): Run the existing 3-source ML pipeline
   plus the new LLM mass function for each column. Compute per-column conflict
   K from Dempster's rule. Identify disagreements where the LLM and ML top
   predictions differ AND K exceeds the threshold.

3. **Targeted Revisit** (back to `LLM_SWEEP`): Re-classify high-K columns
   with enriched context — the ML prediction, belief interval, and conflict
   score are included in the prompt. This gives the LLM a chance to reconsider
   with evidence it didn't have in the first pass.

### Convergence Criteria

The loop terminates when:
- `coverage >= 0.95` (95% of columns have a label) **AND**
  `mean_k < 0.2` (average conflict is low), or
- Budget exhausted (`max_iterations` or `max_total_llm_calls` reached)

After convergence, the pipeline completes the standard path:
CLASSIFYING → FUSING → EVALUATING → CONVERGED.

### Agent-Driven Convergence

As an alternative to the programmatic loop, the agent convergence loop
(`agent_loop.py`) delegates revisit strategy to Claude. The agent uses
5 tools — `get_conflict_report`, `revisit_columns`, `check_convergence`,
`get_column_detail`, `declare_converged` — to reason about which columns
need re-examination. See [Keystone Agents](./agents.md) for details.

### LLM Backend

`llm_backend.py` provides a factory-pattern abstraction:

- **`OpenAICompatibleBackend`**: For vLLM, GLM-4.7, and any endpoint
  implementing the OpenAI chat completions API. Default backend.
- **`AnthropicBackend`**: For Claude via the Anthropic SDK.
- **`BedrockBackend`**: For AWS Bedrock (production default on CAI).
  Uses `boto3` Converse API with `modelId`-based routing.
- **`CerebrasBackend`**: OpenAI-compatible with Cerebras-specific defaults
  (`base_url=https://api.cerebras.ai/v1`, `model=zai-glm-4.7`).
- **`create_backend_from_cfg(cfg)`**: Factory that reads HOCON config
  to select and configure the appropriate backend.

Backends fail fast when not configured — no mock fallback in production code.

### Configuration

All bootstrap/LLM settings live in HOCON (`config/base.conf`):

```hocon
classify {
    llm {
        backend = "openai_compatible"  # or "anthropic"
        model = "glm-4.7"
        base_url = null                # vLLM endpoint URL
        columns_per_call = 50
        discount = 0.10                # DST discount for LLM mass
    }
    bootstrap {
        max_iterations = 5
        k_threshold = 0.2
        coverage_target = 0.95
        max_total_llm_calls = 5000
    }
}
```

Environment variable overrides follow the standard pattern:
`ATELIER_LLM_MODEL`, `ATELIER_LLM_BASE_URL`, `ATELIER_BOOTSTRAP_K_THRESHOLD`, etc.

## SHAP Explanations

Per-item feature attribution explaining **why** each column was classified as
it was. Complements the global SAGE importance (which ranks features across
the entire dataset) with item-level explanations.

### Two Methods

| Method | Algorithm | Speed | Features | When Used |
|--------|-----------|-------|----------|-----------|
| CatBoost TreeSHAP | Exact O(TLD) built-in | ~0.1s for 50 items | Grouped: embedding, discrete | Auto when CatBoost model loaded |
| Embedding PermutationSHAP | `shap.PermutationExplainer` | ~50s/item on CPU | 12 named features | Tier-1, explicit request only |

**Auto mode** (`method="auto"`) only uses TreeSHAP — PermutationSHAP is too
slow for default pipeline runs and must be explicitly requested.

### Output

Each classification gains 6 extra columns:
- `shap_top1_name`, `shap_top1_value`
- `shap_top2_name`, `shap_top2_value`
- `shap_top3_name`, `shap_top3_value`

These flow through to JSON, parquet, and evaluation output.

### Configuration

```hocon
classify.shap {
    enabled = true        # Enable SHAP in pipeline (auto-selects method)
    top_k = 3             # Number of top features to report per item
}
```

## Configurable Discounts

All DST discount factors are configurable via HOCON. The `DiscountConfig`
dataclass bundles all parameters with `DiscountConfig.from_cfg(cfg)` factory:

```hocon
classify.discounts {
    cosine = 0.30                    # Cosine similarity → Theta mass
    svm = 0.20                       # SVM → Theta mass
    pattern_theta = 0.10             # Pattern detection → Theta mass
    name_match_exact = 0.70          # Exact label match singleton mass
    name_match_code = 0.50           # Formal code/abbrev match mass
    name_match_alias = 0.50          # Common name alias match mass
    name_match_overlap = 0.30        # Word overlap match mass
    catboost_base = 0.10             # Adaptive discount base
    catboost_variance_scale = 1.6    # Variance-to-discount scaling
    catboost_max = 0.50              # Cap on adaptive discount
    catboost_fallback = 0.15         # When no variance available
    confusable_ratio_threshold = 3.0 # CatBoost confusable pair threshold
}
```

Environment variable overrides: `ATELIER_DISCOUNT_COSINE`, `ATELIER_DISCOUNT_SVM`, etc.

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | Cosine + pattern + name match, FSM, pipeline E2E | Done |
| **M0.5** | Schema fix, pignistic probability, HierarchicalClassification | Done |
| **M1** | LLM evidence source, bootstrap convergence loop, LLM↔ML validation | Done |
| **M2** | CatBoost + SVM + synthetic data, 6 evidence sources, Bedrock/Cerebras backends | Done |
| **M3** | Evaluation framework, E2E synth-train-eval, realistic mock LLM, SAGE importance | Done |
| **M4** | SHAP explanations, configurable discounts, thread-safe model loading | Done |
| **M5** | Data sources + versioning, OOTB onboarding (316-leaf ontology, 25 sample tables) | Done |
| **M6** | Agent-driven convergence loop (6 Claude tools), synth framework (316+ generators) | Done |
| **M7** | Monte Carlo stratified sampling, label propagation, background SHAP | Done |
| **M8** | GPU acceleration (NVIDIA driver symlink, batch encoding), meta-tagging overlay | Done |
| **M8.5** | SVM signals alignment (Pipeline+FeatureUnion adoption, evidence independence documentation) | Done |
| **M9** | Frontier-label SVM training (cross-model distillation via MC sampling) | Done |
| M10 | MLflow experiment tracking, Hive data source integration | [Proposed](./integrations.md) |
