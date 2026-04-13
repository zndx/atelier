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

| Source | Type | Discount | Status |
|--------|------|----------|--------|
| Cosine similarity | Sentence-transformer (all-MiniLM-L6-v2) | 0.30 | M0 |
| Pattern detection | 8 regex detectors | 0.10 | M0 |
| Name matching | Column name ↔ label/abbrev/common_names | varies | M0 |
| LLM | OpenAI-compatible / Anthropic / Bedrock / Cerebras | 0.10 | M1 |
| CatBoost | Gradient boosted trees (virtual ensembles) | adaptive | M2 |
| SVM | TF-IDF + LinearSVC (Platt scaling) | 0.20 | M2 |

The **discount** controls how much mass goes to Θ (total ignorance). Higher
discount = more conservative = wider belief intervals.

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

Bootstrap path:
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → LLM_SWEEP → VALIDATING ──┐
                                                    ▲                     │
                                                    └─── (disagreements) ─┘
                                                          (converged) ────► CLASSIFYING → FUSING → EVALUATING → CONVERGED → IDLE
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
├── sampler.py           # Hive metadata sampling + mock fixtures
├── synth.py             # Synthetic data generation (17 category generators)
├── svm_classifier.py    # Dual TF-IDF + LinearSVC + Platt scaling
├── catboost_classifier.py # CatBoost with virtual ensemble uncertainty
├── ml_train.py          # Training orchestrator (synth → models)
├── ml_inference.py      # Lazy-loading inference wrappers
├── evaluation.py        # Structured evaluation (per-category P/R/F1, confusion matrix)
├── train_eval_cycle.py  # Synth → train → classify → evaluate orchestrator
├── mock_llm.py          # Realistic mock LLM (confusable pairs, seeded mistakes)
├── sage.py              # SAGE feature importance (permutation-based)
├── pipeline.py          # Single-pass ML orchestration (6 evidence sources)
├── fsm.py               # AgentFSM state machine
└── fixtures/
    ├── mock_annotations.json  # 24-category mock vocabulary
    └── mock_tables.json       # 8 tables, 50 columns with ground truth
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
    ├── classifications.json           # Per-column DST results
    ├── evaluation_report.json         # Per-category P/R/F1, confusion matrix
    └── atelier_embeddings.parquet     # For embedding-atlas
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

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | Cosine + pattern + name match, FSM, pipeline E2E | Done |
| **M0.5** | Schema fix, pignistic probability, HierarchicalClassification | Done |
| **M1** | LLM evidence source, bootstrap convergence loop, LLM↔ML validation | Done |
| **M2** | CatBoost + SVM + synthetic data, 6 evidence sources, Bedrock/Cerebras backends | Done |
| **M3** | Evaluation framework, E2E synth-train-eval, realistic mock LLM, SAGE importance | Done |
| M4 | SHAP explanations, adaptive discounting, production scaling | Planned |
