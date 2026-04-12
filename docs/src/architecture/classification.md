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
| Name matching | Column name ↔ category label/abbrev | varies | M0 |
| CatBoost | Gradient boosted trees | 0.15 | M1 stub |
| SVM | TF-IDF + LinearSVC | 0.20 | M1 stub |

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
IDLE → LOADING_VOCAB → DISCOVERING → SAMPLING → CLASSIFYING → FUSING → EVALUATING → CONVERGED
                                                                                        ↓
                                                                                      IDLE
```

State transitions are persisted to PostgreSQL. The Status page polls
`/api/fsm/status` for live progress updates.

### Module Structure

```
src/atelier/classify/
├── __init__.py          # Public API: run_pipeline(), get_fsm_status()
├── belief.py            # DST core: BeliefAssignment, FocalElement, dempster_combine()
├── mass_functions.py    # 5 evidence→mass converters (3 active, 2 stubs)
├── features.py          # 12 features + 8 pattern detectors
├── taxonomy.py          # ReferenceCategory, HierarchicalCategorySet
├── embedding.py         # Sentence-transformer cosine classifier
├── sampler.py           # Hive metadata sampling + mock fixtures
├── synth.py             # Synthetic data generation (M1)
├── pipeline.py          # End-to-end orchestration
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
├── models/              # Embedding caches
└── results/{run_id}/
    ├── classifications.json           # Per-column DST results
    └── atelier_embeddings.parquet     # For embedding-atlas
```

### Controlled Vocabulary

Loaded from hive `default.annotations` (296 rows, 10 columns):

| Column | Maps to | Purpose |
|--------|---------|---------|
| `id` | code | Hierarchical dot-notation identifier |
| `ontology` | label group | Sensitivity tier |
| `annotation` | label | Tag name |
| `definition` | description | Human-readable text |
| `common_names` | abbrev | Pipe-separated aliases |
| `deprecated` | filter | "yes" = exclude |
| `non_corp`, `emp_contractor`, `individual`, `corp` | sensitivity | Per-role ratings |

## API

### REST Endpoints

- `GET /api/fsm/status` — Current pipeline state + progress
- `POST /api/fsm/start` — Start a classification run
- `GET /api/fsm/runs` — List past runs

### gRPC RPCs

- `GetFSMStatus()` → FSMStatusResponse
- `StartClassification()` → StartClassificationResponse

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M0** | Cosine + pattern + name match, FSM, pipeline E2E | Done |
| M1 | CatBoost + SVM, synthetic data via Agent SDK | Planned |
| M2 | Bootstrap convergence loop, LLM sweep → ML validation | Planned |
| M3 | SAGE importance, SHAP explanations, adaptive discounting | Planned |
| M4 | Production scaling, async pipeline, Qdrant index | Planned |
