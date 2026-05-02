<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Classification Pipeline

Atelier's core objective: agent-mediated metadata classification using
Dempster-Shafer Theory (DST) to produce belief intervals instead of flat
confidence scores, exposing epistemic uncertainty and source disagreement.

## Terminology — reference-label provenance

Four distinct sources of per-column labels show up in our writeups.
Conflating them is load-bearing error, so we name each explicitly:

| Term | Source | Authority level | Where it appears |
|---|---|---|---|
| **Published benchmark** | External, human-curated labels (SOTAB, GitTables) | Gold standard — memorization-safe check | SOTAB pilot artifacts; `docs/notes/2026-04-19/…phase_gate_2.md` |
| **Curated reference** | Generator-derived (synth pairs an answer-key "reference column" per target) + spot-checked by hand | Definitive for the synthetic corpus; not equivalent to a published benchmark | `build/meta-tagging-clean/curated_reference.csv` |
| **LLM commitment** | A single LLM's pass-1 or pass-2 output | Classifier opinion; not a truth | parquet `llm_code`, `predicted_code` |
| **CatBoost prior** | CatBoost fit to LLM labels, used for revisit enrichment | **Not independent evidence** — it is a compressed self-consensus of the LLM; valuable specifically for rescuing abstentions | parquet `predicted_code` via DST fusion |

An **ablation** (as used in our writeups) is a controlled experiment
that holds most of the pipeline fixed and varies exactly one component
at a time, so changes in accuracy can be attributed to that component
rather than to the combination.

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
| Pattern detection | 16 regex detectors + post-regex validators | 0.25 | `classify.discounts.pattern_theta` | M0 |
| Name matching | Column name ↔ label/abbrev/common_names | varies | `classify.discounts.name_match_*` | M0 |
| LLM | OpenAI-compatible / Anthropic / Bedrock / Cerebras | 0.10 | `classify.llm.discount` | M1 |
| CatBoost | Gradient boosted trees (virtual ensembles) | adaptive | `classify.discounts.catboost_*` | M2 |
| SVM | Dual TF-IDF (char+word n-grams) + LinearSVC (Platt scaling) | 0.20 | `classify.discounts.svm` | M2 |

The **discount** controls how much mass goes to Θ (total ignorance). Higher
discount = more conservative = wider belief intervals.

Pattern mass is **graduated**: `detect_patterns()` returns a match fraction
(0.0-1.0) per pattern, and `pattern_to_mass()` scales evidence mass by the
average match fraction. A 95% match produces ~3x more mass than a 35% match,
eliminating the binary cliff at the 1/3 detection threshold.

Pattern theta (0.25) is deliberately higher than LLM theta (0.10), so the
LLM cleanly dominates when pattern and LLM evidence conflict — the LLM
considers full context (name, type, values, siblings), while patterns
operate on value structure alone.

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

- **Singleton class filtering** — `fit()` drops categories with < 2 training
  examples before `CalibratedClassifierCV`, since `StratifiedKFold` requires
  every class to have >= 2 samples. With 316 categories and few tables, some
  categories inevitably have only one example. Dropped categories are logged
  and still receive predictions from the other 5 DST evidence sources.
- **`_min_class_count()`** — returns the actual minimum (no longer clamped to 2)
- **`feature_importances(top_n)`** — navigates `CalibratedClassifierCV` →
  `LinearSVC` to extract `coef_`, averages absolute coefficients across classes,
  cross-references with `FeatureUnion.get_feature_names_out()` for named
  feature importance
- **`is_fitted`** property for safe state checking before prediction

#### Incremental SVM Training (M9)

> **Terminology note.**  The SVM trained here is the *incremental SVM*
> in active-learning nomenclature — it is retrained as new oracle
> labels accumulate during the bootstrap loop.  The labels feeding it
> come from the *frontier-tier* (Opus-class) LLM that runs the sweep
> and revisit, so "frontier" still appears below as a label-source
> qualifier.  The word "frontier" *as a noun* is reserved for the
> Pareto frontier of pipeline configurations — see
> [Pareto Capability Evolution](./pareto-capability-evolution.md).

The Monte Carlo sampling architecture enables a stronger training signal for
the SVM without breaking independence. After the bootstrap LLM sweep, the
SVM is **retrained on blended synth + frontier-tier labels** — high-quality
classifications from the Opus-tier model on the stratified importance sample.

```
_llm_sweep() → frontier columns get Opus labels
     ↓
  RETRAIN #1: Blend synth data + frontier-tier labels
  Incremental SVM hot-swapped before first ML validation
     ↓
_run_ml_validation() — uses incremental SVM
     ↓
  Convergence loop:
    Agent path: agent calls retrain_svm tool when it judges
                enough new labels have accumulated
    Programmatic path: retrain after each revisit iteration
                       that adds ≥10 new frontier-tier labels
     ↓
  RETRAIN #3 (final): Only if NOT converged
     ↓
  CLASSIFYING — final pass uses best available SVM
```

**Blending** ensures categories not in the frontier sample still have
coverage from synth data (broad vocabulary), while corpus-specific patterns
dominate via frontier-tier signal (depth).

**Independence is preserved** because:
- Training signal: Opus (frontier model, used in LLM sweep)
- Bulk LLM source in DST fusion: Sonnet/Haiku (subagent model)
- SVM feature space: sparse TF-IDF (orthogonal to all other sources)

The three independence axes:
1. Different models at training time (Opus) vs. fusion time (Sonnet/Haiku)
2. Different feature spaces (sparse TF-IDF vs. semantic LLM reasoning)
3. Different inductive biases (maximum-margin classifier vs. autoregressive LM)

The incremental SVM becomes the **transmission mechanism** for
frontier-tier-quality signal — MC sampling bounds the Opus cost; the
SVM amortizes Opus's accuracy across the entire table-space.

##### Configuration

```hocon
classify.bootstrap {
  frontier_svm_retrain = true    # Enable/disable frontier retraining
  frontier_svm_min_labels = 20   # Minimum frontier labels to trigger retrain
}
```

##### Implementation

- `train_svm_on_frontier_labels()` in `ml_train.py` — collects
  frontier-tier labels (`label_source in ("llm", "llm_revisit")`),
  blends with synth data, trains the incremental `SVMClassifier`,
  saves to `results_dir/svm_frontier.pkl`
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

High K means the sources disagree — a valuable diagnostic signal. Note that
K is **not** the convergence criterion — see [Belief-Gap Convergence](#belief-gap-convergence)
below.

### Confusable Pairs

When DST evidence splits between two known-confusing categories, mass is
redistributed from the runner-up singleton to a **compound focal element**
representing the pair. This captures honest ambiguity instead of forcing a
singleton prediction that may be wrong.

Four confusable pairs are active (filtered to vocabulary at runtime):

| Pair | Rationale |
|------|-----------|
| Record Identifier ↔ Device Identifier | Both are opaque identifiers; context determines which |
| Timestamp ↔ Date of Birth | Both are temporal; DOB is a specific semantic subtype |
| Transaction Amount ↔ Bank Account Number | Both are financial numbers |
| IP Address ↔ Device Identifier | IP addresses can identify devices |

**Mechanics**: When the top-2 singleton masses form a known pair and their
ratio is below `confusable_ratio_threshold` (default 3.0), half of the
runner-up's mass transfers to the pair focal element. The pair's mass
propagates up the hierarchy via `belief_at()` — Bel at the common ancestor
reflects the combined evidence.

### Pattern Validation

Pattern detection uses a two-stage architecture: 16 regex patterns for
**recall**, plus a `_VALIDATORS` registry for **precision**. A value must
pass both the regex AND the validator (if one exists) to count.

| Validator | Pattern | Checks |
|-----------|---------|--------|
| `_luhn_check` | `credit_card_pattern` | Luhn checksum (ISO/IEC 7812) |
| `_is_valid_ipv4` | `ipv4_pattern` | All 4 octets in 0-255 range |
| `_is_plausible_date` | `date_iso_pattern`, `datetime_iso_pattern` | Month 01-12, day 01-31 |
| `_is_iso_currency` | `iso_currency_pattern` | ISO 4217 whitelist (~40 codes) |

The `phone_pattern` uses a **suppression mechanism**: when a more specific
digit-heavy pattern also fires (SSN, date, credit card, IP, postal code,
monetary, IBAN), the phone match is suppressed. This prevents the phone
regex from injecting false evidence on columns whose values happen to
contain formatted digits.

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
├── features.py          # 12 features + 16 pattern detectors + 5 post-regex validators
├── taxonomy.py          # ReferenceCategory, HierarchicalCategorySet
├── embedding.py         # Sentence-transformer cosine classifier
├── llm_backend.py       # LLM backend factory (Anthropic, OpenAI-compat, Bedrock tool-use, Cerebras)
├── bootstrap.py         # Bootstrap convergence loop (LLM sweep + ML validation)
├── agent_loop.py        # Agent-driven convergence (6 Claude tools)
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
│   └── fixture_tables.json        # 8 tables, 50 cols — fixture reference for unit tests
│                                    (NOT the UAT-corpus curated reference; see
│                                    build/meta-tagging-clean/curated_reference.csv)
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
  via the configured fusion strategy, ranks by pignistic probability

Confidence is **pignistic probability** `BetP(singleton)`, the decision-theoretic
transform that distributes multi-element focal set mass equally among members.

### Fusion Strategies

Two DST combination rules are implemented, selectable via `classify.fusion_strategy`:

- **`dempster`** (default) — Classical Dempster's rule with `(1-K)` normalization.
  Under high conflict, surviving singletons are amplified.
- **`yager`** — Yager's modified rule. Conflict mass is redirected to Θ
  (ignorance) instead of being normalized away. Preserves epistemic honesty
  at the cost of higher ignorance mass and typically lower peak belief values.
  When `K=0`, produces identical results to Dempster.

Yager is available as an opt-in alternative for empirical validation.
The default (Dempster) remains in place pending A/B comparison on real
pipeline runs — Yager's increased conservatism may or may not improve
overall classification quality, and compensatory adjustments to per-source
discounting or decision thresholds may be needed.

## Bootstrap Convergence Loop

The bootstrap pipeline wraps the single-pass ML pipeline in an iterative
LLM↔ML convergence loop. It adds LLM evidence and repeats until
predictions are **settled** — measured by belief-gap convergence, not
raw conflict K.

### Three Phases

1. **LLM Sweep** (`LLM_SWEEP`): Batch-classify all columns via the configured
   LLM backend (Claude via Bedrock/Anthropic, or any OpenAI-compatible endpoint).
   Columns are sent in table-aware batches with sibling context. If every batch
   fails, the sweep raises `RuntimeError` (fail-fast) instead of silently
   proceeding with zero labels.

2. **ML Validation** (`VALIDATING`): Run the full 6-source DST pipeline for
   each column. Compute per-column belief interval `[Bel, Pl]`, conflict K,
   and uncertainty gap `Pl - Bel`. Identify **uncertain columns** where
   predictions need revisiting.

3. **Targeted Revisit** (back to `LLM_SWEEP`): Re-classify uncertain columns
   with enriched context — the ML prediction, belief interval, pattern signals,
   and value descriptions are included in the prompt. This gives the LLM
   evidence it didn't have in the first pass.

### Belief-Gap Convergence

The primary convergence measure is the **uncertainty gap** `Pl - Bel` for
each column's predicted category. This directly answers "how settled is this
prediction?" — unlike K, which only measures source disagreement.

A column can have K=0.9 but Bel=0.95 — the sources fought hard during
combination, but the normalizing denominator `(1-K)` concentrated surviving
mass on the agreed-upon singleton. That column's prediction is **settled**
despite high conflict; it doesn't need revisiting.

**Convergence criteria** (all must hold):

| Criterion | Metric | Default | Meaning |
|-----------|--------|---------|---------|
| **Primary** | `mean_gap < gap_threshold` | 0.15 | Predictions are tight |
| **Secondary** | `frac_unclear < clarity_target` | 0.10 | At most 10% of columns need clarification |
| **Coverage** | `coverage >= coverage_target` | 0.95 | 95% of columns have labels |

**Revisit targeting**: `_identify_uncertain_columns()` selects columns
where `gap > 0.3` OR `Bel < bel_floor` (default 0.50), sorted by gap
descending (most uncertain first).

**Early stopping**: The proof-of-progress paradigm monitors the gap trend.
When mean gap plateaus for 2 consecutive iterations (no verifiable progress),
the loop stops even if the threshold hasn't been reached.

### K as Diagnostic

Conflict K remains in logs, iteration metrics, and agent tools as a
diagnostic for **source disagreement**. It is useful for identifying
calibration issues (e.g., a pattern detector producing false positives)
but does not gate convergence. The cumulative K formula
`K = 1 - Π(1 - Kᵢ)` tends to be high (~0.5-0.8) with 6 partially
correlated sources; this is expected and does not indicate poor quality.

### Agent-Driven Convergence

As an alternative to the programmatic loop, the agent convergence loop
(`agent_loop.py`) delegates revisit strategy to Claude. The agent uses
6 tools — `get_conflict_report`, `revisit_columns`, `check_convergence`,
`get_column_detail`, `retrain_svm`, `declare_converged` — to reason about
which columns need re-examination. The agent sees both gap-based and K-based
metrics and can make nuanced decisions. See [Keystone Agents](./agents.md).

### LLM Backend

`llm_backend.py` provides a factory-pattern abstraction:

- **`OpenAICompatibleBackend`**: For vLLM, GLM-4.7, and any endpoint
  implementing the OpenAI chat completions API. Default backend.
- **`AnthropicBackend`**: For Claude via the Anthropic SDK.
- **`BedrockBackend`**: For AWS Bedrock via the Converse API.
- **`BedrockStructuredBackend`**: Production default on CAI. Uses
  `invoke_model` with **tool-use** for structured output (`output_config`
  is not supported on Bedrock). When extended thinking is enabled,
  `tool_choice` must be `"auto"` (Anthropic constraint); a text-block
  fallback parser handles this case. Both backends use `region_from_arn()`
  to extract the target region from cross-region inference profile ARNs.
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
        backend = "openai_compatible"  # or "anthropic", "bedrock_structured"
        model = "glm-4.7"
        base_url = null                # vLLM endpoint URL
        columns_per_call = 50
        discount = 0.10                # DST discount for LLM mass
    }
    bootstrap {
        max_iterations = 5
        k_threshold = 0.2              # diagnostic (not convergence-gating)
        coverage_target = 0.95
        max_total_llm_calls = 5000
        # Belief-gap convergence (primary criteria)
        gap_threshold = 0.15           # mean(Pl - Bel) target
        clarity_target = 0.10          # max fraction of unclear columns
        bel_floor = 0.50               # min belief for "settled"
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
    pattern_theta = 0.25             # Pattern detection → Theta mass (graduated by match fraction)
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
| **M9** | Incremental SVM training on frontier-tier labels (cross-model distillation via MC sampling) | Done |
| **M10** | Phase Gate #2 — belief-gap convergence pivot, Cautious-Code Review, TreeSHAP per-feature attribution, reasoning-trace citation analyzer (+9 pts iterative gain), 97.8% phase-gate validation on meta-tagging | Done |
| M11 | MLflow experiment tracking, Hive data source integration | [Proposed](./integrations.md) |
