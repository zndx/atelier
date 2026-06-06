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
| MaxSim | ColBERT v2 per-token multi-vectors, Qdrant native MaxSim late-interaction | 0.20 | `classify.discounts.maxsim` | M0 |
| Pattern detection | 16 regex detectors + post-regex validators | 0.25 | `classify.discounts.pattern_theta` | M0 |
| Name matching | Column name ↔ label/abbrev/common_names | varies | `classify.discounts.name_match_*` | M0 |
| LLM | OpenAI-compatible (incl. Cerebras) / Anthropic / Bedrock | 0.15 | `classify.llm.discount` | M1 |
| CatBoost | Gradient boosted trees (virtual ensembles) | adaptive | `classify.discounts.catboost_*` | M2 |
| SVM | ModernBERT mean-pool → factorized fully-hierarchical NHSVM (registered head) | 0.22 | `classify.discounts.svm` | M2 |

The **MaxSim** source replaced the legacy single-vector cosine channel
(`cosine_to_mass` removed 2026-05-25). It encodes each entity and each
annotation with ColBERT v2 (BERT + 768→128 projection → per-token 128-dim
multi-vectors) and scores them with Qdrant's native MaxSim over a single
`colbert` multi-vector field; `maxsim_bridge.py` converts the top-K hits into a
DST mass. It is **fail-fast**: when enabled but unable to run (no enriched
collection registered, Qdrant unreachable, `qdrant-client` missing) the bridge
raises `MaxSimUnavailable` and the run errors in the FSM — there is **no silent
fallback** to single-vector cosine. (Cosine survives only as the *metric* the
MiniLM single-vector encoder uses for CatBoost/SVM feature embeddings and the
Monte-Carlo pre-classifier; see [Evidence Independence](#evidence-independence).)

The **SVM** source is a ModernBERT-backed, factorized fully-hierarchical NHSVM
(Choi et al. 2015) — see [SVM Architecture](#svm-architecture). The legacy
TF-IDF char/word n-gram + LinearSVC + Platt-scaling path
(`per_vocab_legacy`) remains only as a deprecated emergency-rollback baseline.

The **discount** controls how much mass goes to Θ (total ignorance). Higher
discount = more conservative = wider belief intervals.

Pattern mass is **graduated**: `detect_patterns()` returns a match fraction
(0.0-1.0) per pattern, and `pattern_to_mass()` scales evidence mass by the
average match fraction. A 95% match produces ~3x more mass than a 35% match,
eliminating the binary cliff at the 1/3 detection threshold.

Pattern theta (0.25) is deliberately higher than LLM theta (0.15), so the
LLM cleanly dominates when pattern and LLM evidence conflict — the LLM
considers full context (name, type, values, siblings), while patterns
operate on value structure alone.

### Evidence Independence

Dempster's rule of combination requires **cognitively independent** evidence
sources (Shafer 1976) — each mass function must reflect information not derived
from the other sources being combined. Atelier achieves this through a
combination of architectural separation (distinct feature spaces and training
signals) and **per-source reliability discounting** (Denoeux 2008): where two
sources share a partial upstream dependency, the dependent source is discounted
so Dempster's rule cannot double-count the overlap.

| Source | Feature Space | Training Signal | Independence Basis |
|--------|---------------|-----------------|-------------------|
| Name match | String/lexical | None (deterministic) | Symbolic matching only |
| Pattern | Regex | None (deterministic) | Hand-crafted rules only |
| MaxSim | ColBERT v2 per-token multi-vectors | Pre-trained encoder + enriched annotations | Late-interaction token-level semantics |
| LLM | Semantic (Cerebras/GLM, Claude, or other backend) | Pre-trained weights | In-context classification |
| CatBoost | MiniLM dense embedding (384-dim) + 12 features | Synthetic data (fit-to-LLM) | Gradient-boosted ensemble |
| **SVM** | **ModernBERT mean-pool dense embedding (768-dim)** | **Synthetic corpus** | **Factorized hierarchical max-margin** |

**MaxSim is the independent semantic channel.**  Where the LLM reads a
column holistically in-context and CatBoost compresses a single MiniLM
sentence embedding, MaxSim scores **per-token** late interaction: each
ColBERT query token contributes its maximum cosine against every
annotation token, summed (the operation Qdrant performs natively over
the `colbert` multi-vector field).  This is a structurally different
view of the same text — token-level alignment rather than a single
pooled vector or an autoregressive judgement — and it is the source
whose removal most degrades accuracy (−13.6pp measured when the legacy
single-vector cosine source stood in for it, which is why the fallback
was deleted rather than retained).  Its only upstream coupling is the
offline annotation-enrichment LLM (shared with the SVM's training
corpus and CatBoost's labels), so it carries the `classify.discounts.maxsim
= 0.20` reliability discount rather than being treated as fully naïve.

**The SVM is a dense, hierarchy-aware channel — not a sparse one.**  An
earlier generation of this design made the SVM the *lexical* counterweight
(sparse TF-IDF char/word n-grams) on the theory that it was feature-space
orthogonal to the dense embeddings the other learned sources used.  That
framing no longer holds: the registered SVM operates on **ModernBERT
mean-pool dense embeddings**, the same kind of representation CatBoost
consumes (different encoder, MiniLM vs ModernBERT).  Its independence is
therefore **not** feature-space orthogonality — it comes from a different
training signal (the balanced synthetic corpus, not LLM labels) and a
fundamentally different decision geometry: a **factorized
fully-hierarchical max-margin classifier** in which every hierarchy node —
including non-leaf nodes — is a first-class prediction target with its own
weight vector, and a path score is accumulated over each candidate's
root-to-leaf ancestors.  CatBoost's gradient-boosted trees and the NHSVM's
per-node hyperplanes make different errors on the same embedding; the
synth-vs-LLM training-signal split keeps them from sharing a label-derived
error mode.

#### SVM Architecture

The registered SVM head is a **factorized fully-hierarchical NHSVM**
(Choi et al. 2015; `factorized_nhsvm.py`, promoted via
`src/atelier/registry/nhsvm_head.py`).  The default
`classify.svm.source = "registered"` requires a current head in the
registry matching `(taxonomy_id, encoder)` and fails loudly if absent
(run `just optimize` to train + promote one) — the no-silent-DST-degradation
posture, not a fallback to the legacy path.

```
Column metadata text ("email_addr | user@example.com")
        │
        ▼
    ModernBERT (answerdotai/ModernBERT-base), mean-pool → 768-dim dense vector
        │
        ▼
    Factorized fully-hierarchical NHSVM
    ├── one weight vector wₙ + alpha αₙ per hierarchy node n
    │   (non-leaf nodes are first-class prediction targets)
    └── path score(leaf) = Σ over root→leaf ancestors of ⟨wₙ, x⟩
        │
        ▼
    Calibrated softmax (learned temperature T) over user codes
        │
        ▼
    Probability distribution {user_code: probability}
```

Key implementation details:

- **Native user codes** — the registered head emits the user taxonomy's
  codes **directly**.  There is **no runtime ICE→user alignment step**: the
  head is trained against the user vocabulary, so the legacy
  subsumption/LLM-mediated alignment dance is gone for this path.
- **Non-leaf targets** — because each node carries its own weight vector and
  the score is a path sum, an internal node can be the predicted code when the
  evidence supports a concept family without committing to a specific leaf.
  This is genuine fully-hierarchical prediction, not leaf-only with post-hoc
  roll-up.
- **Calibrated temperature** — a learned softmax temperature (`T`) reshapes
  raw path scores into well-scaled probabilities; the calibrated head lifts
  the SVM's otherwise-low raw mass into a first-class voice in fusion (see the
  `classify.mass_calibration.svm_alpha` note in
  [Configurable Discounts](#configurable-discounts)).

#### Why dense + factorized (the 98.9%-vs-4.3% motivation)

The factorized form is not stylistic — it is the *only* dense form that works.
Drop ModernBERT mean-pool embeddings into the **old Kronecker NHSVM** and top-1
accuracy collapses to **4.3%**, versus **98.9%** for the TF-IDF baseline on the
same task — a structural failure, not a tuning gap (`factorized_nhsvm.py`
module docstring).  Dense embeddings are everywhere-nonzero with bounded
magnitudes, which the Kronecker construction cannot separate; the **factorized**
per-node parameterization is what makes a dense-embedding hierarchical SVM
viable at all.  That failure is precisely *why* the factorized NHSVM exists,
and why the dense head — not the sparse TF-IDF one — is the version of record.

#### Legacy/baseline TF-IDF path

The TF-IDF char/word n-gram + `LinearSVC` (Crammer–Singer) + Platt scaling
(`CalibratedClassifierCV`) + SVD path in `svm_classifier.py` is the
**legacy/baseline** SVM (`classify.svm.source = "per_vocab_legacy"`).  It is a
deprecated emergency-rollback knob, slated for removal; new workflows must not
build on it.  It retrains a fresh per-vocabulary TF-IDF model from enrichment
payloads each run and, lacking a user-trained head, relies on the historical
ICE→user alignment.  It is retained only as the measured 98.9% baseline that
the factorized dense head is benchmarked against — never as "the current SVM".

> **Historical note (2026-05-04 refactor).**  Earlier revisions ran a mid-loop
> `train_svm_on_frontier_labels` (historical function name) that retrained the
> SVM on live LLM labels and hot-swapped the result into the active model slot
> — "M9 incremental SVM retraining" in commit history.  That path was excised
> on 2026-05-04 for source-independence reasons: per-column LLM-label copying
> made the SVM strongly non-distinct with the LLM source under Denoeux 2008.
> SVM is now trained **offline on the synthetic corpus** and promoted through
> the registry; the only residual coupling is the shared offline
> enrichment-LLM upstream (annotation enrichment), structurally identical to
> the MaxSim source's coupling — hence the matching `0.20`–`0.22` discounts.

##### Implementation

- `factorized_nhsvm.py` — the dense factorized fully-hierarchical NHSVM
  (per-node weights + alphas, root-to-leaf path scoring, calibrated softmax
  temperature, native user codes).
- `src/atelier/registry/nhsvm_head.py` — registry promotion/lookup; the
  runtime resolves the current head by `(taxonomy_id, encoder)` under
  `classify.svm.source = "registered"`.
- `svm_classifier.py` — the **legacy/baseline** TF-IDF + `LinearSVC` + Platt +
  SVD path (`per_vocab_legacy`), kept only as the deprecated rollback knob and
  the accuracy baseline.
- Discount: `classify.discounts.svm = 0.22` reflects the offline-trained
  regime — weakly non-distinct only via the shared enrichment-LLM upstream.

### Dempster's Rule of Combination

Sources are fused via the conjunctive combination rule:

```
m₁₂(C) = Σ{m₁(A)·m₂(B) : A∩B=C} / (1 - K)
```

where `K = Σ{m₁(A)·m₂(B) : A∩B=∅}` is the **conflict** between sources.

High K means the sources disagree — a valuable diagnostic signal. Note that
K is **not** the convergence criterion — see [Belief-Gap Convergence](#belief-gap-convergence)
below.

### Compound Focal Elements (Uncertainty Representation)

When DST evidence splits closely between two singleton categories,
collapsing to a single top-1 prediction misrepresents what the evidence
actually says.  DST's native vocabulary for this is the **compound focal
element**: a portion of the runner-up's mass transfers to a focal
element representing the *union* of the two singletons, honestly
reflecting that the evidence supports the disjunction but does not
discriminate between members.  This is the same DST math that supports
queries at any node in the hierarchy via `belief_at()` — the compound
mass propagates up to the common ancestor, so belief at any level
reflects the combined evidence.

The mechanism is unconditional DST: any two singletons whose masses
split closely qualify in principle.  In practice the implementation
maintains a short registry of category pairs where the transfer is
routinely activated — examples below, filtered to vocabulary at
runtime.  These are **illustrations of cases where the mechanism
activates**, not a definitional list of categories the classifier is
expected to "confuse".

| Example pair | Why mass-splitting is common |
|---|---|
| Record Identifier ↔ Device Identifier | Both are opaque identifiers; context determines which |
| Timestamp ↔ Date of Birth | Both are temporal; DOB is a specific semantic subtype |
| Transaction Amount ↔ Bank Account Number | Both are financial numbers |
| IP Address ↔ Device Identifier | IP addresses can identify devices |

**Mechanics**: when the top-2 singleton masses match a registered pair
and their ratio is below `confusable_ratio_threshold` (default 3.0),
half of the runner-up's mass transfers to the compound focal element.
Belief at the common ancestor then reflects the combined evidence via
`belief_at()` propagation.  (The config knob retains its historical
name for backward compatibility; the mechanism itself is honest
uncertainty representation, not pair-discrimination.)

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
LLM_SWEEP classifies the sampled subset only → propagate labels to remainder
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
├── embedding.py         # MiniLM single-vector encoder (CatBoost/SVM feature
│                        #   embeddings + MC pre-classifier; cosine is the metric
│                        #   here, NOT a DST source)
├── colbert_encoder.py   # ColBERT v2 encoder: BERT + 768→128 projection →
│                        #   per-token 128-dim multi-vectors (maxsim source)
├── maxsim_bridge.py     # Qdrant native MaxSim → DST mass (maxsim_to_mass);
│                        #   fail-fast MaxSimUnavailable, no single-vector fallback
├── llm_backend.py       # LLM backend factory (Anthropic, OpenAI-compat
│                        #   incl. Cerebras/GLM, Bedrock tool-use)
├── bootstrap.py         # Bootstrap convergence loop (LLM sweep + ML validation)
├── agent_loop.py        # Agent-driven convergence (5 Claude tools)
├── monte_carlo.py       # MC stratified sampling for scale (pre-classify, stratify, select, propagate)
├── gpu.py               # GPU detection + NVIDIA driver symlink (nix+CUDA)
├── sampler.py           # Hive metadata sampling + fixture data loading
├── synth.py             # Synthetic data generation
├── synth_generators.py  # 316+ hand-coded value generators (shared module)
├── synth_registry.py    # Three-layer generator registry (hand-coded > template > inferred)
├── meta_tagging_overlay.py # 130+ META_TO_ICE mappings for meta-tagging alignment
├── factorized_nhsvm.py  # SVM source (default): ModernBERT mean-pool → dense
│                        #   factorized fully-hierarchical NHSVM (Choi 2015);
│                        #   non-leaf targets, path scoring, calibrated softmax.
│                        #   Promoted via ../registry/nhsvm_head.py
├── svm_classifier.py    # LEGACY/baseline only: TF-IDF + LinearSVC + Platt + SVD
│                        #   (per_vocab_legacy; deprecated rollback knob)
├── catboost_classifier.py # CatBoost with virtual ensemble uncertainty
├── ml_train.py          # Training orchestrator (synth → models)
├── ml_inference.py      # Lazy-loading inference wrappers
├── evaluation.py        # Structured evaluation (per-category P/R/F1, confusion matrix)
├── train_eval_cycle.py  # Synth → train → classify → evaluate orchestrator
├── mock_llm.py          # Realistic mock LLM (seeded uncertainty + mass-splitting between close categories)
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
5 tools — `get_conflict_report`, `revisit_columns`, `check_convergence`,
`get_column_detail`, `declare_converged` — to reason about
which columns need re-examination. (An in-loop `retrain_svm` tool was removed
with the M9 SVM-on-LLM-labels retrain; SVM is now trained offline and resolved
from the registry.) The agent sees both gap-based and K-based
metrics and can make nuanced decisions. See [Keystone Agents](./agents.md).

### LLM Backend

`llm_backend.py` provides a factory-pattern abstraction:

- **`OpenAICompatibleBackend`**: For vLLM, GLM-4.7, **Cerebras**, and any
  endpoint implementing the OpenAI chat completions API. Default backend.
  Cerebras has **no dedicated class** — `backend = "cerebras"` is resolved by
  the factory into an `OpenAICompatibleBackend` with Cerebras defaults
  (`base_url=https://api.cerebras.ai/v1`, `model=zai-glm-4.7`).
- **`AnthropicBackend`** / **`AnthropicStructuredBackend`**: For Claude via the
  Anthropic SDK (the latter uses tool-use for structured output).
- **`BedrockBackend`**: For AWS Bedrock via the Converse API.
- **`BedrockStructuredBackend`**: Production default on CAI. Uses
  `invoke_model` with **tool-use** for structured output (`output_config`
  is not supported on Bedrock). When extended thinking is enabled,
  `tool_choice` must be `"auto"` (Anthropic constraint); a text-block
  fallback parser handles this case. Both backends use `region_from_arn()`
  to extract the target region from cross-region inference profile ARNs.
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
        discount = 0.15                # DST discount for LLM mass
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
    maxsim = 0.20                    # MaxSim (ColBERT late-interaction) → Theta mass
    svm = 0.22                       # SVM (factorized NHSVM) → Theta mass
    pattern_theta = 0.25             # Pattern detection → Theta mass (graduated by match fraction)
    name_match_exact = 0.70          # Exact label match singleton mass
    name_match_code = 0.50           # Formal code/abbrev match mass
    name_match_alias = 0.50          # Common name alias match mass
    name_match_overlap = 0.30        # Word overlap match mass
    catboost_base = 0.55             # Adaptive discount base (raised; CatBoost is fit-to-LLM)
    catboost_variance_scale = 1.6    # Variance-to-discount scaling
    catboost_max = 0.75              # Cap on adaptive discount
    catboost_fallback = 0.55         # When no variance available
    confusable_ratio_threshold = 3.0 # Mass-split ratio that triggers compound focal element transfer
}
```

The `maxsim` (`0.20`) and `svm` (`0.22`) discounts are close because both ride
on the same offline enrichment-LLM upstream and nothing else; CatBoost's much
larger base (`0.55`) reflects its fit-to-LLM coupling, so Dempster's rule cannot
let the derivative CatBoost source swallow the genuinely independent MaxSim
signal (Denoeux 2008; see `config/base.conf` for the full calibration note).

> **Rejected legacy key.** `classify.discounts.cosine` is **not** a live key —
> the config loader fails loudly on it (it is in `_LEGACY_MAXSIM_KEYS`, mapped to
> `classify.discounts.maxsim`). Use `maxsim`. A separate post-discount
> magnitude calibration lives under `classify.mass_calibration.*`
> (`maxsim_alpha`, `svm_alpha`, `catboost_alpha`, `llm_alpha`).

Environment variable overrides follow `ATELIER_` + the uppercased HOCON path:
`ATELIER_CLASSIFY_DISCOUNTS_MAXSIM`, `ATELIER_CLASSIFY_DISCOUNTS_SVM`, etc.
(`ATELIER_MASS_CALIBRATION_COSINE_ALPHA` is likewise a rejected legacy name →
`ATELIER_MASS_CALIBRATION_MAXSIM_ALPHA`.)

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
| **M6** | Agent-driven convergence loop (Claude tool-use), synth framework (316+ generators) | Done |
| **M7** | Monte Carlo stratified sampling, label propagation, background SHAP | Done |
| **M8** | GPU acceleration (NVIDIA driver symlink, batch encoding), meta-tagging overlay | Done |
| **M8.5** | SVM signals alignment (Pipeline+FeatureUnion adoption, evidence independence documentation) | Done |
| **M9** | Incremental SVM training on LLM-classified labels (cross-model distillation via MC sampling) — *subsequently excised, see 2026-05-04 historical note above* | Done |
| **M10** | Phase Gate #2 — belief-gap convergence pivot, Cautious-Code Review, TreeSHAP per-feature attribution, reasoning-trace citation analyzer (+9 pts iterative gain), 97.8% phase-gate validation on meta-tagging | Done |
| M11 | MLflow experiment tracking, Hive data source integration | [Proposed](./integrations.md) |
