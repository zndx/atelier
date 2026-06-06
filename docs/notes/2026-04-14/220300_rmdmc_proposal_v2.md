# Relational Multi-Dimensional Monte Carlo (R-MDMC) — Revised Proposal

## Status

Draft v2 — updated to reflect Atelier implementation through M8, including the
existing column-level MC sampling (`monte_carlo.py`, committed `772e58b`), 12-feature
extraction pipeline, SAGE/SHAP ablation infrastructure, and the bootstrap convergence
loop. This revision maps every proposed operation to a concrete integration point in the
codebase.

---

## Motivation: The "50 Fetched, 5 Used" Gap

SHAP and SAGE analyses consistently identify `sample_values` as the dominant informative
feature in the Agentic-DST classification pipeline — yet the system currently discards
90%+ of available row data before it reaches any classifier:

| Stage | Rows Available | Rows Used | Waste |
|-------|---------------|-----------|-------|
| Hive sampler (`LIMIT 50`) | 50 | 5 (`.head(5)`) | 90% |
| OOTB CSV (100 rows each) | 100 | 5 (`rows[:5]`) | 95% |
| Fixture JSON | 5 | 5 | 0% |

The 5 selected rows are always positional (first N non-null) — never stratified,
never randomized, never varied between bootstrap iterations. Every pipeline stage
(pre-classify, LLM sweep, ML validation, revisit, label propagation) sees the same
5 values for the entire run. This is the single largest untapped signal in the pipeline.

The existing column-level MC (`monte_carlo.py`) solves the horizontal scaling problem
(which columns get frontier inference). R-MDMC adds the vertical dimension: which
*rows* inform each column's features, and how row diversity interacts with the
convergence loop.

---

## Architecture: Three Nested Dimensions

```
Dimension 0 — Tables (Entities)
  Existing: `discover_tables()`, `sample_table_metadata()`, OOTB `load_sample_source()`
  Extension: importance-weighted table selection via Signals taxonomy
  Scope: O(N_t) tables from corpus

  Dimension 1 — Columns (per table)
    Existing: `monte_carlo.py` — stratify(), select_sample(), propagate_labels()
    Extension: none (M7 implementation is complete)
    Scope: frontier vs. propagation partitioning, O(N_c) columns

    Dimension 2 — Rows (per column, per iteration)    ← NEW
      Current: `values[:5]` hardcoded, never varies
      Extension: row reservoir + stratified iteration + embedding aggregation
      Scope: O(N_r × k) row-samples per bootstrap iteration
```

### Complexity

Cost scales as `O(N_t × N_c × N_r × k)` where:
- `N_t` = tables sampled (existing: up to `discover_tables(limit=100)`)
- `N_c` = frontier columns per MC plan (existing: `max_frontier_columns=500`)
- `N_r` = row-sample iterations per column (NEW: default 3)
- `k` = rows per iteration (NEW: default 10, up from hardcoded 5)

This is independent of total corpus row count and cardinality — the `sample_size=50`
SQL LIMIT bounds the row reservoir regardless of table size.

---

## Integration Map

### Phase 0: Row Reservoir (sampler.py)

**Problem:** `ColumnSample.values` stores exactly 5 rows. The remaining 45 rows from
the Hive `LIMIT 50` query are discarded at `sampler.py:148`.

**Change:** Store all fetched rows in a new `all_values` field; keep `values` as the
current-iteration view.

```python
# sampler.py — ColumnSample extension
@dataclass
class ColumnSample:
    name: str
    column_type: str | None = None
    values: list[str] = field(default_factory=list)      # Current iteration's row sample
    all_values: list[str] = field(default_factory=list)   # Full reservoir (up to sample_size)
    total_count: int = 0
    null_count: int = 0
    # ... existing fields ...
```

**Integration points:**

| File | Line | Current | Change |
|------|------|---------|--------|
| `sampler.py:148` | `.head(5).tolist()` | Store all 50 in `all_values`, first 5 in `values` |
| `sampler.py:304` | `rows[:5]` | Store all rows in `all_values`, first 5 in `values` |
| `sampler.py:222` | fixture `values` | `all_values = values` (fixtures have exactly 5) |

**Backward compatibility:** Every downstream consumer reads `col.values`. The initial
`values = all_values[:5]` preserves identical behavior. Row MC activates only when
explicitly enabled in HOCON.

### Phase 1: Row-Sample Iterator (NEW module: `row_sampler.py`)

A lightweight iterator that selects different row subsets from `all_values` on each
bootstrap iteration. Three strategies, mirroring Signals' `ImpalaSampler`:

| Strategy | Selection Rule | When |
|----------|---------------|------|
| `head` | `all_values[:k]` (default, current behavior) | Baseline / no row MC |
| `stratified` | Quantile-based selection (entropy, length, pattern diversity) | Default when row MC enabled |
| `random` | `random.sample(all_values, k)` with seed = iteration | Fallback |

```python
# row_sampler.py (new)
def select_row_sample(
    col: ColumnSample,
    k: int = 10,
    strategy: str = "stratified",
    iteration: int = 0,
    seed: int = 42,
) -> list[str]:
    """Select k values from col.all_values using the given strategy.

    Updates col.values in-place and returns the selection.
    """
```

**Stratified selection algorithm:**

1. Compute value-length histogram over `all_values`
2. Partition into quantile bins (short / medium / long)
3. Within each bin, prefer values that match different patterns (maximize pattern diversity)
4. Sample proportionally from each bin
5. Deterministic: `seed + iteration` ensures reproducibility with iteration-over-iteration diversity

**Why value-length stratification?** Shannon entropy of value lengths is feature #6
(`value_entropy`). Values with diverse lengths activate different pattern detectors
and produce different value descriptions — maximizing the information yield per sample.

### Phase 2: Feature Extraction Integration (features.py)

**Problem:** `extract_features()` at line 282 accepts `values: list[str]` and uses
`values[:max_values]` (default 5). The `max_values` parameter exists but is never
varied.

**Changes:**

1. **Increase `max_values` default** from 5 to 10 when row MC is active. More values
   produce richer pattern detection (threshold at `len(values) // 3`) and more
   representative entropy/numeric_ratio statistics.

2. **No structural changes** to `ColumnFeatures` or `extract_features()`. The row
   sampler sets `col.values` before feature extraction runs — features.py is agnostic
   to how values were selected.

3. **Embedding text enrichment:** The `sample_values_text` segment in
   `to_embedding_text()` (line 241) naturally includes more values when `max_values`
   increases. This gives the sentence-transformer a richer signal without changing the
   embedding pipeline.

### Phase 3: Embedding Aggregation (embedding.py)

For columns where row diversity significantly affects the embedding (as measured by
SAGE importance of `sample_values`), generate multiple embeddings from different row
subsets and aggregate.

**Multi-view strategy:**

```python
# Within classify_cosine_batch() or a new classify_cosine_multiview()
def classify_cosine_multiview(
    features_per_view: list[list[ColumnFeatures]],  # N_r views × N_columns
    category_set: HierarchicalCategorySet,
) -> list[dict[str, float]]:
    """Classify columns using multiple row-sample views.

    Each view produces a similarity vector. Final similarity is the
    element-wise mean across views (mean pooling).
    """
```

**When to use multi-view vs. single-view:**

- Single-view (current): `sample_values` SAGE importance < 0.2 (column name/type dominate)
- Multi-view: `sample_values` SAGE importance >= 0.2 AND `len(all_values) > k`

This is adaptive: SAGE importance from a prior run (or the first iteration) drives
the decision. Columns where the name is unambiguous ("email_address") don't benefit
from row diversity; columns with generic names ("col0", "field_3") benefit enormously.

### Phase 4: Bootstrap Iteration Integration (pipeline.py, bootstrap.py)

**Current flow (same values every iteration):**
```
iteration 1: extract_features(col.values=[v1,v2,v3,v4,v5]) → classify → K=0.35
iteration 2: extract_features(col.values=[v1,v2,v3,v4,v5]) → classify → K=0.33  ← same input!
iteration 3: extract_features(col.values=[v1,v2,v3,v4,v5]) → classify → K=0.31
```

**Proposed flow (different values each iteration):**
```
iteration 1: select_row_sample(col, iteration=1) → [v1,v2,v3,v4,v5]    → K=0.35
iteration 2: select_row_sample(col, iteration=2) → [v6,v8,v12,v15,v20] → K=0.28
iteration 3: select_row_sample(col, iteration=3) → [v3,v7,v14,v22,v31] → K=0.22
```

**Integration in pipeline.py:**

Before each bootstrap iteration (lines 352-403), call `select_row_sample()` for all
columns participating in that iteration. This gives the LLM and ML classifiers
genuinely new evidence on each pass, rather than rehashing the same 5 values.

```python
# pipeline.py — within the bootstrap iteration loop
for iteration in range(1, max_iterations + 1):
    if row_mc_enabled:
        for name in disagreements:
            col = samples_by_name[name]
            select_row_sample(col, k=row_k, strategy=row_strategy, iteration=iteration)
    # ... existing _llm_revisit, _run_ml_validation, _identify_disagreements
```

**Critical insight:** Only disagreement columns (high-K) get new row samples. Columns
that have already converged keep their existing values — no wasted compute.

### Phase 5: LLM Revisit Enrichment (bootstrap.py, llm_backend.py)

**Current:** `_llm_revisit()` (bootstrap.py:224) enriches the prompt with ML prediction,
belief interval, conflict, and confusable pair — but shows the *same values* the LLM
already saw.

**Proposed:** On revisit, show *different* values from the reservoir. This gives the LLM
genuinely new evidence, not a rehash.

```python
# llm_backend.py:214 — build_batch_user_prompt()
# Current:
if sample.values:
    preview = sample.values[:10]

# Proposed (when row MC active):
if sample.values:
    preview = sample.values[:10]
    if sample.all_values and len(sample.all_values) > len(preview):
        # Show a note about total available rows
        lines.append(f"({len(sample.all_values)} total distinct values sampled)")
```

The actual value rotation happens upstream (Phase 4 sets `col.values` before the
prompt is built), so `llm_backend.py` needs minimal changes — just an awareness
annotation for the LLM to know more data exists.

### Phase 6: Row-Level Epistemic Signal

**New convergence signal:** If different row subsets produce different classifications
for the same column, that's direct evidence of epistemic uncertainty — the column's
type depends on *which* values you look at.

```python
# Track per-column classification stability across row samples
row_stability: dict[str, float]  # column_name → fraction of iterations with same label

# In convergence check:
if row_stability[name] < 0.5:
    # Column classification is row-dependent — high epistemic uncertainty
    # Escalate to frontier model with ALL available values
    col.values = col.all_values
```

This integrates with `should_stop_early()` in bootstrap.py: the K plateau detection
(proof-of-progress paradigm) gains a second signal. A column's K may be stable because
it keeps seeing the same rows — not because the evidence has actually converged.

### Phase 7: MC Label Propagation Enhancement (monte_carlo.py)

**Current:** `propagate_labels()` (line 365) computes embedding similarity using a
single embedding per column (from fixed 5 values).

**Proposed:** When row MC is active, propagation uses the **mean embedding** across
row-sample iterations. This produces more robust similarity scores because:

1. A single 5-value sample may be unrepresentative (positional bias from `.head(5)`)
2. The mean embedding smooths out per-sample noise
3. Outlier values that dominate a single sample's embedding are diluted

```python
# monte_carlo.py — propagate_labels() enhancement
if row_mc_enabled and hasattr(col, 'embedding_views') and col.embedding_views:
    # Mean-pool across row-sample views
    chunk_emb = np.mean(col.embedding_views, axis=0)
else:
    # Single embedding (current behavior)
    chunk_emb = model.encode([feat.to_embedding_text()], ...)[0]
```

---

## Configuration (HOCON)

```hocon
classify {
    row_mc {
        enabled = false                              # Master switch
        enabled = ${?ATELIER_ROW_MC_ENABLED}

        k = 10                                       # Values per iteration (up from 5)
        k = ${?ATELIER_ROW_MC_K}

        strategy = "stratified"                      # head | stratified | random
        strategy = ${?ATELIER_ROW_MC_STRATEGY}

        iterations = 3                               # Row-sample iterations per column
        iterations = ${?ATELIER_ROW_MC_ITERATIONS}

        multiview_threshold = 0.2                    # SAGE importance threshold for multi-view
        multiview_threshold = ${?ATELIER_ROW_MC_MULTIVIEW_THRESHOLD}

        adaptive_escalation = true                   # Escalate row-unstable columns
        adaptive_escalation = ${?ATELIER_ROW_MC_ADAPTIVE}
    }
}
```

**Zero-cost when disabled:** `enabled = false` means `ColumnSample.all_values` is still
populated (no information loss), but `select_row_sample()` is never called — `values`
remains `all_values[:5]` exactly as today.

---

## Ontology Grounding

The R-MDMC extension maintains full compatibility with the CCO-mediated BFO foundation
in `atelier-vocab.ttl`:

- Row-sampled values are `cco:InformationContentEntity` instances — the ontological
  type is invariant to which rows are selected
- The ICE trichotomy (DesignativeICE, DescriptiveICE, PrescriptiveICE) applies to
  column classifications, not individual row values
- Pattern detectors that fire on row subsets produce evidence grounded in the same
  `DEFAULT_PATTERN_MAP` → category code mappings
- SAGE/SHAP feature attributions reference the same 12-feature vector regardless
  of row selection — ablation masks are feature-level, not row-level

The Prudhomme criteria (equivalence, subsumption, conservativity, completeness) are
satisfied because R-MDMC changes *which evidence is observed*, not *how evidence maps
to ontological categories*. The mapping from features → mass functions → DST belief
intervals is unchanged.

---

## Signals Compatibility

R-MDMC outputs (heuristic rules, few-shot prompts, embedding lookups) are reusable
across the Signals GitTables taxonomy:

| Signals Component | R-MDMC Integration |
|-------------------|-------------------|
| `ImpalaSampler` strategies (head/random/frequent) | `row_sampler.py` mirrors these at application level |
| SAGE `FeatureMaskModel` (feature index matrices) | Row-sample views extend the index to a 3rd dimension |
| Bootstrap agent (conflict-driven revisit) | Row diversity becomes a new convergence signal |
| Schema discovery (Data Elements) | Row correlation patterns could feed Data Element detection |
| RASE evidence model (Gaius) | Propagated heuristics carry row-stability metadata |

---

## Scaling Projections

| Corpus Size | Tables | Columns | Rows/Col | Row MC Cost | Total Embeddings |
|-------------|--------|---------|----------|-------------|-----------------|
| OOTB (current) | 25 | 300 | 100 | 3 × 300 × 10 | 9,000 |
| Small enterprise | 500 | 5K | 50 | 3 × 750 × 10 | 22,500 |
| GitTables subset | 10K | 150K | 50 | 3 × 500 × 10 | 15,000* |
| Full GitTables | 1.7M | 15M | 50 | 3 × 500 × 10 | 15,000* |

*MC column sampling caps frontier at 500 regardless of corpus size. Row MC applies
only to frontier columns.

**GPU scaling:** At 2,768 embeddings/sec (GPU batch), 15,000 embeddings = ~5.4 seconds.
At 3 row iterations × 500 frontier columns, this adds negligible overhead to the
existing MC pipeline.

---

## Implementation Phases

### Pilot (immediate — OOTB data)

1. Extend `ColumnSample` with `all_values` field
2. Update `sampler.py` loaders to populate `all_values`
3. Implement `select_row_sample()` with `stratified` strategy
4. Wire into bootstrap iteration loop (disagreement columns only)
5. BDD scenarios: row reservoir, stratified selection, iteration diversity

**Validation:** Run on OOTB 25-table corpus (300 columns × 100 rows each). Compare
classification accuracy with/without row MC. Measure: per-column Bel/Pl convergence
rate, SAGE `sample_values` importance shift, K reduction per iteration.

### Integration (with existing MC)

6. Multi-view embedding aggregation
7. Adaptive `max_values` from SAGE importance
8. Row-stability convergence signal in `should_stop_early()`
9. Mean-pooled embeddings for label propagation

### Scale validation (GitTables subsample)

10. Test on 10K-table GitTables subsample
11. Measure frontier-only row MC overhead
12. Validate propagation accuracy improvement from mean-pooled embeddings
13. Tune `k`, `iterations`, `multiview_threshold` parameters

---

## Assessment of Prior Art

No exact precedent exists for formalized nested Monte Carlo sampling across three
relational dimensions (tables, columns, rows) in CTA pipelines. The closest are:

- **Data profiling surveys:** Stratified/reservoir sampling for column profiling is
  standard practice but targets statistical summaries, not agentic LLM augmentation.
- **TableSage:** Uses row/column sampling to fit LLM context windows — analogous to
  our inner row loop but without DST evidence fusion or adaptive iteration.
- **RECA:** Leverages inter-table context for CTA but lacks row-level MC or
  SHAP/SAGE-guided adaptive weighting.

**Novel contributions:**
1. SHAP/SAGE-guided adaptive row weighting (feature importance drives row budget)
2. Row-stability as epistemic uncertainty signal (different rows → different labels = high uncertainty)
3. Multi-view embedding aggregation for label propagation robustness
4. Integration with DST belief intervals (row MC expands the evidence base, not just confidence scores)

---

## Files Modified/Created

| File | Action | Integration Point |
|------|--------|------------------|
| `src/atelier/classify/sampler.py` | Modify | `all_values` field, loader updates |
| `src/atelier/classify/row_sampler.py` | **New** | Row selection strategies |
| `src/atelier/classify/features.py` | Modify | Adaptive `max_values` |
| `src/atelier/classify/embedding.py` | Modify | Multi-view aggregation |
| `src/atelier/classify/pipeline.py` | Modify | Row iteration in bootstrap loop |
| `src/atelier/classify/bootstrap.py` | Modify | Row-stability tracking |
| `src/atelier/classify/monte_carlo.py` | Modify | Mean-pooled propagation embeddings |
| `src/atelier/classify/llm_backend.py` | Modify | Value count annotation in prompt |
| `src/atelier/config.py` | Modify | HOCON mappings for `classify.row_mc` |
| `config/base.conf` | Modify | `classify.row_mc {}` section |
| `features/agent/classification.feature` | Modify | Row MC BDD scenarios |
| `features/agent/step_defs/row_mc_steps.py` | **New** | Step definitions |
