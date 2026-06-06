# Monte Carlo Sampling

At small corpus sizes (< 200 columns), every column receives direct
LLM classification. As the corpus scales to thousands or millions
of columns, this becomes prohibitively expensive. Monte Carlo stratified
sampling selects a representative subset for direct LLM inference and
propagates labels cheaply via embedding similarity to the remainder.

This is a **zero-cost optimization**: when in passthrough, the pipeline
behaves identically to before. The MC layer is *passthrough* (classifies
every column) whenever `total < min_corpus_size` **or** `sample_fraction
>= 1.0` (`monte_carlo.py`). Because the shipped default is `sample_fraction
= 1.00`, MC is passthrough at every corpus size out of the box; sub-sampling
engages only once an operator lowers `sample_fraction` below `1.0` on a
corpus larger than `min_corpus_size`.

## Three-Phase MC Layer

The MC layer operates between SAMPLING and LLM_SWEEP in the existing
pipeline. No new FSM states — it runs as sub-phases.

```
SAMPLING
  ├─ [existing] Extract features for all columns
  ├─ Pre-classify: cheap M0 evidence (name, pattern, cosine) — no LLM
  ├─ Stratify: group by preliminary category + uncertainty
  └─ Select MC sample: importance-weighted within strata

LLM_SWEEP
  ├─ [existing] LLM classifies the MC sample (not all columns)
  └─ Propagate: extend labels to remaining corpus via embedding similarity

VALIDATING
  └─ [existing] Full 6-source DST on ALL columns
      (propagated labels enter as discounted LLM evidence)
      → High-gap / low-belief propagated columns escalate to revisit
```

### Phase 1: Pre-Classification

Run M0 evidence sources only (no LLM, no ML models). For each column:

- **Name matching** → best category + mass
- **Pattern detection** → matched categories
- **Cosine similarity** → top-K categories + scores

Returns a preliminary category code + confidence for every column. Uses the
existing `name_match_to_mass()`, `pattern_to_mass()`, `classify_cosine()`
functions from the pipeline.

### Phase 2: Stratification

Partition columns by their preliminary category code:

- **Rare strata** (< 2 x `min_per_stratum` members): fully sampled
- **UNRESOLVED stratum** (M0 sources disagree or low confidence): fully sampled
- **Normal strata**: proportional allocation with importance weighting

### Phase 3: Sample Selection

Within each normal stratum, select columns via importance-weighted random
sampling without replacement. Importance weight per column:

```
w = (1 - confidence) × (1 + uncertainty)
```

where `confidence` = max cosine similarity, `uncertainty` = ratio of
2nd-best to 1st-best similarity (ambiguity measure).

Total budget: `min(max_sampled_columns, total × sample_fraction)`

> The fractions used in the scaling table below (e.g. `15%`) illustrate an
> *active-MC* configuration with `sample_fraction = 0.15`; recall the shipped
> default is `1.00` (passthrough — see the intro).

## Label Propagation

After the LLM sweep on the sampled subset:

1. For each propagation column, find the nearest directly-classified
   column by cosine similarity (stratum-local to limit search space)
2. If similarity >= `propagation_threshold`: assign same label with
   discounted confidence
3. If similarity < threshold: column gets no LLM evidence in DST

Propagated labels enter DST fusion with a higher discount factor (0.30 vs
0.10 for direct LLM) — they carry less evidential mass. If M0 sources
disagree with the propagated label, conflict K rises and the existing
targeted-revisit loop automatically escalates the column for direct
LLM classification.

## Why This Works with DST

The evidence fusion framework makes MC sampling robust:

- **Propagated evidence** carries less mass (more goes to Theta/ignorance)
- **M0 agreement** with propagated label → high belief, narrow gap (good)
- **M0 disagreement** with propagated label → wide gap → revisit-via-LLM
- **Escalation is automatic** — no special MC-aware revisit logic needed

## Scaling Projections

GitTables corpus: **1.7M tables today, 10M+ near-term**. Average 8-12
columns per table = **15M-120M columns** at full scale.

The table below assumes an *active-MC* configuration with `sample_fraction =
0.15` (the shipped default is `1.00` — see the note above; "Passthrough"
behaviour is also what the default produces at any corpus size).

| Corpus | MC Mode | Direct LLM Calls | Propagated | Cost Reduction |
|--------|---------|-----------------:|-----------:|---------------:|
| 50 | Passthrough | 50 (all) | 0 | 0% |
| 500 | Active | ~75 (15%) | ~425 | 85% |
| 5,000 | Active | ~500 (cap) | ~4,500 | 90% |
| 50K | Active | ~500 (cap) | ~49.5K | 99% |
| 500K | Active | ~500 (cap) | ~499.5K | 99.9% |
| 15M | Active | ~500 (cap) | ~15M | >99.99% |
| 120M | Active | ~500 (cap) | ~120M | >99.99% |

At the `max_sampled_columns=500` cap, stratified importance sampling ensures
every category stratum gets at least `min_per_stratum=3` exemplars. Uniform
random sampling at 500/15M would miss rare categories entirely.

### Scale-Critical Design Decisions

- **Embedding computation**: batch GPU encoding at ~2,768 texts/s (RTX 4090);
  15M columns takes ~90 minutes. One-time cost, GPU-parallelizable.
- **Stratum-local propagation**: similarity search within each stratum
  (not across the full corpus) to limit memory and compute.
- **Memory**: 15M columns × 200B = ~3GB for metadata; 15M × 1.5KB = ~22GB
  for embeddings. Requires streaming/chunked processing.
- **Escalation budget**: ~50-100 additional direct-LLM calls from revisit.
  Total LLM call budget: ~600 calls for a 15M-column corpus.

## Configuration

```hocon
classify {
  monte_carlo {
    min_corpus_size = 200              # Below this, classify everything
    min_corpus_size = ${?ATELIER_MC_MIN_CORPUS_SIZE}
    sample_fraction = 1.00             # Fraction directly classified by LLM (shipped default: classify all)
    sample_fraction = ${?ATELIER_MC_SAMPLE_FRACTION}
    min_per_stratum = 3                # Minimum samples per category stratum
    max_sampled_columns = 500          # Hard cap on directly-classified columns
    max_sampled_columns = ${?ATELIER_MC_MAX_SAMPLED}
    propagation_threshold = 0.80       # Cosine sim for propagation
    propagation_threshold = ${?ATELIER_MC_PROPAGATION_THRESHOLD}
    propagation_discount = 0.30        # LLM mass discount for propagated labels
  }
}
```

## Module Structure

```
src/atelier/classify/monte_carlo.py
├── MCConfig          — Frozen dataclass with from_cfg() factory
├── PreClassification — Per-column M0 result (code + confidence + uncertainty)
├── Stratum           — Column group by preliminary category
├── MCPlan            — Sampling plan (sampled + propagation sets)
├── pre_classify()    — Run M0 evidence for all columns
├── stratify()        — Group by preliminary category + uncertainty
├── select_sample()   — Importance-weighted selection within strata
└── propagate_labels() — Embedding-similarity label extension
```
