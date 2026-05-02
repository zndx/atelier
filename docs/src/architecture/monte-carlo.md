<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Monte Carlo Sampling

At small corpus sizes (< 200 columns), every column receives direct
frontier-LLM classification. As the corpus scales to thousands or millions
of columns, this becomes prohibitively expensive. Monte Carlo stratified
sampling selects a representative subset for LLM inference and propagates
labels cheaply via embedding similarity.

This is a **zero-cost optimization**: below the threshold, the pipeline
behaves identically to before. The MC layer activates transparently at scale.

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
  ├─ [existing] Frontier LLM classifies MC sample (not all columns)
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

Total budget: `min(max_frontier_columns, total × sample_fraction)`

## Label Propagation

After the LLM sweep on frontier columns:

1. For each propagation column, find nearest frontier column by cosine
   similarity (stratum-local to limit search space)
2. If similarity >= `propagation_threshold`: assign same label with
   discounted confidence
3. If similarity < threshold: column gets no LLM evidence in DST

Propagated labels enter DST fusion with a higher discount factor (0.30 vs
0.10 for direct LLM) — they carry less evidential mass. If M0 sources
disagree with the propagated label, conflict K rises and the existing
targeted-revisit loop automatically escalates the column to the frontier model.

## Why This Works with DST

The evidence fusion framework makes MC sampling robust:

- **Propagated evidence** carries less mass (more goes to Theta/ignorance)
- **M0 agreement** with propagated label → high belief, narrow gap (good)
- **M0 disagreement** with propagated label → wide gap → frontier revisit
- **Escalation is automatic** — no special MC-aware revisit logic needed

## Scaling Projections

GitTables corpus: **1.7M tables today, 10M+ near-term**. Average 8-12
columns per table = **15M-120M columns** at full scale.

| Corpus | MC Mode | Frontier Calls | Propagated | Cost Reduction |
|--------|---------|---------------:|-----------:|---------------:|
| 50 | Passthrough | 50 (all) | 0 | 0% |
| 500 | Active | ~75 (15%) | ~425 | 85% |
| 5,000 | Active | ~500 (cap) | ~4,500 | 90% |
| 50K | Active | ~500 (cap) | ~49.5K | 99% |
| 500K | Active | ~500 (cap) | ~499.5K | 99.9% |
| 15M | Active | ~500 (cap) | ~15M | >99.99% |
| 120M | Active | ~500 (cap) | ~120M | >99.99% |

At the `max_frontier_columns=500` cap, stratified importance sampling ensures
every category stratum gets at least `min_per_stratum=3` exemplars. Uniform
random sampling at 500/15M would miss rare categories entirely.

### Scale-Critical Design Decisions

- **Embedding computation**: batch GPU encoding at ~2,768 texts/s (RTX 4090);
  15M columns takes ~90 minutes. One-time cost, GPU-parallelizable.
- **Stratum-local propagation**: similarity search within each stratum
  (not across the full corpus) to limit memory and compute.
- **Memory**: 15M columns × 200B = ~3GB for metadata; 15M × 1.5KB = ~22GB
  for embeddings. Requires streaming/chunked processing.
- **Escalation budget**: ~50-100 additional frontier calls from revisit.
  Total frontier budget: ~600 LLM API calls for a 15M-column corpus.

## Configuration

```hocon
classify {
  monte_carlo {
    min_corpus_size = 200              # Below this, classify everything
    min_corpus_size = ${?ATELIER_MC_MIN_CORPUS_SIZE}
    sample_fraction = 0.15             # Fraction for frontier model
    sample_fraction = ${?ATELIER_MC_SAMPLE_FRACTION}
    min_per_stratum = 3                # Minimum samples per category stratum
    max_frontier_columns = 500         # Hard cap on frontier columns
    max_frontier_columns = ${?ATELIER_MC_MAX_FRONTIER}
    propagation_threshold = 0.85       # Cosine sim for propagation
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
├── MCPlan            — Sampling plan (frontier + propagation sets)
├── pre_classify()    — Run M0 evidence for all columns
├── stratify()        — Group by preliminary category + uncertainty
├── select_sample()   — Importance-weighted selection within strata
└── propagate_labels() — Embedding-similarity label extension
```
