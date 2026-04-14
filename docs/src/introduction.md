# Introduction

Atelier is an agentic classification workbench for Cloudera AI. It classifies
column metadata using six independent evidence sources fused via
Dempster-Shafer Theory (DST), producing **belief intervals** instead of flat
confidence scores. An LLM-in-the-loop convergence agent identifies
disagreements between sources and orchestrates targeted reclassification
until the corpus stabilizes.

## Why Belief Intervals?

Traditional classifiers output a single confidence score (e.g., "85% email
address"). This hides two distinct types of uncertainty:

- **Aleatoric**: inherent randomness in the data
- **Epistemic**: ignorance due to insufficient evidence

DST separates these via belief intervals `[Bel(A), Pl(A)]`. When
`Bel = 0.8, Pl = 0.85`, we have high confidence with low ambiguity. When
`Bel = 0.3, Pl = 0.9`, something supports A but much remains uncertain — a
signal to gather more evidence. This distinction drives the entire pipeline:
high-K (conflict) columns are automatically escalated for re-examination.

## Architecture

```d2
direction: down

ui: React Frontend {
  canvas: XYFlow Agent Canvas
  embeddings: Embeddings Visualization
}

gateway: FastAPI Gateway {
  tooltip: "REST → gRPC bridge\nServes React build"
}

grpc: gRPC Core {
  tooltip: "Proto-first API\n7 RPCs"
}

agents: Claude Agent SDK {
  tooltip: "5-tool convergence loop\nConflict-driven revisit"
}

classify: Classification Pipeline {
  tooltip: "6 evidence sources\nDST fusion\nMC sampling at scale"
}

db: PostgreSQL {
  tooltip: "devenv: PG 16 + pgvector\nCAI: PGlite"
}

qdrant: Qdrant {
  tooltip: "Vector store\nEmbedding search"
}

ui -> gateway: REST /api/*
gateway -> grpc: gRPC :50051
grpc -> classify
grpc -> agents
classify -> db
classify -> qdrant
```

## Six Evidence Sources

| Source | Type | Phase |
|--------|------|-------|
| Cosine similarity | Sentence-transformer embedding | M0 (cheap) |
| Pattern detection | 8 regex detectors (email, SSN, credit card, ...) | M0 |
| Name matching | Column name vs label/abbrev/aliases | M0 |
| LLM | Anthropic / Bedrock / Cerebras / OpenAI-compatible | M1 |
| CatBoost | Gradient-boosted trees with virtual ensembles | M2 |
| SVM | TF-IDF + LinearSVC with Platt scaling | M2 |

Each source independently produces a mass function. Sources are fused via
Dempster's rule of combination. Conflict K between sources is the diagnostic
signal that drives targeted revisit.

## Scale

The pipeline is designed for corpora ranging from 50 columns (OOTB sample)
to 120M+ columns (full GitTables at 10M+ tables). Monte Carlo stratified
sampling selects a representative subset for frontier LLM classification
and propagates labels to the remaining corpus via embedding similarity.
At scale, this reduces LLM inference cost by >99.99% while preserving
classification quality through DST conflict-driven escalation.

## Out-of-the-Box Experience

A fresh deployment auto-seeds on first boot:

1. **300-term BFO-grounded vocabulary** covering the CCO ICE trichotomy
   (Designative, Descriptive, Prescriptive information content entities)
2. **25 sample tables** with 300 columns and committed ground truth
3. One-click classification via the Status page
4. Interactive Embeddings visualization of results

## Quick Start

**Local development** (devenv):

```bash
devenv shell          # Enter dev environment
just install          # Install Python + Node dependencies
just up               # Start gRPC + gateway + Vite dev server
```

**CAI deployment**: Deploy as an AMP from `https://github.com/zndx/atelier`.

## Documentation Map

- **[System Overview](./architecture/overview.md)** — Component diagram
- **[Deployment](./architecture/deployment.md)** — CAI AMP and local dev setup
- **[gRPC & Gateway](./architecture/grpc.md)** — Proto contract, REST endpoints, config lifecycle
- **[Keystone Agents](./architecture/agents.md)** — Agent convergence loop with 5 tools
- **[Classification Pipeline](./architecture/classification.md)** — DST methodology, evidence sources, bootstrap convergence
- **[Monte Carlo Sampling](./architecture/monte-carlo.md)** — Stratified sampling for scale
- **[GPU Acceleration](./architecture/gpu.md)** — CUDA detection and batch encoding
- **[Synthetic Data & Training](./architecture/synth.md)** — 316+ generators, CatBoost + SVM training
- **[Embeddings](./architecture/embeddings.md)** — Interactive parquet visualization
- **[Data Sources](./architecture/data-sources.md)** — Source-aware versioning and OOTB sample
- **[BDD Scenarios](./scenarios/overview.md)** — 127 scenarios across 4 domains
