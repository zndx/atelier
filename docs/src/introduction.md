# Introduction

Atelier is an agentic classification workbench for Cloudera AI. It classifies
column metadata using six independent evidence sources fused via
Dempster-Shafer Theory (DST), producing **belief intervals** instead of point
estimates. An LLM-in-the-loop convergence agent identifies disagreements
between sources and orchestrates targeted reclassification until the corpus
stabilizes.

## Why Belief Intervals?

Traditional classifiers output a single probability \\( P(A) = 0.85 \\) — "85%
email address." This conflates two fundamentally different situations: high
confidence with abundant evidence vs. moderate confidence with sparse evidence.
A Bayesian posterior and a coin flip can both yield 0.5, but they represent
very different epistemic states.

Dempster-Shafer theory separates these via the **belief function**
\\( \text{Bel}(A) \\) and **plausibility function** \\( \text{Pl}(A) \\), where:

$$
\text{Bel}(A) = \sum_{B \subseteq A} m(B), \qquad
\text{Pl}(A) = 1 - \text{Bel}(\bar{A})
$$

The interval \\( [\text{Bel}(A),\; \text{Pl}(A)] \\) bounds the true
probability. Its width \\( \text{Pl}(A) - \text{Bel}(A) \\) quantifies
**epistemic uncertainty** — how much we don't know:

| Interval | Interpretation |
|----------|---------------|
| \\( [0.82,\; 0.87] \\) | Strong evidence, low ambiguity — classify with confidence |
| \\( [0.30,\; 0.90] \\) | Some support for \\(A\\), but high ignorance — gather more evidence |
| \\( [0.45,\; 0.55] \\) | Two sources disagree — wide gap, needs revisit |

This distinction drives the entire pipeline: columns with **wide
belief gaps** (where \\( \text{Pl}(A) - \text{Bel}(A) \\) is large)
are automatically escalated for LLM re-examination with enriched
context.  Conflict \\( K \\) is tracked as a diagnostic but the
**gap width** determines which columns need attention.

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
  tooltip: "6-tool convergence loop\nConflict-driven revisit"
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

Each source independently produces a **mass function**
\\( m_i : 2^\Theta \to [0, 1] \\) over the frame of discernment \\( \Theta \\)
(the set of all category codes). Sources are grouped by computational cost:

| Source | Feature Space | Cost Tier |
|--------|--------------|-----------|
| **MaxSim** | ColBERT v2 per-token multi-vectors (BERT + 768→128 projection), scored by **Qdrant native MaxSim** late-interaction over a single `colbert` multi-vector field; **fail-fast** (`MaxSimUnavailable` → FSM ERROR, no silent fallback) | M0 (local) |
| **Pattern detection** | 27 regex detectors + 9 post-regex validators (email, phone, SSN, IP, UUID, date, URL, credit card + Luhn, MAC, IBAN, currency + ISO 4217, …) with graduated mass scaling by match fraction. A fixed-at-import snapshot of a library that is **dynamic by design** — it must grow for objectives a static set leaves unsatisfiable in situ (rule-based admission / Rete-over-regex + Ægir; roadmap) | M0 |
| **Name matching** | Column name vs vocabulary labels, codes, and aliases (4-tier: exact > code > alias > overlap) | M0 |
| **LLM classification** | Frontier model reasoning (Anthropic / Bedrock / Cerebras / OpenAI-compatible) | M1 (API) |
| **CatBoost** | 12 discrete features + 384-dim MiniLM embedding; virtual ensemble uncertainty via `posterior_sampling` | M2 (trained) |
| **SVM** | ModernBERT mean-pool dense embeddings → factorized fully-hierarchical NHSVM (Choi et al. 2015): one weight vector per hierarchy node, root-to-leaf path scores, **non-leaf nodes are first-class prediction targets**; registry-promoted head trained offline on the synth corpus | M2 (trained) |

Both the SVM and CatBoost classifiers now operate on **dense embeddings**
(ModernBERT mean-pool for the NHSVM; MiniLM 384-dim for CatBoost), so the old
"sparse lexical vs dense semantic" distinction no longer separates them.
Genuine evidence independence for Dempster's rule is therefore not asserted from
feature-space orthogonality — it is **enforced via per-source reliability
discounts** (Denoeux 2008): each source's mass is discounted toward ignorance in
proportion to its calibrated unreliability and its shared-error coupling with
other channels. The discount calibration, not the feature space, carries the
independence guarantee.

### Fusion

Sources are combined via the **conjunctive rule of combination**:

$$
m_{1 \oplus 2}(C) = \frac{1}{1-K} \sum_{\substack{A \cap B = C \\ A,B \subseteq \Theta}} m_1(A) \cdot m_2(B)
$$

where the **conflict** \\( K = \sum_{A \cap B = \varnothing} m_1(A) \cdot m_2(B) \\)
measures the degree to which sources contradict each other. High \\( K \\) is
the diagnostic signal that drives the convergence loop: columns where
independent evidence sources disagree are escalated for targeted LLM revisit
with enriched context (ML prediction, belief interval, source disagreement).

### Hierarchical Classification

The vocabulary forms a **rooted code tree** (e.g.,
`ICE.SENSITIVE.PID.CONTACT.EMAIL`). Belief and plausibility are queryable at
any depth — \\( \text{Bel}(\texttt{ICE.SENSITIVE}) \\) aggregates all
descendants. The `cautious_code(τ)` operator returns the deepest code where
\\( \text{Bel} > \tau \\), enabling principled depth-accuracy tradeoffs:
high \\( \tau \\) yields coarse but reliable labels; low \\( \tau \\) yields
specific but less certain ones.

## Convergence

The bootstrap pipeline iterates three phases until the **belief gap**
(\\( \text{Pl}(A) - \text{Bel}(A) \\)) stabilizes:

1. **LLM sweep** — classify each directly-targeted column via batch LLM calls
2. **ML validation** — run the full 6-source DST pipeline; compute
   per-column belief, plausibility, and gap
3. **Targeted revisit** — re-classify only **uncertain** columns
   (high gap or low belief) with enriched context (ML prediction +
   belief interval + detected patterns + disagreement summary)

The primary convergence measure is **mean belief gap** — the average
width of the \\( [\text{Bel}, \text{Pl}] \\) interval across all
columns.  A narrow gap means the evidence sources agree on a confident
prediction.  Conflict \\( K \\) is tracked as a **diagnostic signal**
(it indicates source disagreement) but does not gate convergence — a
column can have \\( K = 0.9 \\) but \\( \text{Bel} = 0.95 \\): the
sources fought, but the winner is clear.

An **agent-driven** variant (via Claude Agent SDK) delegates the
revisit strategy to an LLM that reasons about uncertainty patterns
and declares convergence when diminishing returns are reached.
(Earlier revisions exposed a `retrain_svm` tool that progressively
improved the SVM on accumulated LLM labels — excised on 2026-05-04
for source-independence reasons; see
[DST Evidence Independence](./architecture/dst-evidence-independence.md).)
The **programmatic** variant uses gap + coverage thresholds for
environments where tool-use isn't available.

### Factorized Hierarchical SVM

The SVM is a **factorized fully-hierarchical NHSVM** (Choi et al. 2015) trained
**once**, offline, on the synthetic corpus over **ModernBERT mean-pool dense
embeddings**.  Rather than keying labels on leaves alone, it learns **one weight
vector per hierarchy node** (plus per-node alphas): an entity is scored along the
**root-to-leaf path** so that **non-leaf nodes are first-class prediction
targets**, and a calibrated softmax temperature turns path scores into a mass
function.  The registry-promoted head **emits user-taxonomy codes natively** —
there is no runtime LLM-mediated alignment step.

The factorized form exists because dense embeddings catastrophically fail the
*old* Kronecker NHSVM (98.9% TF-IDF vs 4.3% naïve-dense); the TF-IDF +
LinearSVC + Platt path survives only as a **legacy/baseline** classifier
(`per_vocab_legacy`), never as the shipped SVM.  Because the SVM is now dense
like CatBoost, its independence is **not** claimed from feature-space
orthogonality but from per-source reliability discounts (Denoeux 2008).  See
[DST Evidence Independence](./architecture/dst-evidence-independence.md)
for the full design rationale and the BM25-reranker future-work plan.

## Scale

The pipeline handles corpora from 50 columns (OOTB sample) to 120M+ columns
(full GitTables at 10M+ tables). **Monte Carlo stratified sampling** selects
a representative subset for direct LLM classification and propagates labels
to the remaining corpus via embedding similarity.

With `max_sampled_columns = 500`, classifying a 120M-column corpus requires
LLM inference on only 0.0004% of columns — a **>99.99% cost reduction** while
preserving classification quality through DST conflict-driven escalation of
uncertain propagations.

## Out-of-the-Box Experience

A fresh deployment auto-seeds on first boot:

1. **316-leaf BFO-grounded vocabulary** (351 categories total) covering the
   CCO Information Content Entity trichotomy: Designative (names, IDs, codes),
   Descriptive (measurements, dates, amounts), Prescriptive (software, specs)
2. **25 sample tables** with 316 columns and a committed curated reference
3. One-click classification via the Status page
4. Interactive Embeddings visualization (UMAP/t-SNE via embedding-atlas)

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
- **[Keystone Agents](./architecture/agents.md)** — Agent convergence loop with 6 tools
- **[Classification Pipeline](./architecture/classification.md)** — DST methodology, evidence sources, bootstrap convergence
- **[Monte Carlo Sampling](./architecture/monte-carlo.md)** — Stratified sampling for scale
- **[GPU Acceleration](./architecture/gpu.md)** — CUDA detection and batch encoding
- **[Synthetic Data & Training](./architecture/synth.md)** — 316+ generators, factorized hierarchical NHSVM, CatBoost fit-to-LLM
- **[Embeddings](./architecture/embeddings.md)** — Interactive parquet visualization
- **[Data Sources](./architecture/data-sources.md)** — Source-aware versioning, OOTB sample, Hive auto-discovery
- **[BDD Scenarios](./scenarios/overview.md)** — 141 scenarios across 4 domains
