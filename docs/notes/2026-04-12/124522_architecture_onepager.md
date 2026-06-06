# Atelier — Technical Architecture One-Pager

## What Is Atelier?

Atelier is an **agentic classification workbench** for Cloudera AI (CAI). It classifies database column metadata against a controlled vocabulary using Dempster-Shafer evidence fusion — combining ML classifiers, pattern detection, name matching, and LLM reasoning into a single belief-theoretic framework.

Deploys as a CAI Application (AMP) from `github.com/zndx/atelier`.

---

## System Topology

```
┌──────────────────────────────────┐
│  React Frontend (:3000 dev)      │
│  XYFlow canvas + Embeddings viz  │
└──────────┬───────────────────────┘
           │ REST /api/*
┌──────────▼───────────────────────┐
│  FastAPI Gateway (:8090)         │
│  REST → gRPC bridge              │
│  Serves compiled React in prod   │
└──────────┬───────────────────────┘
           │ gRPC
┌──────────▼───────────────────────┐
│  gRPC Core Service (:50051)      │
│  Proto-first, thin servicer      │
│  Routes to business logic        │
└──────┬──────────┬────────────────┘
       │          │
┌──────▼──┐  ┌───▼──────────┐
│ Agents  │  │ PostgreSQL   │
│ (Claude │  │ + Qdrant     │
│  SDK)   │  │ (vectors)    │
└─────────┘  └──────────────┘
```

---

## Classification Pipeline

The core value proposition: classify columns from Hive/data-platform tables against an annotation vocabulary (296+ categories, 7-level hierarchy).

### Evidence Sources (Dempster-Shafer Fusion)
1. **CatBoost** — Gradient-boosted trees on embedding features
2. **SVM** — Support vector classifier on same features
3. **Cosine similarity** — Embedding distance to category prototypes
4. **Pattern detection** — Regex/format recognition (dates, emails, IPs, etc.)
5. **Name matching** — Column name → category code fuzzy matching
6. **LLM reasoning** — Claude-mediated classification with alternatives

Each source produces a **mass function** (belief assignment). These are fused via **Dempster's rule of combination** to produce a final classification with calibrated confidence. Conflicting evidence is surfaced, not hidden.

### Bootstrap Convergence Loop
For production use, the pipeline runs an iterative loop:
1. ML classifiers make initial predictions
2. LLM reviews predictions, especially low-confidence and conflicting ones
3. Disagreements between ML and LLM trigger re-evaluation
4. Loop converges when conflict drops below threshold

### Current Baseline (real data, 213 columns)
- **Accuracy: 99.5%** | Hierarchical: 99.5% | Micro-F1: 0.995
- All 4 ML evidence sources firing (catboost, svm, cosine, name_match)
- Template-based synthetic training covers 214/221 vocabulary categories

---

## CAI Deployment Model

### Environment Variables (set in CAI project settings)
| Variable | Purpose |
|----------|---------|
| `AWS_ACCESS_KEY_ID` | Bedrock inference (production default) |
| `AWS_SECRET_ACCESS_KEY` | Bedrock credentials |
| `AWS_REGION` | Bedrock region |
| `ANTHROPIC_API_KEY` | Direct API (onboarding/override) |
| `ATELIER_AGENT_MODEL` | Model ID or Bedrock inference profile ARN |
| `ATELIER_DATA_CONNECTIONS` | Comma-separated CAI Data Platform connections |

Both Bedrock and direct Anthropic credentials can coexist — credentials determine what's available, not a global switch.

### Startup Sequence (`bin/start-app.sh`)

```
Phase 1: Kill stale processes
    ↓
Phase 2: Infrastructure
    ├── PGlite (embedded PostgreSQL via Node.js, port 5440)
    │   └── Proof-of-progress readiness (extends deadline while work advances)
    └── Qdrant (vector store, ports 6333/6334)
    ↓
Phase 3: Core services
    ├── Config resolution (HOCON → env)
    ├── Preflight validation
    ├── Database migrations
    ├── Dataset seeding (GitTables sample if DB empty)
    ├── Keystone agent seeding (5 agents)
    ├── gRPC server (background, port 50051)
    └── HTTP gateway (foreground, CDSW_APP_PORT)
```

### AMP Tasks (`.project-metadata.yaml`)
1. **Install Dependencies** — `scripts/install_deps.py` (uv, React build, Qdrant binary)
2. **Start Atelier** — `scripts/startup_app.py` (restart loop around `bin/start-app.sh`)

### CAI-Specific Gotchas
- Python runtimes do **not** include Node.js — installed via nvm
- PGlite replaces pgserver for embedded PostgreSQL (no C extension issues)
- Platform env vars use legacy `CDSW_*` prefix (`CDSW_APP_PORT`, `CDSW_PROJECT_ID`)
- `uv sync` creates `.venv/` that app sessions don't activate — `pip install -e .` for CAI

---

## Config Pattern

**HOCON is the single source of truth** (`config/base.conf`). No module reads `os.environ` directly.

```
.env → devenv shell → HOCON ${?VAR} substitution → build/config/atelier.{env,json}
```

Python code uses `load_config()` → `AtelierConfig` dataclass. External tools (conftest, CI) use materialized `build/config/atelier.env`.

---

## Three-Tier Compatibility

| Tier | Tools | Where |
|------|-------|-------|
| **devenv** | Nix services, process manager, dotenv | Local dev |
| **just** | Portable task runner | Anywhere with just+uv |
| **Python/bash** | Scripts only, no devenv/just | CAI runtime |

CAI uses tier 3 exclusively.

---

## Development Quick Start

```bash
devenv shell              # Enter dev environment
just install              # uv sync + pnpm install
just proto                # Generate proto stubs from atelier.proto
just up                   # Start full stack (PG, Qdrant, gRPC, Vite)
just bdd                  # Run tier-0 BDD tests (pure Python, no services)
just bdd-full             # Run tier-0 + tier-1 (requires stack)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `src/atelier/proto/atelier.proto` | gRPC service contract (source of truth) |
| `src/atelier/service.py` | gRPC servicer — thin router to logic modules |
| `src/atelier/gateway.py` | FastAPI gateway — REST bridge + React serving |
| `src/atelier/config.py` | Config loading, validation, `AtelierConfig` |
| `src/atelier/classify/` | Classification pipeline (DST fusion, ML, synth) |
| `config/base.conf` | HOCON config (single source of truth) |
| `bin/start-app.sh` | Production startup orchestrator |
| `.project-metadata.yaml` | CAI AMP deployment metadata |
| `ui/` | React frontend (Vite + React 19 + Ant Design) |
| `features/` | BDD test suite (behave, tier-0/tier-1/tier-cai) |

---

## Near-Term Objectives

1. **CAI integration testing** — Exercise the classification pipeline against production Hive tables via CAI Data Platform connections
2. **Bootstrap convergence on real data** — Validate the LLM-mediated DST loop at production vocabulary scale (296 categories)
3. **Embedding visualization** — Atlas projection of classification results for data steward review
4. **Agent orchestration** — XYFlow canvas showing keystone agent topology and pipeline progress
