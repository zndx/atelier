# Scenario Overview

Atelier uses [behave](https://behave.readthedocs.io/) (BDD) to capture platform decisions as executable specifications. Every scenario answers a concrete question: *Does the config load? Can the runtime start? Does the classification pipeline converge?*

These aren't just tests. They're the design context that connects architectural choices to the deployment realities of Cloudera AI.

## Active Domains

**127 scenarios across 31 features, 4 domains.**

### Infrastructure (infra)

Health checks and configuration lifecycle for the services Atelier depends on.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| Config lifecycle | `@config` | 0 | 3 | HOCON load, CLI override precedence, materialize + validate |
| PostgreSQL health | `@postgres` | 1 | 2 | Connection with pgvector extension, migration state |
| Qdrant health | `@qdrant` | 1 | 1 | Vector store HTTP health endpoint |
| PGlite process | `@pglite` | 0 | 2 | Node.js script existence, npm dependency declarations |
| Preflight | `@preflight` | 0 | 3 | Structured deny/warn checks, GPU detection |

### Deployment

CAI deployment modalities and the runtime profile that catches failures before pushing.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| Runtime profile | `@runtime-profile` | 0 | 6 | Import chain, script executability, config resolution, migration parsing |
| AMP lifecycle | `@amp` | 0 + cai | 5 | `.project-metadata.yaml` structure, task patterns, install + start |
| Application modality | `@application` | 0 + 1 | 3 | HOST binding logic, full local stack startup |
| Studio modality | `@studio` | 0 | 2 | `IS_COMPOSABLE` root directory routing |
| Embeddings integration | `@embeddings` | 0 | 4 | npm dependency, page component, React Router, preparation script |
| Naming conventions | `@naming` | 0 | 2 | User-facing surfaces say "Embeddings", no Apache Atlas confusion |

### Gateway

HTTP gateway endpoints, gRPC bridge, and live service integration.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| API endpoints | `@api` | 0 + 1 | 8 | REST endpoint contracts, response shapes |
| API testclient | `@testclient` | 0 | 7 | FastAPI TestClient integration (no running server) |
| Status endpoint | `@status` | 0 + 1 | 4 | Aggregated health report, config state |
| Pipeline integration | `@pipeline` | 1 | 2 | Classification pipeline via gateway |
| SPA routes | `@spa` | 0 | 0 | (placeholder for client-side routing tests) |

### Agent

Classification pipeline, DST evidence fusion, ML classifiers, and agent orchestration.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| Classification pipeline | `@gpu` | 0 | 19 | DST belief, Dempster combination, features, patterns, name matching, pipeline E2E, Monte Carlo sampling |
| Bootstrap convergence | `@bootstrap` | 0 | 10 | LLM sweep, ML validation, targeted revisit, convergence criteria |
| Agent convergence loop | `@gpu` | 0 | 6 | 5-tool agent loop, conflict reports, convergence, mock client |
| Agent smoke test | `@agent` | 0 | 6 | Agent metadata, tool definitions, state formatting |
| LLM backends | `@backend` | 0 | 8 | Backend factory, Anthropic/Bedrock/Cerebras/OpenAI clients |
| ML classifiers | `@ml` | 0 | 4 | CatBoost + SVM training, inference, virtual ensemble UQ |
| ML E2E | `@ml-e2e` | 0 | 2 | Full synth → train → classify → evaluate cycle |
| Belief path | `@belief-path` | 0 | 3 | Hierarchical navigation, cautious classification |
| SAGE importance | `@sage` | 0 | 1 | Permutation-based feature importance |
| SHAP explanations | `@shap` | 0 | 2 | TreeSHAP + PermutationSHAP attribution |
| Synth generation | `@synth` | 0 | 2 | Synthetic data + ground truth generation |
| Synth framework | `@synth-framework` | 0 | 2 | Generator registry, coverage reporting |
| Meta-tagging | `@meta-tagging` | 0 | 2 | META_TO_ICE mappings, coverage |
| Experimentation | `@experimentation` | 0 | 3 | Discount tuning, comparative evaluation |
| Real data | `@real-data` | 0 | 3 | Production annotation validation (requires build/data/) |

### By Tier

| Tier | Requires | Scenarios | Pass locally |
|------|----------|-----------|--------------|
| 0 | Python only | ~105 | Yes |
| 1 | devenv stack | ~15 | Yes (with `devenv up`) |
| cai | Live CAI session | ~5 | Skipped (documentation-only) |

Additional tags: `@slow` (~17 scenarios requiring extended runtime),
`@gpu` (GPU detection/acceleration scenarios — run on CPU too, just slower).

## Why BDD for a Deployment Platform?

CAI deployment has four modalities — Project, Application, AMP, and Studio — each with different constraints on networking, filesystem layout, and process lifecycle. Traditional unit tests verify module behavior in isolation. BDD scenarios verify that the *system* hangs together across these modalities.

Consider the Application modality: when `CDSW_APP_PORT` is set, the startup script must bind to `127.0.0.1` because CAI's reverse proxy handles external traffic. Bind to `0.0.0.0` instead and you bypass the proxy's auth layer. This isn't a bug in any single module — it's a deployment contract that only a scenario can express clearly:

```gherkin
Scenario: start-app.sh binds to 127.0.0.1 when CDSW_APP_PORT is set
  Given CDSW_APP_PORT is set to "8090"
  When I parse bin/start-app.sh for the HOST variable
  Then HOST is "127.0.0.1"
```

The scenario *is* the spec. A colleague reading this knows exactly what the constraint is, why it matters, and can verify it passes with `just behave`.
