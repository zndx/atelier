# Scenario Overview

Atelier uses [behave](https://behave.readthedocs.io/) (BDD) to capture platform decisions as executable specifications. Every scenario answers a concrete question: *Does the config load? Can the runtime start? Does the AMP install job produce a working system?*

These aren't just tests. They're the design context that connects architectural choices to the deployment realities of Cloudera AI.

## Active Domains

**24 scenarios across 8 features, 2 domains.**

### Infrastructure (infra)

Health checks and configuration lifecycle for the services Atelier depends on.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| Config lifecycle | `@config` | 0 | 3 | HOCON load, CLI override precedence, materialize + validate |
| PostgreSQL health | `@postgres` | 1 | 2 | Connection with pgvector extension, migration state |
| Qdrant health | `@qdrant` | 1 | 1 | Vector store HTTP health endpoint |
| PGlite process | `@pglite` | 0 | 2 | Node.js script existence, npm dependency declarations |

### Deployment

CAI deployment modalities and the runtime profile that catches failures before pushing.

| Feature | Tag | Tier | Scenarios | What it validates |
|---------|-----|------|-----------|-------------------|
| Runtime profile | `@runtime-profile` | 0 | 6 | Import chain, script executability, config resolution, migration parsing |
| AMP lifecycle | `@amp` | 0 + cai | 5 | `.project-metadata.yaml` structure, task patterns, install + start |
| Application modality | `@application` | 0 + 1 | 3 | HOST binding logic, full local stack startup |
| Studio modality | `@studio` | 0 | 2 | `IS_COMPOSABLE` root directory routing |

### By Tier

| Tier | Requires | Scenarios | Pass locally |
|------|----------|-----------|--------------|
| 0 | Python only | 18 | Yes |
| 1 | devenv stack | 4 | Yes (with `devenv up`) |
| cai | Live CAI session | 2 | Skipped (documentation-only) |

## Planned Domains

### Gateway (stub)

HTTP gateway and gRPC health checks. Will cover REST-to-gRPC bridge correctness, proto contract validation, and streaming endpoint behavior.

### Agent (stub)

Claude Agent SDK orchestration. Will cover keystone agent lifecycle, classification workflow execution, and embedding-atlas integration.

## Why BDD for a Deployment Platform?

CAI deployment has four modalities — Project, Application, AMP, and Studio — each with different constraints on networking, filesystem layout, and process lifecycle. Traditional unit tests verify module behavior in isolation. BDD scenarios verify that the *system* hangs together across these modalities.

Consider the Application modality: when `CDSW_APP_PORT` is set, the startup script must bind to `127.0.0.1` because CAI's reverse proxy handles external traffic. Bind to `0.0.0.0` instead and you bypass the proxy's auth layer. This isn't a bug in any single module — it's a deployment contract that only a scenario can express clearly:

```gherkin
Scenario: start-app.sh binds to 127.0.0.1 when CDSW_APP_PORT is set
  Given CDSW_APP_PORT is set to "8090"
  When I parse bin/start-app.sh for the HOST variable
  Then HOST is "127.0.0.1"
```

The scenario *is* the spec. A colleague reading this knows exactly what the constraint is, why it matters, and can verify it passes with `just bdd`.
