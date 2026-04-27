<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Greenfield Atelier Implementation

## What was done

Built the complete Atelier greenfield CAI Application from scratch:

### Infrastructure
- **devenv.nix** — Python 3.12, Node.js 22, pnpm, process management (`devenv up` starts gRPC + Vite)
- **devenv.yaml** — Added `nixpkgs-python` input for Python version pinning
- **pyproject.toml** — hatchling build, core deps (grpcio, fastapi, pyhocon, sqlalchemy)
- **justfile** — Config management, dev, build, test, docs recipes with `env -i` pattern

### HOCON Configuration
- **config/base.conf** — Single source of truth (grpc, gateway, agents, db, data, cml sections)
- **src/atelier/config.py** — `load_config()` → `AtelierConfig` dataclass, `materialize_config()` → flat env file
- Config workflow: `.env` → HOCON `${?VAR}` → `just resolve-config` → `build/config/atelier.env` → `env -i`

### gRPC Service (proto-first, Fine Tuning Studio pattern)
- **atelier.proto** — HealthCheck, ListAgents, GetAgent, ListDatasets RPCs
- **service.py** — AtelierServicer (thin router)
- **server.py** — gRPC server startup with config
- **client.py** — AtelierClient wrapper around generated stub
- **gateway.py** — FastAPI bridge (REST → gRPC) + serves React build

### React Frontend
- **Vite + TypeScript + React 19** — @xyflow/react 12.x, Ant Design 5.x
- **Landing page** — Health status fetch, stat cards, feature cards
- **Cloudera branding** — Logo SVG from RAG Studio, themed header/footer
- Builds to `ui/dist/` (596KB gzipped to 191KB)

### CAI Deployment
- **.project-metadata.yaml** — Install Dependencies + Start Atelier tasks
- **scripts/install_deps.py** + **scripts/startup_app.py** — CML lifecycle
- **bin/start-app.sh** — Shell orchestrator (gRPC background + uvicorn foreground)
- **cdsw-build.sh** — Build hook

### Git Submodules
- `external/embedding-atlas` — git@github.com:rch/oss-embedding-atlas.git
- `external/hermes-agent` — git@github.com:zndx/oss-hermes-agent.git

### Documentation
- mdbook scaffold with d2/mermaid/katex preprocessors
- Architecture docs: overview (d2 diagram), deployment, agents, grpc

## Verification Results

| Check | Status |
|-------|--------|
| `uv sync` | 38 packages installed |
| `pnpm install` | 154 packages (via devenv shell) |
| `pnpm build` (React) | Builds successfully (596KB) |
| Proto generation | All 10 message types + servicer/stub |
| Config resolution | `build/config/atelier.env` materialized |
| Preflight validation | OK |
| gRPC round-trip | HealthCheck: ok v0.1.0 |
| Client wrapper | Connected and working |
| Gateway import | Atelier v0.1.0 loaded |

## Fix applied during build
- Removed `autopep8` from `devenv.nix` packages (not a nix package; already in pyproject.toml dev deps)
- Added `nixpkgs-python` input to `devenv.yaml` (required for `languages.python.version`)
- Added `ui/src/vite-env.d.ts` for SVG module type declarations
