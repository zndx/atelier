# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atelier is an agentic classification workbench for Cloudera AI (CAI). It combines a gRPC core service, React frontend (with XYFlow canvas + embedding-atlas), and Claude Agent SDK orchestration. Deploys as a CAI Application from `https://github.com/zndx/atelier`.

## Naming: CAI vs CML vs CDSW

Cloudera's ML platform has been rebranded over time. Use the correct name in context:

- **CAI** (Cloudera AI) — Current name. Use this in user-facing docs, README, comments.
- **CML** (Cloudera Machine Learning) — Previous name. Still appears in some Cloudera repos and docs.
- **CDSW** (Cloudera Data Science Workbench) — Legacy name. Persists in environment variables (`CDSW_APP_PORT`, `CDSW_PROJECT_ID`, `CDSW_DOMAIN`) which are set by the platform at runtime. Do not rename these — they are platform-provided.

When writing new code or docs, prefer "CAI". When referencing env vars, use the actual `CDSW_*` names the platform provides.

## Architecture

- **gRPC Core** (`src/atelier/`) — Proto-first API (Fine Tuning Studio pattern). Servicer is a thin router; logic in separate modules.
- **HTTP Gateway** (`src/atelier/gateway.py`) — FastAPI bridging REST→gRPC, serves compiled React in production.
- **React Frontend** (`ui/`) — Vite + React 19 + Ant Design + @xyflow/react. Dev server on :3000 proxies /api to :8090.
- **PostgreSQL** — State persistence. devenv `services.postgres` (PG 16 + pgvector, port 5533) for local dev; PGlite (Node.js process, `scripts/pglite-server.mjs`) for CAI when no external PG is available.
- **Qdrant** — Vector store. devenv `pkgs.qdrant` process for local dev; binary download for CAI.
- **HOCON Config** (`config/base.conf`) — Single source of truth. Materializes to `build/config/atelier.env` for `env -i` consumption.
- **Submodules** — `external/embedding-atlas` (fork), `external/hermes-agent` (fork). Dev-only, not used in CAI deployment.

## Development (devenv-first)

The project uses devenv as the primary development environment. `devenv up` starts the full stack (PostgreSQL, Qdrant, gRPC, gateway, Vite).

```bash
devenv shell              # Enter dev environment
devenv up                 # Start everything
```

### just recipes (convenience wrappers)

```bash
just install              # uv sync + pnpm install
just proto                # Generate proto stubs from atelier.proto
just resolve-config       # Materialize HOCON → build/config/atelier.env
just preflight            # Validate materialized config
just migrate              # Run dbmate migrations
just up                   # devenv up
just start                # Production-like startup (no devenv)
just build-ui             # Build React → ui/dist/
just test                 # pytest
just docs-serve           # mdbook serve docs/
```

## Three-tier compatibility

1. **devenv** — Full local dev (Nix-managed services, process manager, dotenv)
2. **just** — Portable task runner (works anywhere just + uv are available)
3. **Python/bash scripts** — Maximum compatibility for CAI (no devenv or just)

CAI deployment uses tier 3 exclusively: `scripts/install_deps.py`, `scripts/startup_app.py`, `bin/start-app.sh`.

## Config Pattern

HOCON (`config/base.conf`) captures env vars via `${?VAR}` substitution. No module reads `os.environ` directly for config values. Precedence: CLI args > env vars > base.conf defaults.

Workflow: `just resolve-config` → `just preflight` → `devenv up`

## Proto-First Development

1. Edit `src/atelier/proto/atelier.proto`
2. Run `just proto` to regenerate stubs
3. Implement methods in `src/atelier/service.py`
4. Add REST endpoints in `src/atelier/gateway.py`

## Key Files

- `devenv.nix` — Dev environment definition (services, processes, packages)
- `config/base.conf` — HOCON config (source of truth)
- `src/atelier/proto/atelier.proto` — gRPC service contract
- `src/atelier/service.py` — gRPC servicer (router)
- `src/atelier/gateway.py` — FastAPI HTTP gateway
- `src/atelier/config.py` — Config loading, materialization, validation
- `src/atelier/db/bootstrap.py` — Migration runner for CAI (dbmate-compatible)
- `db/migrations/` — dbmate-compatible SQL migrations
- `.project-metadata.yaml` — CAI AMP deployment metadata
- `bin/start-app.sh` — Production service orchestrator
- `scripts/startup_app.py` — CAI entry point (restart loop)

## Branch Convention

- Main branch: `trunk`
