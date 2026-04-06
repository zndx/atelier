# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atelier is an agentic classification workbench for Cloudera AI. It combines a gRPC core service, React frontend (with XYFlow canvas + embedding-atlas), and Claude Agent SDK orchestration. Deploys as a CAI Application from `https://github.com/zndx/atelier`.

## Architecture

- **gRPC Core** (`src/atelier/`) — Proto-first API (Fine Tuning Studio pattern). Servicer is a thin router; logic in separate modules.
- **HTTP Gateway** (`src/atelier/gateway.py`) — FastAPI bridging REST→gRPC, serves compiled React in production.
- **React Frontend** (`ui/`) — Vite + React 19 + Ant Design + @xyflow/react. Dev server on :3000 proxies /api to :8090.
- **HOCON Config** (`config/base.conf`) — Single source of truth. Materializes to `build/config/atelier.env` for `env -i` consumption.
- **Submodules** — `external/embedding-atlas` (fork), `external/hermes-agent` (fork).

## Common Commands

```bash
just install          # uv sync + pnpm install
just proto            # Generate proto stubs from atelier.proto
just resolve-config   # Materialize HOCON → build/config/atelier.env
just preflight        # Validate materialized config
just show-config      # Print resolved config
just up               # devenv up (gRPC + Vite dev server)
just grpc             # gRPC server only (via env -i)
just gateway          # FastAPI gateway only (via env -i)
just ui               # Vite dev server only
just build-ui         # Build React → ui/dist/
just test             # pytest
just docs-build       # mdbook build docs/
just docs-serve       # mdbook serve docs/
```

## Config Pattern

HOCON (`config/base.conf`) captures env vars via `${?VAR}` substitution. No module reads `os.environ` directly for config values. Precedence: CLI args > env vars > base.conf defaults.

Workflow: `just resolve-config` → `just preflight` → `just up`

## Proto-First Development

1. Edit `src/atelier/proto/atelier.proto`
2. Run `just proto` to regenerate stubs
3. Implement methods in `src/atelier/service.py`
4. Add REST endpoints in `src/atelier/gateway.py`

## Key Files

- `config/base.conf` — HOCON config (source of truth)
- `src/atelier/proto/atelier.proto` — gRPC service contract
- `src/atelier/service.py` — gRPC servicer (router)
- `src/atelier/gateway.py` — FastAPI HTTP gateway
- `src/atelier/config.py` — Config loading, materialization, validation
- `.project-metadata.yaml` — CAI AMP deployment metadata
- `bin/start-app.sh` — Production service orchestrator

## Branch Convention

- Main branch: `trunk`
