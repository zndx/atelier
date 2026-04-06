# Atelier

Agentic classification workbench for Cloudera AI. Combines the Claude Agent SDK for adaptive keystone-agent orchestration with embedding-atlas for interactive visualization of classification results produced by the [signals](https://github.com/rch/signals) pipeline.

## Development Environment

Atelier is **devenv-first**. [devenv](https://devenv.sh) provides a reproducible Nix-based development shell with all system dependencies (Python 3.12, Node.js 22, PostgreSQL 16, Qdrant, dbmate, protobuf, grpcurl, mdbook, etc.) and a process manager that starts the full stack in one command.

```bash
devenv shell              # Enter the dev environment (loads .env automatically)
devenv up                 # Start everything: PostgreSQL, Qdrant, gRPC, gateway, Vite
```

That's it. Visit http://localhost:3000.

### What `devenv up` starts

| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL 16 | 5533 | State database (with pgvector) |
| Qdrant | 6333 / 6334 | Vector store (HTTP / gRPC) |
| gRPC server | 50051 | Core service |
| FastAPI gateway | 8090 | REST-to-gRPC bridge |
| Vite dev server | 3000 | React UI with hot reload |

### First-time setup

On first clone, install dependencies and initialize:

```bash
devenv shell
just install              # uv sync + pnpm install
just proto                # Generate gRPC stubs from atelier.proto
just resolve-config       # Materialize HOCON → build/config/atelier.env
devenv up                 # Start all services (postgres initializes on first run)
# In another terminal:
just migrate              # Apply database migrations via dbmate
```

### devenv utilities

devenv provides more than just the process manager:

| Command | What it does |
|---------|-------------|
| `devenv shell` | Enter the dev environment with all tools on PATH |
| `devenv up` | Start all services and processes |
| `devenv test` | Run the devenv test suite |
| `devenv info` | Show environment info and available services |

The `.env` file is loaded automatically via `dotenv.enable = true`. Copy `.env.example` to `.env` for local overrides.

### just recipes

`just` provides task shortcuts that complement devenv. These are convenience wrappers, not replacements for devenv itself.

| Recipe | Description |
|--------|-------------|
| `just up` | Alias for `devenv up` |
| `just install` | `uv sync && cd ui && pnpm install` |
| `just proto` | Generate proto stubs |
| `just migrate` | Run dbmate migrations |
| `just resolve-config` | Materialize HOCON config |
| `just build-ui` | Build React → `ui/dist/` |
| `just start` | Production-like startup (no devenv required) |
| `just docs-serve` | mdbook with live reload |

### Git submodules

The `external/` directory contains development reference submodules (embedding-atlas, hermes-agent). These are for local development only and are **not required** for deployment.

```bash
git submodule update --init --recursive   # Optional: fetch submodule sources
```

## Deploying to Cloudera AI

There are two ways to deploy Atelier on Cloudera AI (CML): as an **AMP** (automated) or as a manual **Application**. devenv is not used in CML — the deployment scripts handle all infrastructure (pgserver for embedded PostgreSQL, Qdrant binary download). Git submodules are not cloned and not needed.

### Option 1: AMP Deployment (Recommended)

AMPs (Applied ML Prototypes) use `.project-metadata.yaml` to automate the full setup.

1. In the CML UI, go to **AMPs** or create a new Project from Git URL
2. Enter the repository URL: `https://github.com/zndx/atelier`
3. CML parses `.project-metadata.yaml` and presents the defined tasks
4. Run the tasks in order:
   - **Install Dependencies** — installs uv, Python deps, Node.js deps, builds React, downloads Qdrant binary
   - **Atelier** (start_application) — launches Qdrant, gRPC server (with embedded PostgreSQL via pgserver), and HTTP gateway on `CDSW_APP_PORT`
5. Access the application at `https://atelier.<CDSW_DOMAIN>`

### Option 2: Manual Application Deployment

1. Create a new CML Project from Git URL: `https://github.com/zndx/atelier`
2. Open a **Session** (Python 3.10 kernel) and run:
   ```bash
   !pip3 install uv
   !uv sync --frozen
   !cd ui && npm install && npm run build
   !bash scripts/install_qdrant.sh
   ```
3. Go to **Applications > New Application** and configure:
   - **Name:** Atelier
   - **Subdomain:** atelier
   - **Script:** `scripts/startup_app.py`
   - **Kernel:** Python 3
   - **CPU:** 2 cores, **Memory:** 4 GB
4. Start the application. It will bind to `CDSW_APP_PORT` automatically.

### Entry Point

The entry point for both methods is **`scripts/startup_app.py`**:
- Starts Qdrant (background), gRPC with pgserver auto-bootstrap + migrations (background), FastAPI gateway (foreground)
- Binds to `127.0.0.1:$CDSW_APP_PORT` (CML's reverse proxy handles external routing)
- Wraps execution in a restart loop for resilience

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude Agent SDK | (none) |
| `UV_HTTP_TIMEOUT` | uv HTTP request timeout in ms | 60000 |
| `ATELIER_DB_URL` | PostgreSQL connection URI (overrides pgserver) | auto (pgserver) |
| `QDRANT_HOST` | Qdrant hostname | localhost |
| `QDRANT_PORT` | Qdrant HTTP port | 6333 |

### Local vs CML Infrastructure

| Component | Local (devenv) | CML |
|-----------|---------------|-----|
| PostgreSQL | `services.postgres` (PG 16 + pgvector, port 5533) | pgserver (pip-installed embedded PG) |
| Qdrant | `pkgs.qdrant` (devenv process) | Binary download from GitHub releases |
| Migrations | `just migrate` (dbmate CLI) | Auto-applied on startup via SQLAlchemy |
| Node.js | pnpm (via devenv) | npm (CML base image) |
| Git submodules | Available for development | Not cloned, not needed |

## Architecture

- **gRPC Core Service** — Proto-first API (port 50051)
- **FastAPI HTTP Gateway** — Serves React build + bridges REST to gRPC
- **React Frontend** — Ant Design UI with XYFlow canvas for agent workflows
- **PostgreSQL** — State persistence (devenv or pgserver embedded)
- **Qdrant** — Vector store for embedding search
- **Claude Agent SDK** — Keystone agents for classification orchestration
- **embedding-atlas** — Interactive parquet visualization
- **HOCON Configuration** — Single source of truth with env var substitution

## Documentation

```bash
just docs-serve       # mdbook at localhost:3000
```
