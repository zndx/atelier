# Atelier

Agentic classification workbench for Cloudera AI. Combines the Claude Agent SDK for adaptive keystone-agent orchestration with an Embeddings for interactive visualization of classification results produced by the [signals](https://github.com/rch/signals) pipeline.

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
| gRPC server | 50071 | Product servicer (`ATELIER_GRPC_PORT`; CAI default is 50051) |
| FastAPI gateway | 8090 | REST-to-gRPC bridge |
| Vite dev server | 3000 | React UI with hot reload |
| llama.cpp | 8080 | Turn-key classify LLM (skipped if the port is already bound) |

The Signals lattice capability engine (`python -m atelier.engine.server` on
**:50251**) is **not** started by `devenv up`. That is `atelier.service` on
GPU lab hosts — see [Signals Peer Unit](docs/src/operations/peer-unit.md).

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

The `external/` directory contains forked submodules:

- **[embedding-atlas](https://github.com/rch/oss-embedding-atlas)** — Fork of Apple's embedding-atlas with important modifications for Atelier's Embeddings page. Pre-built dist/ is committed to the fork so CAI deployment doesn't need the full build toolchain (Emscripten, Rust, uv). Required for both dev and deployment.
- **sdg-corpora** — Ontology-grounded SDG corpora (ontology → SKOS vocabulary → DDL footprint → generated corpus). The source of classification research work; each commit is a reproducible convergence snapshot.

```bash
git submodule update --init --recursive   # Required: embedding-atlas fork (pre-built dist/)
```

### macOS + Gateway configuration

devenv fully supports macOS (Apple silicon) — Linux-only pieces are
gated behind `pkgs.stdenv.isLinux` in `devenv.nix`, so `devenv up`
brings up the same five-service stack. Platform differences to know:

| Capability | Linux | macOS |
|------------|-------|-------|
| Full stack (`devenv up`) | ✓ | ✓ |
| GPU acceleration (CUDA: SAGE/SHAP kernels, CatBoost, cuML UMAP) | ✓ (NVIDIA host) | — CPU fallbacks |
| d2 diagram rendering in docs (`mdbook-d2`) | ✓ | — (d2's renderer needs linux-only mesa; `mdbook serve` otherwise works) |
| libpq for psycopg | devenv `LD_LIBRARY_PATH` | bundled `psycopg[binary]` wheel (automatic, darwin-only dependency marker) |

For deployments that reach Anthropic through an internal gateway
(e.g. on a VPN) instead of `api.anthropic.com`, configure token auth
in `.env` — either `ANTHROPIC_API_KEY` *or* the token pair below
satisfies credential gating everywhere (Web Terminal Agent catalog,
Overwatch, classify LLM sweep, enrichment):

```bash
ANTHROPIC_AUTH_TOKEN=...                                  # sent as Authorization: Bearer
ANTHROPIC_BASE_URL=https://ai-gateway.example.com         # redirects all direct-API clients + the claude CLI

# Gateways host their own model catalog (often Bedrock-style IDs) —
# pin the agent model and the CLI's internal sub-models to IDs the
# gateway actually serves:
ATELIER_AGENT_MODEL=anthropic.claude-sonnet-4-6
ANTHROPIC_DEFAULT_SONNET_MODEL=anthropic.claude-sonnet-4-6
ANTHROPIC_DEFAULT_HAIKU_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
CLAUDE_CODE_SUBAGENT_MODEL=anthropic.claude-sonnet-4-6
CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1
ENABLE_TOOL_SEARCH=false
```

With a base URL configured, the Web Terminal Agent picker gains a
**Gateway** entry that routes `ATELIER_AGENT_MODEL` through the
gateway. Bedrock-style model IDs route over the Anthropic protocol
when no AWS credentials are present — the gateway proxies them.

To demo the classification pipeline without activating the Agent SDK
surfaces, add `ATELIER_OVERWATCH_ENABLED=false` — otherwise Overwatch
auto-activates as soon as any Anthropic credential appears.

### Local LLM (llama.cpp) — turn-key classification

devenv ships [llama.cpp](https://github.com/ggml-org/llama.cpp) on
both platforms — Metal acceleration on Apple silicon, CPU BLAS on
Linux (for CUDA offload, override `llama-cpp` with `cudaSupport` in
`devenv.nix`). `llama-server` is OpenAI-compatible, so it plugs into
the classify backend with no credentials.

The fully self-contained flow (no external LLM provider, no API key):

```bash
# .env — three lines:
ATELIER_LLAMA_AUTOSTART=1
ATELIER_LLM_BASE_URL=http://localhost:8080/v1
ATELIER_LLM_MODEL=local

devenv up -d
```

That starts the usual five services **plus** a llama.cpp process
serving NVIDIA Nemotron 3 Nano 30B-A3B (MoE, ~3B active parameters —
fast on Metal) via [Unsloth's GGUF quant](https://huggingface.co/unsloth/Nemotron-3-Nano-30B-A3B-GGUF)
— auto-downloaded and cached on first start (~18 GB). NVIDIA doesn't
publish first-party GGUFs; override the quant with `ATELIER_LLAMA_HF`
or serve any local file with `ATELIER_LLAMA_MODEL`.

With the `sdg-corpora` submodule initialized, startup also registers
an **SDG** data source — one relational collection
(`ATELIER_SDG_COLLECTION`, default `research-project`) classified
blind against the SKOS annotations vocabulary. Pick it in the UI's
Data Source selector and start a classification run.

To browse the same sample relationally (with BFO/template provenance
as table comments):

```bash
just sdg-load     # loads schema → data → views into devenv PG (database `sdg`)
psql postgresql://localhost:5533/sdg
```

## Deploying to Cloudera AI

There are two ways to deploy Atelier on Cloudera AI (CML): as an **AMP** (automated) or as a manual **Application**. devenv is not used in CML — the deployment scripts handle all infrastructure (PGlite for embedded PostgreSQL, Qdrant binary download). The embedding-atlas submodule is cloned during install (pre-built dist/ committed to the fork).

### Option 1: AMP Deployment (Recommended)

AMPs (Applied ML Prototypes) use `.project-metadata.yaml` to automate the full setup. Atelier follows the modern `create_job`/`run_job` pattern (same as RAG Studio) — install jobs persist in the CML project and can be re-run from the **Jobs** tab without redeploying the AMP.

1. In the CML UI, go to **AMPs** or create a new Project from Git URL
2. Enter the repository URL: `https://github.com/zndx/atelier`
3. CML parses `.project-metadata.yaml` and runs the tasks in order:

| Step | Type | What it does |
|------|------|-------------|
| Install Dependencies | `create_job` + `run_job` | `pip install -e .` into system Python, PGlite npm deps, `npm run build` for React UI, downloads Qdrant binary |
| Atelier | `start_application` | Launches PGlite, Qdrant, gRPC server, and HTTP gateway on `CDSW_APP_PORT` |

4. Access the application at `https://atelier.<CDSW_DOMAIN>`

> **Re-running install:** If dependencies need refreshing, go to **Jobs > Install Dependencies > Run** — no AMP redeploy required.

### Option 2: Manual Application Deployment

1. Create a new CML Project from Git URL: `https://github.com/zndx/atelier`
2. Open a **Session** (Python 3.10 kernel) and run:
   ```bash
   !pip3 install -e .
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
- Starts PGlite (background, if no `ATELIER_DB_URL` set), Qdrant (background), gRPC with auto-migrations (background), FastAPI gateway (foreground)
- Binds to `127.0.0.1:$CDSW_APP_PORT` (CML's reverse proxy handles external routing)
- Wraps execution in a restart loop for resilience

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_ACCESS_KEY_ID` | Bedrock access key | (none) |
| `AWS_SECRET_ACCESS_KEY` | Bedrock secret | (none) |
| `SOPS_AGE_KEY` | age private key that decrypts `.env.cai.enc` at startup | (none) |
| `ANTHROPIC_API_KEY` | Anthropic direct API key (overwatch, local dev) | (none) |
| `ATELIER_DB_URL` | PostgreSQL connection URI (overrides PGlite) | auto (PGlite) |
| `QDRANT_HOST` | Qdrant hostname | localhost |
| `QDRANT_PORT` | Qdrant HTTP port | 6333 |

CAI operators typically only need the first three — the encrypted
`.env.cai.enc` file bundles every other deployment default. See
**[Operations → Encrypted Deployment Defaults](./docs/src/operations/secrets.md)**
in the mdbook for the full pattern (`just docs-serve` to browse
locally).

### Local vs CML Infrastructure

| Component | Local (devenv) | CML |
|-----------|---------------|-----|
| PostgreSQL | `services.postgres` (PG 16 + pgvector, port 5533) | PGlite (Node.js process, WASM PG + pgvector) |
| Qdrant | `pkgs.qdrant` (devenv process) | Binary download from GitHub releases |
| Migrations | `just migrate` (dbmate CLI) | Auto-applied on startup via SQLAlchemy |
| Node.js | pnpm (via devenv) | npm (CML base image) |
| Git submodules | Available for development | embedding-atlas cloned (pre-built dist/) |

## Architecture

- **gRPC Core Service** — Proto-first API (port 50051)
- **FastAPI HTTP Gateway** — Serves React build + bridges REST to gRPC
- **React Frontend** — Ant Design UI with XYFlow canvas for agent workflows
- **PostgreSQL** — State persistence (devenv or PGlite on CAI)
- **Qdrant** — Vector store for embedding search
- **Claude Agent SDK** — Keystone agents for classification orchestration
- **Embeddings** — Interactive parquet visualization (powered by [embedding-atlas](https://github.com/apple/embedding-atlas))
- **HOCON Configuration** — Single source of truth with env var substitution

## Documentation

```bash
just docs-serve       # mdbook at localhost:3000
```
