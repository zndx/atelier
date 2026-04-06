# Atelier

Agentic classification workbench for Cloudera AI. Combines the Claude Agent SDK for adaptive keystone-agent orchestration with embedding-atlas for interactive visualization of classification results produced by the [signals](https://github.com/rch/signals) pipeline.

## Quick Start (Local Development)

```bash
devenv shell          # Enter dev environment (loads .env automatically)
just install          # Install Python + Node dependencies
just proto            # Generate proto stubs
just resolve-config   # Materialize HOCON → build/config/atelier.env
just up               # Start gRPC + gateway + Vite dev server
```

Visit http://localhost:3000 for the React UI (hot reload, proxies `/api` to gateway).

### Production-like local test

```bash
just build-ui         # Build React → ui/dist/
just start            # gRPC + gateway on :8090 (serves built UI)
```

## Deploying to Cloudera AI

There are two ways to deploy Atelier on Cloudera AI (CML): as an **AMP** (automated) or as a manual **Application**.

### Option 1: AMP Deployment (Recommended)

AMPs (Applied ML Prototypes) use `.project-metadata.yaml` to automate the full setup. This is the one-click path.

1. In the CML UI, go to **AMPs** or create a new Project from Git URL
2. Enter the repository URL: `https://github.com/zndx/atelier`
3. CML parses `.project-metadata.yaml` and presents the defined tasks
4. Run the tasks in order:
   - **Install Dependencies** — installs Python (via uv) and Node.js deps, builds the React frontend
   - **Atelier** (start_application) — launches gRPC server + HTTP gateway on `CDSW_APP_PORT`
5. Access the application at `https://atelier.<CDSW_DOMAIN>`

Environment variables to configure (optional):

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API key for Claude Agent SDK | (none) |
| `UV_HTTP_TIMEOUT` | uv HTTP request timeout in ms | 60000 |

### Option 2: Manual Application Deployment

If you prefer to set up the project manually or need more control:

1. Create a new CML Project from Git URL: `https://github.com/zndx/atelier`
2. Open a **Session** (Python 3.10 kernel) and run:
   ```bash
   !pip3 install uv
   !uv sync --frozen
   !cd ui && npm install && npm run build
   ```
3. Go to **Applications > New Application** and configure:
   - **Name:** Atelier
   - **Subdomain:** atelier
   - **Script:** `scripts/startup_app.py`
   - **Kernel:** Python 3
   - **CPU:** 2 cores, **Memory:** 4 GB
4. Start the application. It will bind to `CDSW_APP_PORT` automatically.

### Entry Point

The entry point for both deployment methods is **`scripts/startup_app.py`**. This script:
- Detects the CML environment (`CDSW_APP_PORT`, `IS_COMPOSABLE`)
- Calls `bin/start-app.sh` which starts gRPC (background) + FastAPI gateway (foreground)
- Wraps execution in a restart loop for resilience
- Binds to `127.0.0.1:$CDSW_APP_PORT` (CML's reverse proxy handles external routing)

## Architecture

- **gRPC Core Service** — Proto-first API (port 50051)
- **FastAPI HTTP Gateway** — Serves React build + bridges REST to gRPC
- **React Frontend** — Ant Design UI with XYFlow canvas for agent workflows
- **Claude Agent SDK** — Keystone agents for classification orchestration
- **embedding-atlas** — Interactive parquet visualization
- **HOCON Configuration** — Single source of truth with env var substitution

## Documentation

```bash
just docs-serve       # mdbook at localhost:3000
```
