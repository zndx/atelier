<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Atelier is an agentic classification workbench for Cloudera AI (CAI). It combines a gRPC core service, React frontend (with XYFlow canvas + Embeddings), and Claude Agent SDK orchestration. Deploys as a CAI Application from `https://github.com/zndx/atelier`.

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
- **PostgreSQL** — State persistence. devenv `services.postgres` (PG 16 + pgvector, port **5533**) for local dev; PGlite (Node.js process, `scripts/pglite-server.mjs`) on port **5440** for CAI deployments when no external PG is available. The two ports are NOT interchangeable; the CAI pod has nothing on 5533. Code that needs the live URL reads `ATELIER_DB_URL` (exported by `bin/start-app.sh` line 278) — never hardcode the port.
- **Qdrant** — Vector store. devenv `pkgs.qdrant` process for local dev; binary download for CAI.
- **HOCON Config** (`config/base.conf`) — Single source of truth. Materializes to `build/config/atelier.env` for `env -i` consumption.
- **Submodules** — `external/embedding-atlas` ([fork](https://github.com/rch/oss-embedding-atlas) with important modifications, used in both dev and CAI deployment — pre-built dist/ committed to the fork), `external/hermes-agent` (fork, dev-only reference).

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

## Running inside the CAI Application pod

The Claude Agent SDK terminal embedded in the deployed Atelier app
runs *inside* the CAI Application pod and shares its process group +
filesystem with the live gateway, gRPC servicer, PGlite, and Qdrant.
Mistakes here crash the operator's session in their browser.  Read
this section before touching processes or ports.

### Ports in this environment

| Service | devenv (local) | **CAI Application pod** |
|---|---|---|
| HTTP gateway | 8090 | `$CDSW_APP_PORT` (= 8090, but managed by the platform) |
| gRPC servicer | 50051 | 50051 |
| PostgreSQL | 5533 (devenv `services.postgres`) | **PGlite on 5440** — port 5533 is unused in CAI |
| Qdrant HTTP | 6333 | 6333 |

The PGlite port (`5440`) is hardcoded in `bin/start-app.sh:251` and
deliberately avoids CAI's platform Postgres on 5432.  Anything that
*reads* the URL should read `os.environ["ATELIER_DB_URL"]` — exported
by the same script at line 278 — and never assume the devenv default.

### Processes you must NOT kill

`bin/start-app.sh` runs once at AMP startup and includes its own
`kill_stale_processes()` that `pkill -f`s `pglite-server.mjs`,
`qdrant/qdrant`, `atelier.server`, `atelier.gateway`.  That helper
is intended for AMP-lifecycle cleanup, not for agent use mid-session.
Inside a running pod:

- **`pkill -f pglite-server.mjs`** kills the live database. The
  gateway's next DB call hangs or 500s; the operator sees a broken
  Embeddings page until the AMP is restarted.
- **`pkill -f atelier.gateway`** kills the HTTP server the operator
  is talking to. Their browser session goes white.
- **`pkill -f atelier.server`** kills the gRPC servicer the gateway
  proxies to. Same blast radius.
- **Re-running `bin/start-app.sh`** from a live pod is destructive
  in the same way — the script's first action is to kill everything
  it expects to start.

The supervisor owning these processes is the AMP runtime, not the
agent.  To pick up code changes, the operator restarts the
Application from the CAI Workspace UI.

### Diagnosing DB issues from inside the pod

Symptom: `connection failed: connection to server at "127.0.0.1",
port 5533 failed: Connection refused`.  Cause: code (or an agent
session) is reading the devenv default; CAI uses 5440.  Fix: read
`ATELIER_DB_URL` instead of constructing a URL.

To inspect the live DB safely (read-only):

```bash
python -c "
import sys; sys.path.insert(0, 'src')
from atelier.db.dao import AtelierDao
print(AtelierDao().list_datasets())
"
```

The DAO reads `ATELIER_DB_URL` automatically; no port assumptions.

### Filesystem realities

- Project root is `/home/cdsw` (CAI convention; matches `$HOME`).
- PGlite data lives at `.app/pgdata/`; PGlite log at `.app/pglite.log`.
- `build/results/<run_id>/` accumulates per-run artifacts (parquet,
  classifications.json, model files); nothing here is committed.
- `/home/cdsw` is typically NFS-shared with the CAI Session pod
  hosting the maintainer's IDE/Jupyter — file edits made elsewhere
  appear here without warning.  Cross-check `git status` before
  attributing diffs to your own session.

## Config Pattern

HOCON (`config/base.conf`) captures env vars via `${?VAR}` substitution. No module reads `os.environ` directly for config values. Precedence: CLI args > env vars > base.conf defaults.

Workflow: `just resolve-config` → `just preflight` → `devenv up`

### Runtime overlay (Settings page)

`src/atelier/config_overlay.py` provides an in-memory overlay applied inside
`run_classification_pipeline` via `apply_to_config(cfg)`. The `/settings` UI
(gear icon, top-right of the header) reads/writes this overlay through
`GET/PATCH /api/settings` and `POST /api/settings/reset`. Overlay keys must
match `AtelierConfig` dataclass field names and validate against
`SETTINGS_METADATA` (choice enums, float ranges). The overlay is
session-only — it resets when the gateway process restarts. For permanent
tuning, edit `config/base.conf` or set the corresponding env var.

## Classification Pipeline

The core of the project lives in `src/atelier/classify/` and is driven by
`run_classification_pipeline()` in `pipeline.py`. Triggered from the UI via
`POST /api/fsm/start` (gateway) → FSM-tracked run → results written to
`build/results/{run_id}/`.

Key concepts worth internalizing before editing:

- **Dempster-Shafer evidence fusion** (`belief.py`, `mass_functions.py`) —
  up to 6 evidence sources (name-match, pattern, cosine, LLM, CatBoost,
  SVM) combined into a `HierarchicalClassification` with belief,
  plausibility, and conflict per code. Fusion strategy is configurable —
  `dempster` normalizes conflict by `(1−K)`, `yager` redirects conflict
  mass to Θ (ignorance).
- **Belief-gap convergence** (`bootstrap.py`) — the bootstrap loop
  converges on `mean(Pl − Bel)`, not on K (conflict). Gap is the primary
  signal; K is diagnostic.
- **Bootstrap loop**: LLM sweep → ML validation → revisit disagreements
  until gap threshold / bel-floor / max-iterations reached.  Earlier
  revisions ran an M9 in-loop SVM-on-LLM-labels retrain (historical
  function name ``train_svm_on_frontier_labels``); that path was
  excised on 2026-05-04 for Denoeux-2008 source-independence reasons
  (see `docs/src/architecture/dst-evidence-independence.md`).  SVM
  is now trained offline on the synth corpus; going forward,
  SVM-on-synthetic via the procedural-ML stack (P5).
- **Monte Carlo stratification** (`monte_carlo.py`, `row_sampler.py`) —
  for large corpora, a stratified subset is directly LLM-classified;
  the remainder receives label propagation with an elevated discount.
  The MCPlan's ``sampled_columns`` set is the directly-classified
  subset; ``propagation_columns`` is the remainder.
- **Terminology note**: the term *frontier* is reserved for the Pareto
  sense (see `docs/src/architecture/pareto-capability-evolution.md`)
  and the AI-industry "frontier model" sense (capability-leading LLMs).
  It must NOT be used to describe MC-sampled columns, LLM-classified
  labels, or the excised M9 SVM retrain — prefer "sampled",
  "directly-classified", or "LLM-classified" instead.  This rule keeps
  ``frontier`` distinct enough to carry its capability-evolution
  meaning unambiguously.
- **FSM** (`fsm.py`) — authoritative state machine; every phase advances
  through `LOADING_VOCAB → DISCOVERING → SAMPLING → LLM_SWEEP →
  VALIDATING → CLASSIFYING → FUSING → EVALUATING → CONVERGED|ERROR`.

Terminology (Atlas Lexicon — use in UI / docs): **entities** (not rows),
**terms** (not categories), **classifications** (applied tags).

## Overwatch & Governance (optional capabilities)

- **Overwatch** (`src/atelier/overwatch/agent.py`) — single-turn Opus
  analysis that writes `build/results/{run_id}/overwatch.md` with
  pipeline recommendations. Follows the Web Terminal Agent's selected
  model (direct Anthropic API or Bedrock) — single source of truth so
  operators have one provider knob instead of two.  Gated by
  `cfg.has_overwatch` (which is now `overwatch.enabled AND any WTA
  catalog entry is runnable`). Triggered at the end of a pipeline run
  when `overwatch.enabled = true`.
- **Governance** (`src/atelier/governance/`) — optional Atlas sync
  (taxonomy → classification types; results → entity tags). Gated by
  `cfg.governance_auto_sync && cfg.has_atlas`. Knox-proxied Atlas auth
  uses `HTTPBasicAuth` + `CDPUrlResolver` (not `cdpcurl`, which is
  control-plane only).

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

## BDD Testing (behave)

Feature files in `features/` organized by domain: infra, gateway, deployment, agent.
Domain step definitions live in `<domain>/step_defs/` (not `steps/`) to avoid
behave auto-discovery conflicts. Re-exported via `features/steps/__init__.py`.
**Important:** Never name a features/ subdirectory after a stdlib module (e.g., `platform`).

Tier system controls what runs:
- `just behave` — canonical BDD entry point; tier-0 + tier-1, excludes
  `@slow` (auto-starts devenv if the stack isn't up)
- `just behave-slow` — adds `@slow` scenarios (pipeline convergence,
  ML training)
- `@tier-cai` scenarios are documentation-only; skipped locally
- `@gpu` scenarios skip automatically when no CUDA device is present

Scenarios should model user/operator workflows, not implementation details.
Import checks and unit-level assertions belong in pytest, not BDD.
Heavy synth-scale validation (full 9782-col × 512-perm SAGE) is triggered
via UI pipeline runs, not BDD.

Run a single feature or scenario:

```bash
uv run behave features/agent/settings.feature              # one feature
uv run behave features/agent/settings.feature -n "Reset"   # name regex
uv run behave --tags @tier-0 --tags @agent                 # tag filter
```

CAI Runtime Profile (`features/deployment/runtime_profile.feature`) validates
deployment readiness without CAI access. Run before every push.

## GPU Acceleration

`preflight_gpu()` (`src/atelier/classify/gpu.py`) probes nvidia-smi +
`torch.cuda` at pipeline start. When CUDA is available, the pipeline
auto-routes SAGE, PermutationSHAP, CatBoost training, and (optional)
UMAP 2D projection onto GPU kernels — full CPU fallback preserved.

- **SAGE / SHAP**: `src/atelier/classify/gpu_importance.py` — custom
  vectorized kernel with fixed global donors, precomputed embedding
  cache, and chunk-batched losses via `torch.matmul`. Replaces
  `sage-importance` and `shap.PermutationExplainer` on GPU hosts; both
  libraries remain as CPU fallbacks.
- **Multi-GPU**: `MultiDeviceEncoder` lazy-loads replicas; shard_threshold
  defaults to 200K because MiniLM-L6 saturates a single 4090 before
  GIL-bound thread coordination pays off. Lower it for larger embedding
  models (BGE-large, E5-mistral).
- **RAPIDS extra**: `uv sync --extra gpu` installs `cuml` + `cupy` for
  `cuml.UMAP`. Pipeline falls back to `umap-learn` when absent.
- **Settings**: `classify.gpu.{enabled,shard_threshold,sage_chunk_permutations}`
  in `config/base.conf`. Runtime status at `GET /api/acceleration`;
  visible in the UI's Settings page.
- **SAGE auto-enable**: on GPU hosts, SAGE is auto-enabled even when
  `classify.sage.enabled = false` (the kernel is fast enough to be
  default-on). CPU hosts keep the opt-in behavior (too slow otherwise).

See [`docs/src/architecture/gpu-acceleration.md`](docs/src/architecture/gpu-acceleration.md)
for full design notes.

## Model Defaults

`agents.model` in `config/base.conf` tracks the latest Opus on the
Anthropic direct API (currently `claude-opus-4-7`).  Bedrock
deployments override via `ATELIER_AGENT_MODEL` with a Bedrock ARN —
Bedrock lags direct-API releases, so the two are not kept in
lockstep.

Overwatch has no separate model setting — it follows whatever the
operator selects in the Web Terminal Agent picker (which itself
defaults to `cfg.agent_model`).  This keeps provider+model selection
to a single, intuitive knob.

## Branch Convention

- Main branch: `trunk`
