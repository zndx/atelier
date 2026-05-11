<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# gRPC & Gateway

Atelier follows the Fine Tuning Studio proto-first pattern: the gRPC
service contract defines the API, and a FastAPI gateway bridges REST
to gRPC while serving the React frontend.

## Proto Definition

The service contract lives in `src/atelier/proto/atelier.proto`.

### RPCs

| RPC | Request → Response | Purpose |
|-----|-------------------|---------|
| `HealthCheck` | `HealthCheckRequest` → `HealthCheckResponse` | Prove gRPC is alive (status + version) |
| `ListAgents` | `ListAgentsRequest` → `ListAgentsResponse` | List agent metadata (id, name, role, tools) |
| `GetAgent` | `GetAgentRequest` → `GetAgentResponse` | Single agent by ID |
| `ListDataSources` | `ListDataSourcesRequest` → `ListDataSourcesResponse` | List OOTB + Hive sources |
| `ListDatasets` | `ListDatasetsRequest` → `ListDatasetsResponse` | Classification datasets (filterable by source_id) |
| `GetFSMStatus` | `FSMStatusRequest` → `FSMStatusResponse` | Pipeline state + progress JSON |
| `StartClassification` | `StartClassificationRequest` → `StartClassificationResponse` | Trigger a classification run |

### Key Messages

- **`DataSource`** — id, source_type (`sample`/`hive`), source_uri, display_name, vocabulary_mode
- **`ClassificationDataset`** — id, name, parquet_path, source_id, version_number, is_active, summary
- **`FSMStatusResponse`** — run_id, state, started_at, progress_json, error
- **`AgentMetadata`** — id, name, description, role, tool_ids

### Generating Stubs

```bash
just proto    # runs bin/generate-proto.sh
```

This invokes `grpc_tools.protoc` to produce `_pb2.py`, `_pb2_grpc.py`,
and `.pyi` type stubs.

## Architecture Layers

```
Proto (atelier.proto)     ← Service contract and message definitions
    ↓
Servicer (service.py)     ← Thin router dispatching to business logic
    ↓
Client (client.py)        ← Wrapper around generated stub with error handling
    ↓
Gateway (gateway.py)      ← FastAPI bridge from REST to gRPC + React SPA
```

## Gateway REST Endpoints

### Infrastructure

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | gRPC health check |
| `/api/status` | GET | Aggregated health: gRPC + PostgreSQL + Qdrant + config state |
| `/api/agents/validate-credentials` | POST | Test all configured LLM providers |
| `/api/agents/model-discovery` | GET | Check for model upgrades via Anthropic Models API |

### Data Sources & Datasets

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/data-sources` | GET | List registered data sources |
| `/api/datasets` | GET | List datasets (optional `source_id` filter) |
| `/api/datasets/{id}/activate` | POST | Set dataset version as active |
| `/api/datasets/{id}/data` | GET | Serve parquet file |
| `/api/data-connections` | GET | List CAI data connections |
| `/api/data-connections/{name}/test` | POST | Test a CAI connection |
| `/api/vocabulary/stats` | GET | Term count (source-aware routing) |

### Classification Pipeline

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/fsm/status` | GET | Current pipeline state + progress |
| `/api/fsm/start` | POST | Start classification (optional `source_id`) |
| `/api/fsm/runs` | GET | List past classification runs |

### Agents & Skills

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/agents` | GET | List agent metadata |
| `/api/skills` | GET | Skill definitions from `.claude/commands/` |
| `/api/skills/{skill_id}` | GET | Single skill markdown content |
| `/api/agents/smoke-test` | POST | Minimal Claude Agent SDK verification |

### WebSocket

| Endpoint | Purpose |
|----------|---------|
| `/ws/terminal/{session_id}` | Persistent terminal backed by Claude Agent SDK |
| `/ws/orchestration` | Live agent events (spawned, reasoning, tool_call, completed) |

#### Persistent Terminal Sessions

Terminal sessions survive page navigation and browser reload. The WebSocket
endpoint accepts a client-provided `session_id` (persisted in localStorage).
On disconnect, the session stays alive server-side — SDK queries continue
running and output accumulates in a **ring buffer** (64KB `collections.deque`).
On reconnect, the buffer is replayed so the user sees everything that happened
while they were away.

- **Session registry**: Module-level `_sessions` dict in `terminal.py`
- **Idle cleanup**: Background asyncio task sweeps sessions with no client
  for 30 minutes (`/api/terminal/sessions` lists active sessions)
- **Dedicated page**: `/terminal` route renders a full-screen Ghostty WASM
  terminal; the Landing page embeds the same component at preview size

### SPA Fallback

`/{path}` serves `ui/dist/index.html` for client-side routing.

## Aggregated Status Endpoint

`GET /api/status` returns a comprehensive health report:

```json
{
  "grpc": {"status": "ok", "latency_ms": 12},
  "postgres": {"status": "ok"},
  "qdrant": {"status": "ok"},
  "config": {
    "has_anthropic": true,
    "has_bedrock": false,
    "agent_model": "claude-sonnet-4-5-20250929",
    "db_url": "postgresql://...(masked)"
  },
  "overall_status": "connected"
}
```

PostgreSQL probes retry 3x with 1s backoff (PGlite can have transient stalls).
Overall status is `connected` when gRPC responds, `degraded` when gRPC is up
but other services are flaky.

## Gateway Lifespan

The FastAPI `lifespan` hook runs three startup tasks:

1. **OOTB seed**: Check if `ootb-sample` source has any dataset versions;
   if none, create version 1 with metadata.
2. **Hive auto-discovery**: `discover_hive_sources()` probes all configured
   data connections (`ATELIER_DATA_CONNECTIONS`), iterates databases, finds
   `annotations` tables matching the known schema (legacy or universal format),
   and auto-registers them via `get_or_create_data_source()`.
3. **Terminal cleanup**: Background asyncio task sweeps idle terminal sessions
   every 60 seconds.

All three tasks are wrapped in try/except — failures are logged as warnings
but don't prevent gateway startup.

## Config Lifecycle

HOCON (`config/base.conf`) is the single source of truth. No module reads
`os.environ` directly for configuration values.

```
.env → devenv shell → HOCON ${?VAR} substitution → AtelierConfig dataclass
```

`load_config()` reads the HOCON file with live environment variable
substitution. External tools that need a flat key=value file use
`just resolve-config` to materialize `build/config/atelier.env`.

## Preflight Validation

`just preflight` runs structured deny/warn checks via
`atelier.preflight.run_preflight()`:

- **Deny** = blocking (service cannot start). Examples: missing API keys
  when both Anthropic and Bedrock are unconfigured.
- **Warn** = advisory (degraded functionality). Examples: GPU detected but
  CUDA unavailable, Qdrant not reachable.

Preflight is called during gateway startup to surface configuration
problems early rather than during the first pipeline run.
