"""HTTP gateway — serves React frontend and bridges REST to gRPC.

In production (CML), this is the process that binds to CDSW_APP_PORT.
It serves the compiled React build from ui/dist/ and proxies /api/*
requests to the co-located gRPC server.
"""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from atelier.client import AtelierClient
from atelier.proto import atelier_pb2

app = FastAPI(title="Atelier", version="0.1.0")

_project_root = Path(__file__).resolve().parent.parent.parent

_client: AtelierClient | None = None
_engine = None  # Cached SQLAlchemy engine for /api/status health checks


def _get_client() -> AtelierClient:
    global _client
    if _client is None:
        _client = AtelierClient()
    return _client


def _reset_client() -> None:
    """Drop the cached gRPC client so the next call opens a fresh channel."""
    global _client
    _client = None


def _get_status_engine():
    """Return a cached SQLAlchemy engine for /api/status health checks.

    Creating a new engine per request hits a pglite-socket@0.0.13 bug where
    rapid connect/disconnect cycles cause "server closed the connection
    unexpectedly". Reusing a pooled engine with pool_pre_ping lets us both
    detect stale connections and avoid the setup-loop hazard.
    """
    global _engine
    if _engine is None:
        from sqlalchemy import create_engine
        from atelier.config import load_config
        _engine = create_engine(
            load_config().db_url,
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=0,
            connect_args={"connect_timeout": 10},
        )
    return _engine


def _reset_status_engine() -> None:
    """Drop the cached engine so the next call rebuilds the pool."""
    global _engine
    if _engine is not None:
        try:
            _engine.dispose()
        except Exception:
            pass
    _engine = None


def _error_envelope(detail: str, *, status: int = 503) -> "JSONResponse":
    """Return a JSON error envelope so the frontend never has to parse
    ``Internal Server Error`` as JSON."""
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=status, content={"error": detail})


# ── REST → gRPC bridge ────────────────────────────────────────────


# NOTE: endpoints below are declared `def` (not `async def`) so FastAPI runs
# them in its threadpool. The gRPC stub calls are synchronous blocking
# operations; running them on the event loop would serialize every request
# and hang the gateway if any one call stalls.


@app.get("/api/health")
def health():
    try:
        client = _get_client()
        resp = client.stub.HealthCheck(
            atelier_pb2.HealthCheckRequest(), timeout=5
        )
        return {"status": resp.status, "version": resp.version}
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"gRPC health check failed: {exc}")


@app.get("/api/agents")
def list_agents():
    try:
        client = _get_client()
        resp = client.stub.ListAgents(
            atelier_pb2.ListAgentsRequest(), timeout=5
        )
    except Exception as exc:
        # Surface as JSON envelope so the Workflows page (which calls
        # .json() on the response) can render the error instead of
        # blowing up with "Unexpected token 'I', 'Internal S'...".
        _reset_client()
        return _error_envelope(f"ListAgents failed: {exc}")
    return {
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "role": a.role,
                "tool_ids": list(a.tool_ids),
            }
            for a in resp.agents
        ]
    }


@app.get("/api/skills")
def list_skills():
    """Return skill definitions from .claude/commands/ markdown files."""
    commands_dir = _project_root / ".claude" / "commands"
    skills = []
    if commands_dir.is_dir():
        for md_file in sorted(commands_dir.glob("*.md")):
            content = md_file.read_text()
            # First line is the title (# Title)
            lines = content.strip().splitlines()
            title = lines[0].lstrip("# ").strip() if lines else md_file.stem
            # Second non-empty line is the description
            description = ""
            for line in lines[1:]:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    description = stripped
                    break
            skills.append({
                "id": md_file.stem,
                "title": title,
                "description": description,
                "content": content,
            })
    return {"skills": skills}


@app.get("/api/skills/{skill_id}")
def get_skill(skill_id: str):
    """Return a single skill's markdown content."""
    from fastapi.responses import Response

    md_file = _project_root / ".claude" / "commands" / f"{skill_id}.md"
    if not md_file.exists():
        return Response(status_code=404, content="Skill not found")
    content = md_file.read_text()
    lines = content.strip().splitlines()
    title = lines[0].lstrip("# ").strip() if lines else skill_id
    description = ""
    for line in lines[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            description = stripped
            break
    return {"id": skill_id, "title": title, "description": description, "content": content}


@app.get("/api/data-sources")
def list_data_sources():
    """Return registered data sources (OOTB sample, hive connections)."""
    try:
        client = _get_client()
        resp = client.stub.ListDataSources(
            atelier_pb2.ListDataSourcesRequest(), timeout=5
        )
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"ListDataSources failed: {exc}")
    return {
        "sources": [
            {
                "id": s.id,
                "source_type": s.source_type,
                "source_uri": s.source_uri,
                "display_name": s.display_name,
                "vocabulary_mode": s.vocabulary_mode,
                "created_at": s.created_at,
                "metadata": s.metadata_json,
            }
            for s in resp.sources
        ]
    }


@app.get("/api/datasets")
def list_datasets(source_id: str | None = None):
    try:
        client = _get_client()
        req = atelier_pb2.ListDatasetsRequest()
        if source_id:
            req.source_id = source_id
        resp = client.stub.ListDatasets(req, timeout=5)
    except Exception as exc:
        _reset_client()
        return _error_envelope(f"ListDatasets failed: {exc}")
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "parquet_path": d.parquet_path,
                "row_count": d.row_count,
                "source_id": d.source_id,
                "version_number": d.version_number,
                "is_active": d.is_active,
                "summary": d.summary,
                "fsm_run_id": d.fsm_run_id,
                "created_at": d.created_at,
            }
            for d in resp.datasets
        ]
    }


@app.post("/api/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str):
    """Set a dataset version as active for its source."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
        ds = dao.get_dataset(dataset_id)
        if ds is None:
            return _error_envelope("Dataset not found", status=404)
        if not ds.get("source_id"):
            return _error_envelope("Dataset has no source", status=400)
        dao.set_active_version(ds["source_id"], dataset_id)
        return {"ok": True, "dataset_id": dataset_id, "source_id": ds["source_id"]}
    except Exception as exc:
        return _error_envelope(f"activate_dataset failed: {exc}")


@app.get("/api/datasets/{dataset_id}/data")
def get_dataset_data(dataset_id: str):
    """Serve a dataset's parquet file for the Embeddings page."""
    from fastapi.responses import Response

    client = _get_client()
    resp = client.stub.ListDatasets(
        atelier_pb2.ListDatasetsRequest(), timeout=5
    )
    dataset = next((d for d in resp.datasets if d.id == dataset_id), None)
    if dataset is None:
        return Response(status_code=404, content="Dataset not found")

    parquet_path = Path(dataset.parquet_path)
    if not parquet_path.is_absolute():
        parquet_path = _project_root / parquet_path
    if not parquet_path.exists():
        return Response(status_code=404, content="Parquet file not found")

    return FileResponse(
        str(parquet_path),
        media_type="application/octet-stream",
        filename=f"{dataset_id}.parquet",
    )


# ── Status / health ───────────────────────────────────────────────


@app.get("/api/status")
def status():
    """Aggregated infrastructure health for the operator dashboard.

    Declared as ``def`` (not ``async def``) so FastAPI runs it in a
    threadpool. All three probes below are synchronous blocking calls;
    running them on the event loop would freeze every other request
    when one backend is slow — which on CAI triggers the platform's
    health check to declare the app dead and restart it.
    """
    import time
    import urllib.request

    from atelier.config import load_config

    cfg = load_config()
    checks: dict = {}

    # gRPC server — timeout prevents hang on stale channel
    try:
        t0 = time.monotonic()
        client = _get_client()
        resp = client.stub.HealthCheck(
            atelier_pb2.HealthCheckRequest(), timeout=5
        )
        ms = int((time.monotonic() - t0) * 1000)
        checks["grpc"] = {"ok": True, "version": resp.version, "latency_ms": ms}
    except Exception as e:
        _reset_client()
        checks["grpc"] = {"ok": False, "error": str(e)}

    # PostgreSQL — retry with backoff. PGlite can have transient
    # stalls (pglite-socket@0.0.13 wedge); one timeout shouldn't
    # flip the status card amber. Three attempts with 1s backoff.
    pg_ok = False
    pg_err = None
    for attempt in range(3):
        try:
            t0 = time.monotonic()
            from sqlalchemy import text
            engine = _get_status_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            ms = int((time.monotonic() - t0) * 1000)
            checks["postgres"] = {"ok": True, "latency_ms": ms}
            pg_ok = True
            break
        except Exception as e:
            pg_err = e
            if attempt < 2:
                time.sleep(1)
                _reset_status_engine()

    if not pg_ok:
        _reset_status_engine()
        checks["postgres"] = {"ok": False, "error": str(pg_err)}

    # Qdrant — short timeout
    try:
        t0 = time.monotonic()
        url = f"http://{cfg.qdrant_host}:{cfg.qdrant_http_port}/healthz"
        urllib.request.urlopen(url, timeout=3)
        ms = int((time.monotonic() - t0) * 1000)
        checks["qdrant"] = {"ok": True, "latency_ms": ms}
    except Exception as e:
        checks["qdrant"] = {"ok": False, "error": str(e)}

    # Config state (safe subset — no secrets)
    db_masked = cfg.db_url.split("@")[-1] if "@" in cfg.db_url else cfg.db_url
    model_discovery = None
    try:
        from atelier.agents import check_model_upgrade
        model_discovery = check_model_upgrade(cfg)
    except Exception:
        pass
    checks["config"] = {
        "has_anthropic": cfg.has_anthropic,
        "has_bedrock": cfg.has_bedrock,
        "has_classify_llm": cfg.has_classify_llm,
        "agent_model": cfg.agent_model,
        "qdrant_host": cfg.qdrant_host,
        "qdrant_http_port": cfg.qdrant_http_port,
        "db_url_masked": db_masked,
        "model_discovery": model_discovery,
    }

    # "connected" = gRPC reachable. gRPC is the only strict dealbreaker;
    # the frontend cannot function without it. Postgres and Qdrant can
    # flake transiently (PGlite wedges, Qdrant cold-start), but that
    # shouldn't mark the whole app as disconnected on the Landing page.
    # We separately compute "degraded" so the dashboard can show a more
    # useful state than a binary badge.
    grpc_ok = checks.get("grpc", {}).get("ok", False)
    all_ok = all(
        checks.get(svc, {}).get("ok", False)
        for svc in ("grpc", "postgres", "qdrant")
    )
    connected = grpc_ok
    degraded = grpc_ok and not all_ok

    return {**checks, "connected": connected, "degraded": degraded}


# ── Agent SDK endpoints ────────────────────────────────────────────


@app.post("/api/agents/validate-credentials")
def validate_credentials():
    """Validate all configured LLM provider credentials.

    Synchronous (runs in threadpool) — the validation does blocking
    network calls to Anthropic/Bedrock which must not block the event
    loop.
    """
    try:
        from atelier.agents import validate_credentials as _validate
    except ImportError:
        return {"any_valid": False, "error": "agents extra not installed (pip install atelier[agents])"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        return _validate(cfg)
    except Exception as exc:
        return {"any_valid": False, "providers": {}, "configured": [], "error": str(exc)}


@app.post("/api/agents/smoke-test")
async def agent_smoke_test():
    """Run a minimal Claude Agent SDK query to prove the pipeline works."""
    try:
        from atelier.agents.client import run_smoke_test, run_smoke_test_async, _NeedsAsync
    except ImportError:
        return {"success": False, "error": "agents extra not installed (pip install atelier[agents])"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        try:
            return run_smoke_test(cfg)
        except _NeedsAsync:
            return await run_smoke_test_async(cfg)
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@app.get("/api/agents/model-discovery")
def model_discovery():
    """Check for model upgrades via the Anthropic Models API."""
    try:
        from atelier.agents import check_model_upgrade
    except ImportError:
        return {"upgrade_available": False, "reason": "agents extra not installed"}

    from atelier.config import load_config
    cfg = load_config()
    try:
        return check_model_upgrade(cfg)
    except Exception as exc:
        return {"upgrade_available": False, "reason": "error", "error": str(exc)}


# ── CAI Data Platform ──────────────────────────────────────────────


@app.get("/api/data-connections")
def list_data_connections():
    """Return the HOCON-configured CAI Data Platform connection names."""
    try:
        from atelier.config import load_config
        from atelier.data.connections import list_connections
        cfg = load_config()
        return {"connections": list_connections(cfg)}
    except Exception as exc:
        return _error_envelope(f"list_data_connections failed: {exc}")


@app.post("/api/data-connections/{name}/test")
def test_data_connection(name: str):
    """Run ``show databases`` against the named CAI connection via cml.data_v1."""
    try:
        from atelier.config import load_config
        from atelier.data.connections import test_connection
        return test_connection(load_config(), name)
    except Exception as exc:
        return _error_envelope(f"test_data_connection failed: {exc}")


# ── Vocabulary ────────────────────────────────────────────────────


@app.get("/api/vocabulary/stats")
def vocabulary_stats():
    """Return vocabulary term count from cache, hive, or mock."""
    try:
        from atelier.config import load_config
        from atelier.classify.taxonomy import (
            load_annotations_from_json,
            load_annotations_from_hive as _hive_vocab,
            load_universal_vocabulary,
            compose_vocabularies,
        )

        cfg = load_config()
        project_root = Path(__file__).resolve().parent.parent.parent
        cache_path = project_root / "build" / "data" / "annotations" / "annotations.json"

        # Universal base (always available)
        universal = load_universal_vocabulary(hierarchical=True)

        # Try cache first (domain extensions — validated, reject empty)
        if cache_path.exists():
            cs = load_annotations_from_json(cache_path, hierarchical=True)
            if len(cs.categories) > 0:
                composed = compose_vocabularies(universal, cs)
                return {"terms": len(composed.categories), "source": "cache"}

        # Try hive domain extensions
        try:
            cs = _hive_vocab(cfg)
            if len(cs.categories) > 0:
                composed = compose_vocabularies(universal, cs)
                return {"terms": len(composed.categories), "source": "hive"}
        except Exception:
            pass

        # Universal only (no domain extensions)
        return {"terms": len(universal.categories), "source": "universal"}
    except Exception as exc:
        return _error_envelope(f"vocabulary_stats failed: {exc}")


# ── Terminal WebSocket ─────────────────────────────────────────────


@app.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    """Interactive terminal session backed by the Claude Agent SDK.

    The read loop is kept concurrent with any in-flight SDK query by
    routing output through an async ``send`` callback registered on
    the session. ``handle_input`` returns as soon as it schedules the
    SDK query as a background task, so the next ``receive_text()``
    fires immediately and Ctrl-C bytes from the client can cancel
    the task in flight — the same pause/redirect UX as the Claude
    Code CLI.
    """
    await websocket.accept()

    from atelier.terminal import TerminalSession

    session = TerminalSession()

    # Serialize websocket writes across the read loop and the
    # background SDK-query task. ``WebSocket.send_json`` is not
    # reentrant — concurrent sends corrupt the frame stream.
    send_lock = asyncio.Lock()

    async def send(frame: dict) -> None:
        async with send_lock:
            await websocket.send_json(frame)

    session.set_emit(send)

    for frame in session.welcome():
        await send(frame)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if frame.get("type") == "input":
                await session.handle_input(frame.get("data", ""))
    except WebSocketDisconnect:
        pass
    finally:
        await session.shutdown()


# ── Classification FSM ─────────────────────────────────────────────


@app.get("/api/fsm/status")
def fsm_status():
    """Return current classification pipeline FSM state."""
    try:
        from atelier.classify import get_fsm_status
        status = get_fsm_status()
        if status is None:
            return {"state": "IDLE", "progress": {}}
        return status
    except Exception as exc:
        return _error_envelope(f"FSM status failed: {exc}")


@app.post("/api/fsm/start")
def fsm_start():
    """Start a classification pipeline run.

    The pipeline requires an LLM backend.  When ANTHROPIC_API_KEY is
    available (always true with Claude Code), the pipeline auto-defaults
    to AnthropicStructuredBackend + Haiku 4.5.  An explicit classify LLM
    (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL) overrides the default.

    For dev/CI testing without real API calls, inject ``samples=`` and
    ``llm_backend=`` via the Python API.
    """
    import threading
    try:
        from atelier.classify import get_fsm
        from atelier.classify.pipeline import run_classification_pipeline
        from atelier.config import load_config

        cfg = load_config()
        fsm = get_fsm()

        # Check if already running
        current = fsm.get_status()
        if current and current.state.value not in ("IDLE", "CONVERGED", "ERROR"):
            return {"run_id": current.id, "started": False,
                    "error": f"Already running: {current.state.value}"}

        if not cfg.has_classify_llm:
            return {"started": False,
                    "error": "No classification LLM configured. "
                    "Set ANTHROPIC_SUBAGENT_MODEL or ATELIER_LLM_API_KEY."}

        def _background():
            # Pipeline owns run creation via fsm.start_run() — don't
            # create a run here (avoids double-run bug).
            run_classification_pipeline(cfg, fsm)

        t = threading.Thread(target=_background, daemon=True)
        t.start()

        return {"started": True}
    except Exception as exc:
        return _error_envelope(f"FSM start failed: {exc}")


@app.post("/api/fsm/start-bootstrap")
def fsm_start_bootstrap():
    """Deprecated — redirects to /api/fsm/start.

    The convergence loop is now built into the single pipeline entry point.
    """
    return fsm_start()


@app.get("/api/fsm/runs")
def fsm_runs():
    """List past classification pipeline runs."""
    try:
        from atelier.classify import get_fsm
        fsm = get_fsm()
        runs = fsm.list_runs()
        return {"runs": [r.to_dict() for r in runs]}
    except Exception as exc:
        return _error_envelope(f"FSM runs failed: {exc}")


# ── Orchestration WebSocket ────────────────────────────────────────


@app.websocket("/ws/orchestration")
async def orchestration_ws(websocket: WebSocket):
    """Live orchestration events for the XYFlow canvas.

    Placeholder — sends a greeting and keeps the connection alive.
    Future: stream agent_spawned, agent_active, artifact_produced events
    as Claude orchestrates keystone agents via the SDK.
    """
    await websocket.accept()
    await websocket.send_json({
        "type": "topology_reset",
        "message": "Orchestration channel connected.",
    })
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass


# ── Serve React build (production) ───────────────────────────────

_ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"

# Serve ghostty-web WASM assets (production — in dev, Vite serves from public/)
_ghostty_dir = _project_root / "ui" / "public" / "ghostty"
if _ghostty_dir.exists():
    app.mount("/ghostty", StaticFiles(directory=str(_ghostty_dir)), name="ghostty")

if _ui_dist.exists():
    # Mount static assets (JS/CSS bundles)
    _assets_dir = _ui_dist / "assets"
    if _assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(_assets_dir)),
            name="assets",
        )

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        """Serve the React SPA for all non-API routes."""
        if full_path.startswith("api/"):
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=404, content={"detail": "Not found"})
        return FileResponse(str(_ui_dist / "index.html"))
