"""HTTP gateway — serves React frontend and bridges REST to gRPC.

In production (CML), this is the process that binds to CDSW_APP_PORT.
It serves the compiled React build from ui/dist/ and proxies /api/*
requests to the co-located gRPC server.
"""

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
            connect_args={"connect_timeout": 5},
        )
    return _engine


# ── REST → gRPC bridge ────────────────────────────────────────────


# NOTE: endpoints below are declared `def` (not `async def`) so FastAPI runs
# them in its threadpool. The gRPC stub calls are synchronous blocking
# operations; running them on the event loop would serialize every request
# and hang the gateway if any one call stalls.


@app.get("/api/health")
def health():
    client = _get_client()
    resp = client.stub.HealthCheck(
        atelier_pb2.HealthCheckRequest(), timeout=5
    )
    return {"status": resp.status, "version": resp.version}


@app.get("/api/agents")
def list_agents():
    client = _get_client()
    resp = client.stub.ListAgents(
        atelier_pb2.ListAgentsRequest(), timeout=5
    )
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


@app.get("/api/datasets")
def list_datasets():
    client = _get_client()
    resp = client.stub.ListDatasets(
        atelier_pb2.ListDatasetsRequest(), timeout=5
    )
    return {
        "datasets": [
            {
                "id": d.id,
                "name": d.name,
                "description": d.description,
                "parquet_path": d.parquet_path,
                "row_count": d.row_count,
            }
            for d in resp.datasets
        ]
    }


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
        checks["grpc"] = {"ok": False, "error": str(e)}

    # PostgreSQL — reuse cached engine (pool_pre_ping handles stale conns)
    try:
        t0 = time.monotonic()
        from sqlalchemy import text
        engine = _get_status_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        ms = int((time.monotonic() - t0) * 1000)
        checks["postgres"] = {"ok": True, "latency_ms": ms}
    except Exception as e:
        checks["postgres"] = {"ok": False, "error": str(e)}

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
        "agent_model": cfg.agent_model,
        "qdrant_host": cfg.qdrant_host,
        "qdrant_http_port": cfg.qdrant_http_port,
        "db_url_masked": db_masked,
        "model_discovery": model_discovery,
    }

    connected = all(
        checks.get(svc, {}).get("ok", False)
        for svc in ("grpc", "postgres", "qdrant")
    )

    return {**checks, "connected": connected}


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


# ── Terminal WebSocket ─────────────────────────────────────────────


@app.websocket("/ws/terminal")
async def terminal_ws(websocket: WebSocket):
    """Interactive terminal session backed by the Claude Agent SDK."""
    await websocket.accept()

    from atelier.terminal import TerminalSession

    session = TerminalSession()

    for frame in session.welcome():
        await websocket.send_json(frame)

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if frame.get("type") == "input":
                async for out in session.feed(frame.get("data", "")):
                    await websocket.send_json(out)
    except WebSocketDisconnect:
        pass


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
