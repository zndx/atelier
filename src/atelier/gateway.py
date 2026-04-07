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


def _get_client() -> AtelierClient:
    global _client
    if _client is None:
        _client = AtelierClient()
    return _client


# ── REST → gRPC bridge ────────────────────────────────────────────


@app.get("/api/health")
async def health():
    client = _get_client()
    resp = client.stub.HealthCheck(atelier_pb2.HealthCheckRequest())
    return {"status": resp.status, "version": resp.version}


@app.get("/api/agents")
async def list_agents():
    client = _get_client()
    resp = client.stub.ListAgents(atelier_pb2.ListAgentsRequest())
    return {
        "agents": [
            {
                "id": a.id,
                "name": a.name,
                "description": a.description,
                "role": a.role,
            }
            for a in resp.agents
        ]
    }


@app.get("/api/datasets")
async def list_datasets():
    client = _get_client()
    resp = client.stub.ListDatasets(atelier_pb2.ListDatasetsRequest())
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
async def get_dataset_data(dataset_id: str):
    """Serve a dataset's parquet file for the Embeddings page."""
    from fastapi.responses import Response

    client = _get_client()
    resp = client.stub.ListDatasets(atelier_pb2.ListDatasetsRequest())
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
async def status():
    """Aggregated infrastructure health for the operator dashboard."""
    import time
    import urllib.request

    from atelier.config import load_config

    cfg = load_config()
    checks: dict = {}

    # gRPC server
    try:
        t0 = time.monotonic()
        client = _get_client()
        resp = client.stub.HealthCheck(atelier_pb2.HealthCheckRequest())
        ms = int((time.monotonic() - t0) * 1000)
        checks["grpc"] = {"ok": True, "version": resp.version, "latency_ms": ms}
    except Exception as e:
        checks["grpc"] = {"ok": False, "error": str(e)}

    # PostgreSQL
    try:
        t0 = time.monotonic()
        from sqlalchemy import create_engine, text
        engine = create_engine(cfg.db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        ms = int((time.monotonic() - t0) * 1000)
        checks["postgres"] = {"ok": True, "latency_ms": ms}
    except Exception as e:
        checks["postgres"] = {"ok": False, "error": str(e)}

    # Qdrant
    try:
        t0 = time.monotonic()
        url = f"http://{cfg.qdrant_host}:{cfg.qdrant_http_port}/healthz"
        urllib.request.urlopen(url, timeout=5)
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
async def validate_credentials():
    """Validate all configured LLM provider credentials."""
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
async def model_discovery():
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
