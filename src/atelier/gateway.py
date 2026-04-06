"""HTTP gateway — serves React frontend and bridges REST to gRPC.

In production (CML), this is the process that binds to CDSW_APP_PORT.
It serves the compiled React build from ui/dist/ and proxies /api/*
requests to the co-located gRPC server.
"""

from pathlib import Path

from fastapi import FastAPI
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
    """Serve a dataset's parquet file for the Embeddings Viewer."""
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


# ── Serve React build (production) ───────────────────────────────

_ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"

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
        return FileResponse(str(_ui_dist / "index.html"))
