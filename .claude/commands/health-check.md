# Health Check

System health and service discovery — verify connectivity and report the operational landscape.

## Strategy

Use **parallel subagents** to maximize discovery speed. Spawn up to 3
agents concurrently, each probing an independent service domain:

1. **Core + Pipeline agent**: Gateway health, PostgreSQL, Qdrant, FSM
   status, recent runs, data sources, vocabulary stats, data connections
2. **ML Platform agent**: cmlapi (model serving, jobs, apps, runtimes)
   and MLflow (tracking URI, experiments, atelier experiment)
3. **Governance + Credentials agent**: Atlas SDK + connectivity, Ranger
   SDK, credential providers, Atelier version
4. **AI Studio agent**: Probe sibling studio applications on the same
   CAI workspace (each is a separate AMP with its own port)

Each agent runs a Python script and returns structured JSON. After all
complete, synthesize the results into a single sitrep.

If subagents are not available (e.g. single-turn mode), fall back to
running all probes sequentially in a single script.

## Instructions

Run the probes below. When using subagents, split into the 3 domains
above. When running sequentially, execute as a single Python script.
Each probe is independent — if one fails, continue with the rest.
Use Python with `urllib.request` for HTTP calls (no external deps needed).

### Probes to run

```python
import json, sys, os
from urllib.request import urlopen, Request
from urllib.error import URLError

results = {}
gateway = "http://localhost:8090"

# 1. Core services (gateway aggregated health)
try:
    r = json.loads(urlopen(f"{gateway}/api/status", timeout=5).read())
    results["grpc"] = r.get("grpc", {}).get("ok", False)
    results["postgres"] = r.get("postgres", {}).get("ok", False)
    results["qdrant"] = r.get("qdrant", {}).get("ok", False)
    results["config"] = r.get("config", {})
except Exception as e:
    results["gateway_error"] = str(e)

# 2. Credentials & model
try:
    from atelier.config import load_config
    cfg = load_config()
    results["has_anthropic"] = cfg.has_anthropic
    results["has_bedrock"] = cfg.has_bedrock
    results["agent_model"] = cfg.agent_model
    results["has_classify_llm"] = cfg.has_classify_llm
except Exception as e:
    results["config_error"] = str(e)

# 3. Data sources
try:
    r = json.loads(urlopen(f"{gateway}/api/data-sources", timeout=5).read())
    results["sources"] = [
        {"id": s["id"], "type": s["source_type"], "vocab_uri": s.get("vocab_uri", "")}
        for s in r.get("sources", [])
    ]
except Exception as e:
    results["sources_error"] = str(e)

# 4. Vocabulary
try:
    r = json.loads(urlopen(f"{gateway}/api/vocabulary/stats", timeout=5).read())
    results["terms"] = r.get("terms", 0)
except Exception:
    pass

# 5. Pipeline state
try:
    r = json.loads(urlopen(f"{gateway}/api/fsm/status", timeout=5).read())
    results["pipeline_state"] = r.get("state", "UNKNOWN")
    results["pipeline_run_id"] = r.get("id")
    p = r.get("progress", {})
    results["last_accuracy"] = p.get("accuracy")
    results["last_coverage"] = p.get("bootstrap_coverage")
except Exception:
    pass

# 6. Recent runs
try:
    r = json.loads(urlopen(f"{gateway}/api/fsm/runs", timeout=5).read())
    results["recent_runs"] = len(r.get("runs", []))
except Exception:
    pass

# 7. Data connections (CAI)
try:
    r = json.loads(urlopen(f"{gateway}/api/data-connections", timeout=5).read())
    results["connections"] = r.get("connections", [])
except Exception:
    pass

# 8. Governance — use the /api/governance/status endpoint which
#    probes Atlas and Ranger connectivity via the governance SDK
try:
    r = json.loads(urlopen(f"{gateway}/api/governance/status", timeout=10).read())
    results["governance"] = r
except Exception as e:
    results["governance"] = {"error": str(e)}

# 10. ML Platform — cmlapi (compute orchestration)
try:
    import cmlapi
    cml = cmlapi.default_client()
    project_id = os.environ.get("CDSW_PROJECT_ID", "")
    results["cmlapi"] = True
    # Probe accessible services
    ml_services = {}
    try:
        models = cml.list_models(project_id=project_id)
        ml_services["model_serving"] = {"available": True, "count": len(models.models or [])}
    except Exception:
        ml_services["model_serving"] = {"available": False}
    try:
        jobs = cml.list_jobs(project_id=project_id)
        ml_services["jobs"] = {"available": True, "count": len(jobs.jobs or [])}
    except Exception:
        ml_services["jobs"] = {"available": False}
    try:
        apps = cml.list_applications(project_id=project_id)
        ml_services["applications"] = {"available": True, "count": len(apps.applications or [])}
    except Exception:
        ml_services["applications"] = {"available": False}
    try:
        runtimes = cml.list_runtimes(search_filter='{"image_identifier":"*"}')
        ml_services["runtimes"] = {"available": True, "count": len(runtimes.runtimes or [])}
    except Exception:
        ml_services["runtimes"] = {"available": False}
    results["ml_services"] = ml_services
except ImportError:
    results["cmlapi"] = False
except Exception as e:
    results["cmlapi"] = True
    results["cmlapi_error"] = str(e)

# 11. ML Platform — MLflow (experiment tracking, auto-configured on CAI)
try:
    import mlflow
    results["mlflow"] = True
    mlflow_info = {}
    try:
        tracking_uri = mlflow.get_tracking_uri()
        mlflow_info["tracking_uri"] = tracking_uri
    except Exception:
        pass
    try:
        experiments = mlflow.search_experiments()
        mlflow_info["experiment_count"] = len(experiments)
        mlflow_info["experiments"] = [
            {"name": e.name, "id": e.experiment_id}
            for e in experiments[:10]
        ]
    except Exception:
        mlflow_info["experiment_count"] = 0
    try:
        # Check if we can create/access an atelier experiment
        exp = mlflow.set_experiment("atelier")
        mlflow_info["atelier_experiment_id"] = exp.experiment_id
    except Exception:
        pass
    results["mlflow_info"] = mlflow_info
except ImportError:
    results["mlflow"] = False

# 12. AI Studio siblings — probe sibling AMPs on same workspace
#     Each studio runs on its own port; try common ports and known
#     health endpoints. Use short timeouts (2s) since unreachable
#     studios should not slow down the sitrep.
studios = {}

# Agent Studio — workflow engine on port 5000, or app port
for port in [5000]:
    try:
        r = urlopen(f"http://localhost:{port}/docs", timeout=2)
        studios["agent_studio"] = {"available": True, "port": port, "endpoint": "/docs"}
        break
    except Exception:
        pass
if "agent_studio" not in studios:
    studios["agent_studio"] = {"available": False}

# RAG Studio — LLM service on port 8081
try:
    r = json.loads(urlopen("http://localhost:8081/llm-service/models/llm", timeout=2).read())
    studios["rag_studio"] = {"available": True, "port": 8081, "models": len(r) if isinstance(r, list) else 0}
except Exception:
    studios["rag_studio"] = {"available": False}

# Fine Tuning Studio — Streamlit + gRPC on 50051
try:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2)
    s.connect(("localhost", 50051))
    s.close()
    studios["fine_tuning_studio"] = {"available": True, "port": 50051, "protocol": "gRPC"}
except Exception:
    studios["fine_tuning_studio"] = {"available": False}

# Synthetic Data Studio — FastAPI on port 8001
try:
    r = json.loads(urlopen("http://localhost:8001/health", timeout=2).read())
    studios["synthetic_data_studio"] = {"available": True, "port": 8001}
except Exception:
    studios["synthetic_data_studio"] = {"available": False}

results["ai_studios"] = studios

# 13. Atelier version
try:
    from atelier import __version__
    results["version"] = __version__
except Exception:
    results["version"] = "unknown"

print(json.dumps(results, indent=2))
```

### Report Format

After running the probes, present a sitrep like this:

```
SITREP — Atelier v{version}
──────────────────────────────────────

Core Services
  {✓|✗} gRPC          {status}
  {✓|✗} PostgreSQL     {status}
  {✓|✗} Qdrant         {status}

Credentials
  {✓|✗} Anthropic      {configured/not configured}
  {✓|✗} Bedrock        {configured/not configured}
  • Agent Model        {model name}
  • Classify LLM       {configured/not configured}

Data Sources           {count} registered
  • {source_id}        {type} (vocab: {vocab_uri})
  ...
  • Terms              {count}

Pipeline               {state}
  • Last accuracy      {N}%
  • Recent runs        {count}

Governance (via /api/governance/status)
  {✓|✗} Atlas          {url, classification_count, entity_type_count or not configured}
  {✓|✗} Ranger         {url, service_count or not configured}
  • Cluster            {cluster_name}
  • Auto-sync          {enabled/disabled}
  • Dry-run            {enabled/disabled}

CAI Compute (cmlapi)
  {✓|✗} cmlapi         {available/not available}
  {✓|✗} Model Serving  {count} endpoints
  {✓|✗} Jobs           {count} registered
  {✓|✗} Applications   {count} running
  {✓|✗} Runtimes       {count} available

ML Tracking (MLflow)
  {✓|✗} MLflow         {available/not available}
  • Tracking URI     {uri or auto-configured}
  • Experiments      {count} ({list names})
  • Atelier exp      {id or not yet created}

AI Studios (sibling AMPs)
  {✓|✗} Agent Studio   {port or not detected}
  {✓|✗} RAG Studio     {port, model count or not detected}
  {✓|✗} Fine Tuning    {port or not detected}
  {✓|✗} Synth Data     {port or not detected}

Data Connections
  {✓|✗} Connections    {list or none configured}

Recommendations:
  • [actionable next steps based on what's available/missing]
```

Adjust recommendations based on findings. For example:
- If Atlas SDK is available but no URL configured: "Configure ATLAS_URL to enable governance sync"
- If pipeline is IDLE with sources available: "Run /classify-columns on {source}"
- If MLflow available: "MLflow tracking active — pipeline runs will be logged as experiments"
- If MLflow available but no atelier experiment: "Will create 'atelier' experiment on first pipeline run"
- If cmlapi model serving available: "Model serving accessible — trained classifiers can be deployed as endpoints"
- If cmlapi jobs available: "Jobs API accessible — pipeline can be scheduled as recurring CML jobs"
- If MLflow not available and not on CAI: "MLflow not detected (local dev) — pipeline metrics logged to build/results/ only"
- If Synth Data Studio available: "Synthetic Data Studio accessible — can generate training data at scale"
- If RAG Studio available: "RAG Studio accessible — classification results can enrich RAG knowledge bases"
- If Fine Tuning Studio available: "Fine Tuning Studio accessible — can fine-tune models on classification data"
- If Agent Studio available: "Agent Studio accessible — workflows can orchestrate multi-step classification pipelines"
