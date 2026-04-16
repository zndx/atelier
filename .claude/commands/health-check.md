# Health Check

System health and service discovery — verify connectivity and report the operational landscape.

## Instructions

Run a comprehensive health check of the Atelier environment. Each probe is independent — if one fails, continue with the rest. Use Python with `urllib.request` for HTTP calls (no external deps needed).

Execute all probes in a single Python script, then present a compact sitrep.

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

# 8. Governance — Atlas
try:
    sys.path.insert(0, "src/atelier/governance/src")
    from governance.atlas import AtlasClient
    results["atlas_sdk"] = True
except ImportError:
    results["atlas_sdk"] = False

# 9. Governance — Ranger
try:
    from governance.ranger import RangerClient
    results["ranger_sdk"] = True
except ImportError:
    results["ranger_sdk"] = False

# 10. ML Platform (CAI only)
try:
    import cmlapi
    results["cmlapi"] = True
except ImportError:
    results["cmlapi"] = False

# 11. Atelier version
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

Governance
  {✓|✗} Atlas SDK      {available/not found}
  {✓|✗} Ranger SDK     {available/not found}

CAI Platform
  {✓|✗} cmlapi         {available/not available}
  {✓|✗} Data Conns     {list or none configured}

Recommendations:
  • [actionable next steps based on what's available/missing]
```

Adjust recommendations based on findings. For example:
- If Atlas SDK is available but no URL configured: "Configure ATLAS_URL to enable governance sync"
- If pipeline is IDLE with sources available: "Run /classify-columns on {source}"
- If cmlapi available: "Model Registry accessible — pipeline can track CatBoost/SVM versions"
