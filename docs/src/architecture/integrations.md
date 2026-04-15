# Proposed Integrations

This page documents two planned integration points that extend the
data source model: MLflow experiment tracking (Phase 5) and Hive data
connections (Phase 6). Both are designed but not yet implemented.

---

## MLflow Integration (Phase 5)

### Motivation

On CAI deployments, MLflow is available as a managed service. Logging
pipeline runs to MLflow provides:

- **Experiment history**: compare accuracy, conflict, and coverage across
  pipeline versions without the Atelier UI
- **Model registry**: when CatBoost/SVM models are trained, register them
  as versioned artifacts
- **Artifact persistence**: classifications.json, evaluation reports, and
  parquet files survive pod restarts
- **Cross-project visibility**: other CAI workloads can discover Atelier's
  registered models

### Architecture: Write-Then-Reconcile

The MLflow bridge follows the RAG Studio reconciler pattern — the pipeline
never blocks on MLflow I/O.

```
Pipeline thread                    Reconciler (background)
──────────────                     ───────────────────────
write JSON to queue dir ──────►   poll queue dir
  (non-blocking)                   parse JSON envelope
                                   log to MLflow (retries)
                                   move to archive/
```

This design is resilient to:
- MLflow downtime (queue accumulates, reconciler catches up)
- Pipeline latency (no synchronous API calls in the hot path)
- Pod restarts (queue dir is on persistent storage)

### Queue Format

Each pipeline state transition writes a JSON envelope to
`build/mlflow_queue/`:

```json
{
  "event": "run_complete",
  "run_id": "abc123",
  "source_id": "ootb-sample",
  "timestamp": "2026-04-14T12:00:00Z",
  "payload": {
    "params": {
      "source_id": "ootb-sample",
      "vocabulary_mode": "universal",
      "sample_size": 50,
      "llm_model": "glm-4.7",
      "discount_cosine": 0.30
    },
    "metrics": {
      "accuracy": 0.847,
      "micro_f1": 0.832,
      "macro_f1": 0.791,
      "mean_belief": 0.724,
      "mean_conflict": 0.089,
      "coverage": 0.973,
      "llm_calls": 42,
      "bootstrap_iterations": 3
    },
    "artifacts": [
      "build/results/abc123/classifications.json",
      "build/results/abc123/evaluation_report.json",
      "build/results/abc123/atelier_embeddings.parquet"
    ]
  }
}
```

### MLflow Experiment Structure

Each data source maps to an MLflow experiment:

```
Experiment: atelier/ootb-sample
├── Run: v1 (params, metrics, artifacts)
├── Run: v2 (params, metrics, artifacts)
└── Run: v3 (params, metrics, artifacts)

Experiment: atelier/hive-prod-default
└── Run: v1 (params, metrics, artifacts)
```

### What Gets Logged

| Category | Items | Notes |
|----------|-------|-------|
| **Params** | source_id, vocabulary_mode, sample_size, llm_model, discount factors | Static per run |
| **Metrics** | accuracy, micro_f1, macro_f1, mean_belief, mean_conflict, coverage | Numeric scalars |
| **Artifacts** | classifications.json, evaluation_report.json, parquet | Full result set |
| **Models** | CatBoost (.cbm), SVM (.pkl) | Registered when newly trained |

### Module Design

```python
# src/atelier/classify/mlflow_bridge.py

class MLflowBridge:
    """Async write-then-reconcile bridge to MLflow."""

    def __init__(self, queue_dir: Path, experiment_prefix: str = "atelier"):
        self.queue_dir = queue_dir
        self.experiment_prefix = experiment_prefix

    def enqueue(self, event: str, run_id: str, source_id: str, payload: dict):
        """Write an event envelope to the queue (non-blocking)."""
        ...

    def reconcile(self):
        """Process all pending queue items. Called by background thread."""
        ...
```

Pipeline integration points:

```python
# In pipeline.py — at key state transitions:
bridge.enqueue("run_started", run_id, source_id, {"params": {...}})
# ... pipeline work ...
bridge.enqueue("run_complete", run_id, source_id, {"metrics": {...}, "artifacts": [...]})
```

### Gating

MLflow is **only active on CAI** (`cfg.is_cml`). In devenv, the bridge
is a no-op. The `mlflow` package is an optional dependency — import
failure is handled gracefully.

### Configuration

```hocon
# config/base.conf (proposed)
mlflow {
    enabled = false
    enabled = ${?ATELIER_MLFLOW_ENABLED}
    tracking_uri = null
    tracking_uri = ${?MLFLOW_TRACKING_URI}
    queue_dir = "build/mlflow_queue"
}
```

### Implementation Notes

- The reconciler runs as a daemon thread started in the gateway lifespan,
  similar to the sample source seeding
- Queue items are atomic files (write to `.tmp`, rename to `.json`)
  to prevent partial reads
- Failed reconciliation retries with exponential backoff (max 5 min)
- Archive dir (`build/mlflow_queue/archive/`) retains processed items
  for debugging

### Files (Proposed)

| File | Action |
|------|--------|
| `src/atelier/classify/mlflow_bridge.py` | New: bridge + reconciler |
| `src/atelier/classify/pipeline.py` | Extend: bridge calls at transitions |
| `config/base.conf` | Extend: mlflow config block |
| `src/atelier/config.py` | Extend: mlflow fields |
| `src/atelier/gateway.py` | Extend: reconciler daemon thread |

---

## Hive Data Source (Phase 6)

### Motivation

The OOTB sample source demonstrates the pipeline with synthetic data.
In production on CAI, the real value comes from classifying columns in
the customer's actual Hive tables via CAI data connections.

### How It Works

Hive sources are **auto-discovered at gateway startup**. The gateway
lifespan hook calls `discover_hive_sources(cfg)` which:

1. Iterates all connections listed in `ATELIER_DATA_CONNECTIONS`
2. For each connection, runs `SHOW DATABASES` and checks each database
   for an `annotations` table
3. Validates the schema: fetches 1 row and checks for legacy
   (`id`, `ontology`, `annotation`) or universal (`code`, `label`) format
4. Auto-registers valid sources via `get_or_create_data_source()`
   (idempotent — safe to re-run on restart)

Once registered, the pipeline route works automatically:

1. **Pipeline resolves data from the connection**: when `source_id`
   refers to a hive source, the pipeline calls `discover_tables()` and
   `sample_table_metadata()` using that connection
2. **Vocabulary routing**: hive sources use `load_annotations_from_hive()`
   which reads `default.annotations` (domain categories) and composes
   them on top of the universal base
3. **Results register as versions**: each pipeline run creates a new
   version under the hive source, with the same activation/versioning
   semantics as the sample source

### Data Flow

```
CAI Data Connection (Hive/Impala)
        │
        ▼
discover_tables(cfg, connection_name, database)
        │                    ┌─────────────────────────┐
        ▼                    │ load_annotations_from_   │
sample_table_metadata()      │ hive(cfg, connection)    │
        │                    │ → default.annotations    │
        ▼                    └──────────┬──────────────┘
                                        │
    ┌───────────────────────────────────┘
    ▼
compose_vocabularies(universal, hive_domain)
    │
    ▼
run_classification_pipeline(cfg, fsm, source_id="hive-prod-default")
    │
    ▼
Dataset version N+1 registered under hive source
```

### Vocabulary Composition

Hive sources use two-layer vocabulary composition:

```
Layer 1 (always):   Universal vocabulary (16 BFO-grounded PII categories)
                              ╱╲
Layer 2 (hive only): Domain annotations from default.annotations table
                     (290+ customer-specific categories with hierarchical codes)
                              ╱╲
                     Composed CategorySet (300+ terms)
```

Domain categories attach to the universal tree via `parent_code`
references. Categories without a valid parent are logged as warnings
and placed under a catch-all internal node.

### Source Creation

When a user selects a data connection from the Status page dropdown
and clicks "Create Source", the gateway:

1. Validates the connection by running `SHOW DATABASES`
2. Creates a `data_sources` record:
   ```json
   {
     "id": "hive-{connection}-{database}",
     "source_type": "hive",
     "source_uri": "{connection}/{database}",
     "display_name": "hive:{connection}/{database}",
     "vocabulary_mode": "hive"
   }
   ```
3. The source appears in the dropdown immediately

### Pipeline Routing

```python
# In pipeline.py — source-based auto-resolution
if source.source_type == "hive":
    connection_name = source.source_uri.split("/")[0]
    database = source.source_uri.split("/")[1]
    # discover_tables() and sample_table_metadata() use the connection
    # load_annotations_from_hive() uses the connection for vocabulary
```

### Configuration

No new configuration needed. Existing settings control Hive behavior:

```hocon
classify {
    connection_name = ""                # Default CAI data connection
    connection_name = ${?ATELIER_CLASSIFY_CONNECTION}
    database = "default"
    database = ${?ATELIER_CLASSIFY_DATABASE}
}

cml {
    data_connections = ""               # Comma-separated connection names
    data_connections = ${?ATELIER_DATA_CONNECTIONS}
}
```

### Files (Proposed Changes)

| File | Change |
|------|--------|
| `src/atelier/gateway.py` | Add `POST /api/data-sources` endpoint with connection validation |
| `src/atelier/classify/pipeline.py` | Extend source routing to resolve hive connections |
| `ui/src/pages/Status.tsx` | Add "Create Source" button in data connection card |

### Existing Modules Used (No Changes)

| Module | Function | Role |
|--------|----------|------|
| `sampler.py` | `discover_tables()` | List tables via `cml.data_v1` |
| `sampler.py` | `sample_table_metadata()` | Sample column values |
| `taxonomy.py` | `load_annotations_from_hive()` | Load domain vocabulary |
| `taxonomy.py` | `compose_vocabularies()` | Merge universal + domain |

---

## Implementation Priority

| Phase | Integration | Depends On | Testable Without Services |
|-------|-------------|-----------|---------------------------|
| 5 | MLflow bridge | Phase 2 (data model) | Partially — queue/reconcile logic is pure Python |
| 6 | Hive source | Phase 2 (data model) | No — requires CAI data connection |

Phase 5 can be developed and unit-tested independently (the queue
and reconcile logic is pure Python). The MLflow API calls can be
mocked in tier-0 BDD scenarios.

Phase 6 is primarily wiring — the heavy lifting (table discovery,
vocabulary loading, pipeline execution) already exists. The main
new code is the gateway endpoint for source creation and the UI
for triggering it.
