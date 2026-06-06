# Test Infrastructure

## Framework

Atelier uses [behave](https://behave.readthedocs.io/) for BDD and [pytest](https://pytest.org/) for unit tests. The BDD scenarios live in `features/` and are organized by domain.

## Tier System

Scenarios are tagged by the infrastructure they require. The `ATELIER_BDD_TIER` environment variable controls which tiers run.

| Tier | Tag | Requires | Purpose |
|------|-----|----------|---------|
| 0 | `@tier-0` | Python only | Config, imports, classification pipeline, agent loop, ML classifiers |
| 1 | `@tier-1` | devenv stack | PostgreSQL, Qdrant, gRPC, full gateway startup |
| cai | `@tier-cai` | CAI session | Live deployment validation — always skipped locally |

Additional tags:
- `@slow` — scenarios requiring extended runtime (pipeline E2E, ML training)
- `@gpu` — GPU acceleration scenarios (run on CPU too, just slower)

Tier 0 runs everywhere: laptops, CI, CAI sessions. No services, no network calls. This is where the [runtime profile](./runtime-profile.md) lives — the scenarios that catch deployment failures before you push.

Tier 1 requires `devenv up` to be running (PostgreSQL on :5533, Qdrant on :6334). These verify that services are healthy and that the application can actually connect to its data stores.

Tier CAI exists as executable documentation. The step definitions are stubs — they express *what should happen* in a live CAI session without automating it. When debugging a deployment failure, these scenarios are a checklist.

## Running Tests

```bash
# Full BDD suite including gateway checks (preferred)
just behave

# Tier-0 only (no services needed)
just bdd

# Tier-0 + tier-1 (requires devenv up)
just bdd-full

# Runtime profile specifically
just bdd-runtime

# Single domain
ATELIER_BDD_TIER=0 uv run behave features/agent/

# Single feature file
uv run behave features/agent/classification.feature

# By tag
ATELIER_BDD_TIER=0 uv run behave features/ -t @bootstrap

# Verbose (show all steps, not just failures)
just behave --no-capture
```

## Feature Organization

```
features/
├── environment.py                          # Tier filtering, stack health, cleanup hooks
├── steps/__init__.py                       # Central re-exports (behave's discovery point)
├── infra/                                  # Domain: infrastructure & services
│   ├── step_defs/
│   │   ├── helpers.py
│   │   ├── config_steps.py
│   │   ├── health_steps.py
│   │   └── preflight_steps.py
│   ├── config_lifecycle.feature            # 3 scenarios
│   ├── health_postgres.feature             # 2 scenarios
│   ├── health_qdrant.feature               # 1 scenario
│   ├── health_pglite.feature               # 2 scenarios
│   └── preflight.feature                   # 3 scenarios
├── deployment/                             # Domain: CAI deployment workflows
│   ├── step_defs/
│   │   ├── helpers.py
│   │   ├── runtime_steps.py
│   │   ├── amp_steps.py
│   │   └── naming_steps.py
│   ├── runtime_profile.feature             # 6 scenarios
│   ├── amp_lifecycle.feature               # 5 scenarios
│   ├── application.feature                 # 3 scenarios
│   ├── studio.feature                      # 2 scenarios
│   ├── embeddings.feature                  # 4 scenarios
│   └── naming_audit.feature                # 2 scenarios
├── gateway/                                # Domain: HTTP/gRPC gateway
│   ├── step_defs/
│   │   ├── status_steps.py
│   │   ├── http_steps.py
│   │   ├── endpoint_steps.py
│   │   ├── pipeline_steps.py
│   │   └── testclient_steps.py
│   ├── api_endpoints.feature               # 8 scenarios
│   ├── api_testclient.feature              # 7 scenarios
│   ├── status_endpoint.feature             # 4 scenarios
│   ├── pipeline_integration.feature        # 2 scenarios
│   └── spa_routes.feature                  # placeholder
└── agent/                                  # Domain: classification & agents
    ├── step_defs/
    │   ├── agent_steps.py
    │   ├── classification_steps.py
    │   ├── bootstrap_steps.py
    │   ├── backend_steps.py
    │   ├── synth_steps.py
    │   ├── ml_steps.py
    │   ├── ml_e2e_steps.py
    │   ├── sage_steps.py
    │   ├── shap_steps.py
    │   ├── real_data_steps.py
    │   ├── belief_path_steps.py
    │   ├── synth_framework_steps.py
    │   ├── meta_tagging_steps.py
    │   ├── experimentation_steps.py
    │   ├── agent_loop_steps.py
    │   └── monte_carlo_steps.py
    ├── classification.feature              # 19 scenarios (DST, pipeline, MC sampling)
    ├── bootstrap.feature                   # 10 scenarios
    ├── agent_loop.feature                  # 6 scenarios
    ├── agent_smoke.feature                 # 6 scenarios
    ├── backend.feature                     # 8 scenarios
    ├── ml_classifiers.feature              # 4 scenarios
    ├── ml_e2e.feature                      # 2 scenarios
    ├── synth.feature                       # 2 scenarios
    ├── synth_framework.feature             # 2 scenarios
    ├── sage.feature                        # 1 scenario
    ├── shap.feature                        # 2 scenarios
    ├── belief_path.feature                 # 3 scenarios
    ├── meta_tagging.feature                # 2 scenarios
    ├── experimentation.feature             # 3 scenarios
    └── real_data.feature                   # 3 scenarios
```

### Step Discovery

Behave only discovers step definitions from `features/steps/`. Domain step definitions live in `<domain>/step_defs/` directories and are re-exported through `features/steps/__init__.py`:

```python
from features.infra.step_defs.config_steps import *
from features.infra.step_defs.health_steps import *
from features.infra.step_defs.preflight_steps import *
from features.deployment.step_defs.runtime_steps import *
from features.deployment.step_defs.amp_steps import *
from features.deployment.step_defs.naming_steps import *
from features.agent.step_defs.agent_steps import *
from features.agent.step_defs.classification_steps import *
from features.agent.step_defs.bootstrap_steps import *
from features.agent.step_defs.backend_steps import *
from features.agent.step_defs.synth_steps import *
from features.agent.step_defs.ml_steps import *
from features.agent.step_defs.ml_e2e_steps import *
from features.agent.step_defs.sage_steps import *
from features.agent.step_defs.shap_steps import *
from features.agent.step_defs.real_data_steps import *
from features.agent.step_defs.belief_path_steps import *
from features.agent.step_defs.synth_framework_steps import *
from features.agent.step_defs.meta_tagging_steps import *
from features.agent.step_defs.experimentation_steps import *
from features.gateway.step_defs.status_steps import *
from features.gateway.step_defs.http_steps import *
from features.gateway.step_defs.endpoint_steps import *
from features.gateway.step_defs.pipeline_steps import *
from features.agent.step_defs.agent_loop_steps import *
from features.agent.step_defs.monte_carlo_steps import *
from features.gateway.step_defs.testclient_steps import *
```

Two conventions protect against behave's automatic discovery behavior:

1. **Use `step_defs/`, not `steps/`** — Behave walks the feature tree and exec's any `.py` file it finds in a directory named `steps/`. This bypasses Python's import system, breaking relative imports and module context. Using `step_defs/` avoids this entirely.

2. **Never name a `features/` subdirectory after a stdlib module** — When behave imports `features.platform`, Python also registers it as `platform` in `sys.modules`, shadowing the stdlib. This breaks anything that lazily imports `platform` (including pydantic). The `infra/` domain was originally named `platform/` until this caused a cascade of subtle failures.

### Config-Driven BDD

Infrastructure steps load configuration from HOCON via `atelier.config.load_config()` rather than hardcoding values. This means BDD scenarios validate the same config path used in production:

```python
from atelier.config import load_config
cfg = load_config()
_wait_for("PostgreSQL", lambda: _check_pg(cfg.db_url))
```

### Stack Health Gate

Tier-1 scenarios share a one-time stack health check in `environment.py`. Before the first tier-1 scenario runs, the framework verifies PostgreSQL and Qdrant are reachable (with a 60-second retry window). If either service is down, all tier-1 scenarios fail fast with a clear message rather than producing confusing connection errors.

### Cleanup

`after_scenario` in `environment.py` removes temporary files registered via `context._temp_files`. This handles config materialization artifacts and other test-created files.

## Unit Tests

Alongside BDD, `tests/` contains pytest unit tests for isolated module behavior:

```bash
just test                    # Run all pytest tests
uv run pytest tests/ -x     # Stop on first failure
```

BDD and pytest serve complementary roles: pytest validates that individual functions behave correctly; BDD validates that the system's deployment contracts hold.
