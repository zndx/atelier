# Test Infrastructure

## Framework

Atelier uses [behave](https://behave.readthedocs.io/) for BDD and [pytest](https://pytest.org/) for unit tests. The BDD scenarios live in `features/` and are organized by domain.

## Tier System

Scenarios are tagged by the infrastructure they require. The `ATELIER_BDD_TIER` environment variable controls which tiers run.

| Tier | Tag | Requires | Purpose |
|------|-----|----------|---------|
| 0 | `@tier-0` | Python only | Config, imports, script validation, deployment contracts |
| 1 | `@tier-1` | devenv stack | PostgreSQL, Qdrant, gRPC, full gateway startup |
| cai | `@tier-cai` | CAI session | Live deployment validation — always skipped locally |

Tier 0 runs everywhere: laptops, CI, CAI sessions. No services, no network calls. This is where the [runtime profile](./runtime-profile.md) lives — the scenarios that catch deployment failures before you push.

Tier 1 requires `devenv up` to be running (PostgreSQL on :5533, Qdrant on :6334). These verify that services are healthy and that the application can actually connect to its data stores.

Tier CAI exists as executable documentation. The step definitions are stubs — they express *what should happen* in a live CAI session without automating it. When debugging a deployment failure, these scenarios are a checklist.

## Running Tests

```bash
# Tier-0 only (default, no services needed)
just bdd

# Tier-0 + tier-1 (requires devenv up)
just bdd-full

# Runtime profile specifically
just bdd-runtime

# Single domain
ATELIER_BDD_TIER=0 uv run behave features/deployment/

# Single feature file
uv run behave features/deployment/runtime_profile.feature

# By tag
ATELIER_BDD_TIER=0 uv run behave features/ -t @amp

# Verbose (show all steps, not just failures)
just bdd --no-capture
```

## Feature Organization

```
features/
├── environment.py              # Tier filtering, stack health, cleanup hooks
├── steps/__init__.py           # Central re-exports (behave's discovery point)
├── infra/                      # Domain: infrastructure & services
│   ├── step_defs/              # Step definitions (NOT steps/)
│   │   ├── helpers.py
│   │   ├── config_steps.py
│   │   └── health_steps.py
│   ├── config_lifecycle.feature
│   ├── health_postgres.feature
│   ├── health_qdrant.feature
│   └── health_pglite.feature
├── deployment/                 # Domain: CAI deployment workflows
│   ├── step_defs/
│   │   ├── helpers.py
│   │   ├── runtime_steps.py
│   │   └── amp_steps.py
│   ├── runtime_profile.feature
│   ├── amp_lifecycle.feature
│   ├── application.feature
│   └── studio.feature
├── gateway/                    # Domain: HTTP/gRPC (stub)
└── agent/                      # Domain: Claude Agent SDK (stub)
```

### Step Discovery

Behave only discovers step definitions from `features/steps/`. Domain step definitions live in `<domain>/step_defs/` directories and are re-exported through `features/steps/__init__.py`:

```python
from features.infra.step_defs.config_steps import *
from features.infra.step_defs.health_steps import *
from features.deployment.step_defs.runtime_steps import *
from features.deployment.step_defs.amp_steps import *
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
