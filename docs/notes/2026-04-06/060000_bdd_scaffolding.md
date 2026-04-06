# BDD Scaffolding with Behave

## What Was Built

Complete BDD framework using behave, modeled after the signals project's patterns but adapted for Atelier's CAI deployment modalities.

### Directory Structure

```
features/
├── environment.py              # Tier filtering, stack health, cleanup hooks
├── steps/__init__.py           # Central re-exports (only steps/ dir behave discovers)
├── infra/                      # Domain: infrastructure & services
│   ├── step_defs/              # NOT steps/ — avoids behave auto-discovery
│   │   ├── helpers.py, config_steps.py, health_steps.py
│   ├── config_lifecycle.feature, health_*.feature
├── deployment/                 # Domain: CAI deployment workflows
│   ├── step_defs/
│   │   ├── helpers.py, runtime_steps.py, amp_steps.py
│   ├── runtime_profile.feature, amp_lifecycle.feature
│   ├── application.feature, studio.feature
├── gateway/                    # Domain: HTTP/gRPC (stub)
├── agent/                      # Domain: Claude Agent SDK (stub)
```

### Tier System

| Tier | Tag | Requires | Purpose |
|------|-----|----------|---------|
| 0 | `@tier-0` | Python only | Config, imports, script validation |
| 1 | `@tier-1` | devenv stack | PostgreSQL, Qdrant, gRPC, gateway |
| cai | `@tier-cai` | CAI environment | Documentation-only, always skipped locally |

### CAI Runtime Profile

`features/deployment/runtime_profile.feature` — validates deployment readiness:
- All Python modules importable (including proto stubs)
- Scripts exist and are executable
- HOCON config resolves
- Database migrations parseable

### CAI Deployment Modalities Covered

- **AMP**: `.project-metadata.yaml` validation, create_job/run_job pattern
- **Application**: HOST binding logic (127.0.0.1 vs 0.0.0.0)
- **Studio**: IS_COMPOSABLE root_dir logic (future)
- **Project**: Implicit (base for all others)

## Key Issue Resolved: stdlib Module Shadowing

`features/platform/` directory was shadowing Python's stdlib `platform` module in `sys.modules`. When behave imported `features.platform` (via step re-exports), it registered the package as both `features.platform` AND `platform`. This broke pydantic's lazy loading (`__getattr__` → `platform.system()` → `AttributeError`).

**Fix**: Renamed `features/platform/` → `features/infra/`.

**Rule**: Never name `features/` subdirectories after stdlib modules.

## Results

```
6 features passed, 0 failed, 2 skipped
18 scenarios passed, 0 failed, 6 skipped
60 steps passed, 0 failed, 20 skipped
```
