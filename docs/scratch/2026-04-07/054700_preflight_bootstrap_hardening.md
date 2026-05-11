<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Preflight & Bootstrap Hardening

## Summary

Implemented structured preflight validation, conftest/rego policy, readiness probes, and config materialization. Adapted cybersec deny/warn patterns for Atelier's stack.

## What Changed

### New Files
- `src/atelier/preflight.py` — Structured preflight with CheckResult/PreflightResult (deny/warn pattern)
- `bin/resolve-config.py` — Standalone HOCON hydration script, single entry point for config materialization
- `policy/environment/atelier.rego` — Conftest deny/warn rules for CI/CD gating
- `features/infra/preflight.feature` — 3 BDD scenarios for preflight validation
- `features/infra/step_defs/preflight_steps.py` — Step definitions

### Modified Files
- `src/atelier/config.py` — Added `materialize_config_json()`, secret redaction, backward-compat delegation
- `devenv.nix` — Simplified processes (no inline config hydration), readiness probes, dependency ordering
- `bin/start-app.sh` — `wait_for_service()` poll-with-timeout replacing bare `sleep N`
- `justfile` — Updated `preflight` (structured output), added `policy`, `resolve-config` targets
- `features/steps/__init__.py` — Re-export preflight steps
- `features/agent/agent_smoke.feature` — "Keystone agents are seeded" → `@tier-1` (needs DB)

## Key Design Decisions

### devenv-tasks incompatibility
devenv 1.11.2's `devenv-tasks` wrapper prevents multi-line exec blocks from working correctly in process definitions. One-shot processes wrapped in `devenv-tasks` never signal `process_completed_successfully`. Multi-line scripts with `exec` at the end also fail — the wrapper exits after initial commands complete.

**Solution**: Python services call `load_config()` which reads HOCON directly with live env substitution. devenv provides env vars via `dotenv.enable`. No materialized config needed for running services. Simple one-line exec:
```nix
grpc-server.exec = "exec uv run python -m atelier.server";
```

### TCP readiness probe
gRPC server doesn't have reflection enabled, so grpcurl-based health checks fail. Using TCP connect instead:
```nix
exec.command = "bash -c '</dev/tcp/localhost/50051'";
```

### Two audiences for config
- **Running services**: `load_config()` reads HOCON with live env → no materialization needed
- **External tools** (conftest, shell scripts, CI): `just resolve-config` materializes to `build/config/atelier.{env,json}`

## Verification

```
just bdd           # 33 tier-0 scenarios pass
just bdd-full      # 37 pass (pre-existing failures: pgvector, anthropic module)
just resolve-config && just policy   # conftest: 6 pass, 1 warn
curl localhost:8090/api/status       # all services green, has_anthropic: true
```

## Pre-existing Issues Found
- pgvector: devenv declares extension but no `CREATE EXTENSION vector` migration
- `anthropic` Python module not in dependencies (agent SDK smoke test)
- Undefined steps for `bin/start-app.sh` integration test
