# Preflight & Bootstrap Hardening

## Summary

Adapted cybersec project's bootstrap/preflight patterns for Atelier, scoped to our stack (PostgreSQL, Qdrant, gRPC, gateway, LLM credentials).

## Changes

### New: Structured preflight (`src/atelier/preflight.py`)
- `CheckResult` and `PreflightResult` dataclasses with deny/warn/pass semantics
- 6 checks: config_exists, required_ports, db_url_parseable, qdrant_host_set (deny), credentials_configured, parquet_dir_exists (warn)
- `run_preflight(cfg, env_path)` returns structured result

### New: Conftest policy (`policy/environment/atelier.rego`)
- Rego deny/warn rules mirroring Python preflight
- 4 deny rules (ports, db, qdrant) + 3 warn rules (credentials, CML)
- `just policy` invokes conftest against `build/config/atelier.json`

### New: JSON materialization (`config.py`)
- `materialize_config_json()` produces `build/config/atelier.json`
- Secrets redacted, derived booleans included
- Called alongside env materialization in resolve-config

### Updated: Wait-with-retry (`bin/start-app.sh`)
- `wait_for_service()` helper replaces bare `sleep N`
- `wait_for_pg()` helper for PostgreSQL readiness
- Applied to PGlite, Qdrant, gRPC startup

### Updated: Readiness probes (`devenv.nix`)
- Qdrant: HTTP probe on `/healthz`
- gRPC server: exec probe via grpcurl
- Gateway now depends on `grpc-server.condition = "process_healthy"`

### Updated: Justfile
- `just preflight` uses structured output with pass/WARN/DENY labels
- `just policy` runs conftest against materialized JSON

### BDD
- 3 new tier-0 scenarios in `features/infra/preflight.feature`
- 34 scenarios total, all passing

## Verification

```
just resolve-config    # produces .env + .json
just preflight         # 4 pass, 2 warn (no creds, no parquet dir)
just policy            # 6 pass, 1 warn via conftest
just bdd               # 34 scenarios, 105 steps, all green
```
