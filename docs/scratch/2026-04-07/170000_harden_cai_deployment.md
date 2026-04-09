# Harden CAI Deployment for Clean CI Re-deployment

## Problem
AMP deployment on CAI was stuck in crash loops. Each failure left orphaned
processes (PGlite, Qdrant, gRPC) holding ports. The restart loop had no
cooldown, `set -a` polluted the environment causing segfaults, and preflight
validation never ran on CAI.

## Changes

### `bin/start-app.sh` (4 changes)

1. **`kill_stale_processes()` at entry** — Kills PGlite, Qdrant, gRPC, and
   gateway by command pattern before any infrastructure starts. 2s sleep for
   socket TIME_WAIT. Removed the redundant standalone `pkill` from the PGlite
   block.

2. **Export `ATELIER_DB_URL` only after health check** — Uses a local variable
   for the PGlite URL during health check; only exports after `wait_for_pg`
   succeeds. If PGlite fails, `set -e` exits before the URL enters the
   environment.

3. **Replace `set -a && source` with explicit var loading** — `set -a`
   auto-exported every assignment for the rest of the shell, polluting the
   subprocess environment (likely cause of the segfault in psycopg/libpq).
   Replaced with a `while IFS='=' read` loop that only exports key=value pairs.

4. **Preflight gate before migrations** — Runs `run_preflight(load_config())`
   between config resolution and database migrations. Prints each check result
   and exits with code 1 on deny, allowing the backoff loop to handle it
   gracefully.

### `scripts/startup_app.py` (1 change)

5. **Exponential backoff** — Initial delay 5s, doubles each failure, caps at
   60s. Resets to 5s after 30+ seconds of healthy uptime (means the full stack
   started successfully). Prints uptime and next delay for operator
   observability.

## Verification

- `bash -n bin/start-app.sh` — passes
- `python -c "import ast; ast.parse(...)"` — passes
- `just bdd` — 33 scenarios passed, 0 failed
