#!/bin/bash
# Atelier application orchestrator.
# Starts gRPC server (background) + HTTP gateway (foreground).
# Used by CAI startup script and local production-like testing.
set -eo pipefail

cleanup() { pkill -P $$ 2>/dev/null || true; }

for sig in INT QUIT HUP TERM; do
  trap "
    cleanup
    trap - $sig EXIT
    kill -s $sig "'"$$"' "$sig"
done
trap cleanup EXIT

PORT=${1:-${CDSW_APP_PORT:-8090}}

# ── Kill stale processes from previous crash loops ───────────────

kill_stale_processes() {
    echo "Cleaning up stale processes..."
    pkill -f "pglite-server.mjs" 2>/dev/null || true
    pkill -f "qdrant/qdrant" 2>/dev/null || true
    pkill -f "atelier.server" 2>/dev/null || true
    pkill -f "atelier.gateway" 2>/dev/null || true
    sleep 2
}
kill_stale_processes

# ── Service readiness helpers ────────────────────────────────────

wait_for_service() {
    local name="$1" check_cmd="$2" timeout="${3:-30}" interval="${4:-2}"
    local deadline=$((SECONDS + timeout)) last_err=""
    echo "Waiting for $name..."
    while [ $SECONDS -lt $deadline ]; do
        if last_err=$(eval "$check_cmd" 2>&1); then
            echo "$name ready"
            return 0
        fi
        sleep "$interval"
    done
    echo "ERROR: $name not healthy after ${timeout}s" >&2
    [ -n "$last_err" ] && echo "  last error: $last_err" >&2
    return 1
}

wait_for_port() {
    local name="$1" host="$2" port="$3" timeout="${4:-30}"
    wait_for_service "$name" \
        "python -c \"import socket; s = socket.create_connection(('$host', $port), timeout=2); s.close()\"" \
        "$timeout"
}

# ── Proof-of-progress readiness probe ──────────────────────────────
# Instead of a fixed 30s timeout, we use a progress-based approach:
# as long as PGlite is producing observable work (new stdout lines,
# data directory file changes), the deadline extends. If nothing
# changes for STALL_TIMEOUT seconds, we declare it stuck and fail.
#
# Progress signals checked each iteration:
#   1. PGlite stdout (piped to .app/pglite.log) — new log lines
#   2. Data directory (.app/pgdata/) — new/modified files
#   3. TCP port reachable — socket server bound
#   4. SQL connectivity — SELECT 1 succeeds (final gate)

wait_for_pglite() {
    local port="$1" stall_timeout="${2:-10}" max_timeout="${3:-120}"
    local log_file=".app/pglite.log"
    local deadline=$((SECONDS + max_timeout))
    local last_progress=$SECONDS
    local last_log_lines=0
    local last_file_count=0

    echo "Waiting for PGlite (proof-of-progress, stall=${stall_timeout}s, max=${max_timeout}s)..."

    while [ $SECONDS -lt $deadline ]; do
        # ── Check progress signals ────────────────────────────
        local progressed=false

        # Signal 1: new log output
        if [ -f "$log_file" ]; then
            local current_lines
            current_lines=$(wc -l < "$log_file" 2>/dev/null || echo 0)
            if [ "$current_lines" -gt "$last_log_lines" ]; then
                # Show new lines for operator visibility
                tail -n $((current_lines - last_log_lines)) "$log_file" | while IFS= read -r line; do
                    echo "  [pglite] $line"
                done
                last_log_lines=$current_lines
                progressed=true
            fi
        fi

        # Signal 2: data directory file changes
        if [ -d ".app/pgdata" ]; then
            local current_files
            current_files=$(find .app/pgdata -type f 2>/dev/null | wc -l)
            if [ "$current_files" -gt "$last_file_count" ]; then
                echo "  [progress] pgdata: $current_files files (was $last_file_count)"
                last_file_count=$current_files
                progressed=true
            fi
        fi

        # Signal 3+4: TCP + SQL (the final gate)
        if python -c "
import socket, sys
try:
    s = socket.create_connection(('127.0.0.1', $port), timeout=2)
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            # Port is open — try SQL connectivity
            if python -c "
from sqlalchemy import create_engine, text
e = create_engine(
    'postgresql+psycopg://postgres:postgres@127.0.0.1:$port/postgres?sslmode=disable&gssencmode=disable',
    connect_args={'connect_timeout': 3},
)
with e.connect() as c:
    c.execute(text('SELECT 1'))
e.dispose()
" 2>/dev/null; then
                echo "PostgreSQL ready (SQL verified)"
                return 0
            else
                echo "  [progress] port open, SQL not yet ready"
                progressed=true
            fi
        fi

        # ── Stall detection ───────────────────────────────────
        if [ "$progressed" = true ]; then
            last_progress=$SECONDS
        elif [ $((SECONDS - last_progress)) -ge "$stall_timeout" ]; then
            echo "ERROR: PGlite stalled — no progress for ${stall_timeout}s" >&2
            [ -f "$log_file" ] && echo "  last log:" >&2 && tail -3 "$log_file" >&2
            return 1
        fi

        sleep 1
    done

    echo "ERROR: PGlite not ready after ${max_timeout}s (hard ceiling)" >&2
    return 1
}

# CAI's reverse proxy expects 127.0.0.1; local dev needs 0.0.0.0
if [ -n "$CDSW_APP_PORT" ]; then
  HOST="127.0.0.1"
else
  HOST="0.0.0.0"
fi

# Ensure pip-installed tools are on PATH
export PATH="$HOME/.local/bin:$PATH"

# ── Decrypt CAI secrets (if encrypted defaults are present) ────────
# .env.cai.enc is committed to git (SOPS+age encrypted).
# Decrypt at startup so values feed into HOCON ${?VAR} substitution.
# Requires SOPS_AGE_KEY or ~/.config/sops/age/keys.txt.
if [ -f .env.cai.enc ] && ! [ -f .env.cai ]; then
  if command -v sops &>/dev/null; then
    echo "Decrypting .env.cai from SOPS..."
    sops --decrypt --input-type dotenv --output-type dotenv .env.cai.enc > .env.cai 2>/dev/null || true
  fi
fi
if [ -f .env.cai ]; then
  echo "Loading CAI defaults from .env.cai..."
  set -a
  source .env.cai
  set +a
fi

# Load nvm so node/npm are available (needed by PGlite below)
if [ -f scripts/load_nvm.sh ]; then
  source scripts/load_nvm.sh
fi

# Activate virtualenv if present (local dev with uv sync)
if [ -f .venv/bin/activate ]; then
  echo "Activating virtualenv..."
  source .venv/bin/activate
fi

# Verify atelier is importable; if not, install into system python
if ! python -c "import atelier" 2>/dev/null; then
  echo "atelier not found, installing into system python..."
  pip3 install -e .
fi

echo "Python: $(which python)"
echo "Packages: $(python -c 'import atelier; print(atelier.__version__)' 2>&1 || echo 'NOT FOUND')"
echo "Claude CLI: $(which claude 2>/dev/null && claude --version 2>/dev/null || echo 'NOT FOUND — Agent SDK will fail')"

# ── Start infrastructure BEFORE config resolution ────────────────
# PGlite and Qdrant must start first so their URLs are in the
# environment when HOCON ${?VAR} substitution runs.

# Start PGlite if no external database configured
# Port 5440 avoids conflict with CAI's platform Postgres on 5432.
if [ -z "$ATELIER_DB_URL" ] && [ -f scripts/pglite-server.mjs ]; then
  PGLITE_PORT=5440
  echo "Starting PGlite on port $PGLITE_PORT..."
  mkdir -p .app/pgdata
  PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=$PGLITE_PORT \
    node scripts/pglite-server.mjs > .app/pglite.log 2>&1 &
  PGLITE_PID=$!

  wait_for_pglite "$PGLITE_PORT" 10 120

  # Verify process is still alive after probe passed
  if ! kill -0 "$PGLITE_PID" 2>/dev/null; then
    echo "ERROR: PGlite process died during startup" >&2
    cat .app/pglite.log >&2
    exit 1
  fi

  # gssencmode=disable: psycopg sends GSSAPI negotiation before auth handshake;
  # PGlite doesn't speak that protocol. Without this, migrations fail with:
  # "received invalid response to GSSAPI negotiation: R"
  export ATELIER_DB_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:${PGLITE_PORT}/postgres?sslmode=disable&gssencmode=disable"
fi

# Start Qdrant if binary is present (CAI deployment)
if [ -x qdrant/qdrant ]; then
  echo "Starting Qdrant on ports 6333/6334..."
  mkdir -p .app/qdrant/storage
  QDRANT__STORAGE__STORAGE_PATH=.app/qdrant/storage \
  QDRANT__SERVICE__HTTP_PORT=6333 \
  QDRANT__SERVICE__GRPC_PORT=6334 \
  qdrant/qdrant &
  wait_for_service "Qdrant" "curl -sf http://localhost:6333/healthz" 30
fi

# ── Resolve config (infra URLs now in environment) ───────────────
python bin/resolve-config.py
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  export "$key=$value"
done < build/config/atelier.env

# ── Preflight validation ──────────────────────────────────────────
echo "Running preflight validation..."
python -c "
import sys
from atelier.config import load_config
from atelier.preflight import run_preflight
result = run_preflight(load_config())
for c in result.checks:
    tag = {'pass': 'PASS', 'warn': 'WARN', 'deny': 'DENY'}[c.status]
    print(f'  [{tag}] {c.name}: {c.message}')
if not result.ok:
    print('Preflight failed — cannot start.', file=sys.stderr)
    sys.exit(1)
"

# Run database migrations + seed keystone agents (shared with devenv)
echo "Running database bootstrap..."
echo "  DB URL: ${ATELIER_DB_URL:-<not set>}"
echo "  psycopg: $(python -c 'import psycopg; print(psycopg.__version__); import importlib; print("binary" if importlib.util.find_spec("psycopg_binary") else "pure-python")' 2>&1)"
python -m atelier.db.bootstrap

# Seed datasets if parquet exists but DB is empty (CAI-specific)
echo "Checking dataset seed..."
python -c "
from atelier.db.dao import AtelierDao
from pathlib import Path
dao = AtelierDao()
existing = dao.list_datasets()
if not existing and Path('data/gittables_sample.parquet').exists():
    dao.upsert_dataset(
        'gittables-sample', 'GitTables CTA Benchmark',
        'data/gittables_sample.parquet',
        '2517 columns from GitTables with 122 DBpedia instance labels',
        2517,
    )
    print('Seeded gittables-sample dataset')
else:
    print(f'Seed check: {len(existing)} datasets already registered')
"

# Start gRPC server (background)
echo "Starting gRPC server on port 50051..."
python -m atelier.server &

wait_for_service "gRPC server" \
    "python -c \"import grpc; ch = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=2)\"" \
    30

# Start HTTP gateway serving React build + REST-to-gRPC bridge
echo "Starting HTTP gateway on $HOST:$PORT..."
python -m uvicorn atelier.gateway:app --host "$HOST" --port "$PORT"
