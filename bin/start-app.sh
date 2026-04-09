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

# CAI's reverse proxy expects 127.0.0.1; local dev needs 0.0.0.0
if [ -n "$CDSW_APP_PORT" ]; then
  HOST="127.0.0.1"
else
  HOST="0.0.0.0"
fi

# Ensure pip-installed tools are on PATH
export PATH="$HOME/.local/bin:$PATH"

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
    node scripts/pglite-server.mjs &
  wait_for_port "PostgreSQL" "127.0.0.1" "$PGLITE_PORT" 30
  export ATELIER_DB_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:${PGLITE_PORT}/postgres?sslmode=disable"
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

# Run database migrations (SQLAlchemy-based, dbmate-compatible)
echo "Running database migrations..."
echo "  DB URL: ${ATELIER_DB_URL:-<not set>}"
echo "  psycopg: $(python -c 'import psycopg; print(psycopg.__version__); import importlib; print("binary" if importlib.util.find_spec("psycopg_binary") else "pure-python")' 2>&1)"
python -c "
import logging
logging.basicConfig(level=logging.INFO)
from atelier.config import load_config
from atelier.db.bootstrap import run_migrations
run_migrations(load_config().db_url)
"

# Seed datasets if parquet exists but DB is empty
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

# Seed keystone agents if DB is empty
echo "Checking agent seed..."
python -c "
from atelier.db.dao import AtelierDao
dao = AtelierDao()
existing = dao.list_agents()
if not existing:
    agents = [
        ('classifier', 'Column Classifier',
         'Zero-shot classification of table columns using LLM reasoning',
         'classifier'),
        ('evidence-fuser', 'Evidence Fuser',
         'Combines LLM, embedding, and SVM evidence via Dempster-Shafer fusion',
         'evidence_fuser'),
        ('viz-director', 'Visualization Director',
         'Curates embedding projections and manages interactive exploration',
         'visualization_director'),
    ]
    for aid, name, desc, role in agents:
        dao.upsert_agent(aid, name, desc, role)
    print(f'Seeded {len(agents)} keystone agents')
else:
    print(f'Seed check: {len(existing)} agents already registered')
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
