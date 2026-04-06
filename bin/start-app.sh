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

# CAI's reverse proxy expects 127.0.0.1; local dev needs 0.0.0.0
if [ -n "$CDSW_APP_PORT" ]; then
  HOST="127.0.0.1"
else
  HOST="0.0.0.0"
fi

# Ensure pip-installed tools are on PATH
export PATH="$HOME/.local/bin:$PATH"

# Load nvm so node/npm are available (installed by install_deps.py)
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

# Start PGlite if no external database configured
if [ -z "$ATELIER_DB_URL" ] && [ -f scripts/pglite-server.mjs ]; then
  echo "Starting PGlite on port 5432..."
  mkdir -p .app/pgdata
  PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=5432 \
    node scripts/pglite-server.mjs &
  sleep 3
  export ATELIER_DB_URL="postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres?sslmode=disable"
  echo "PGlite ready: $ATELIER_DB_URL"
fi

# Start Qdrant if binary is present (CAI deployment)
if [ -x qdrant/qdrant ]; then
  echo "Starting Qdrant on ports 6333/6334..."
  mkdir -p .app/qdrant/storage
  QDRANT__STORAGE__STORAGE_PATH=.app/qdrant/storage \
  QDRANT__SERVICE__HTTP_PORT=6333 \
  QDRANT__SERVICE__GRPC_PORT=6334 \
  qdrant/qdrant &
  sleep 2
fi

# Run database migrations
echo "Running database migrations..."
python -c "
from atelier.config import load_config
import subprocess, sys
db_url = load_config().db_url.replace('+psycopg', '')
result = subprocess.run(
    ['dbmate', '--url', db_url, '--migrations-dir', 'db/migrations', '--no-dump-schema', 'up'],
    capture_output=True, text=True
)
if result.returncode == 0:
    print('Migrations applied')
else:
    # dbmate may not be installed (CAI); fall back to direct SQL
    print(f'dbmate not available ({result.stderr.strip()}), skipping migrations')
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

# Start gRPC server (background)
echo "Starting gRPC server on port 50051..."
python -m atelier.server &

sleep 3

# Start HTTP gateway serving React build + REST-to-gRPC bridge
echo "Starting HTTP gateway on $HOST:$PORT..."
python -m uvicorn atelier.gateway:app --host "$HOST" --port "$PORT"
