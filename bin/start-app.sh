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

# Start gRPC server (background)
echo "Starting gRPC server on port 50051..."
python -m atelier.server &

sleep 3

# Start HTTP gateway serving React build + REST-to-gRPC bridge
echo "Starting HTTP gateway on $HOST:$PORT..."
python -m uvicorn atelier.gateway:app --host "$HOST" --port "$PORT"
