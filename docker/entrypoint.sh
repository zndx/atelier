#!/usr/bin/env bash
# Docker test entrypoint: materialize config → wait for Postgres →
# migrate → start gRPC core → start the gateway (serves ui/dist + REST).
#
# This is a deliberately minimal alternative to bin/start-app.sh — it
# skips the CAI-only machinery (PGlite supervisor, Qdrant binary, SOPS
# secrets, nvm, embedding-warmup offline lock) because in the compose
# stack Postgres and Qdrant are their own containers.
set -euo pipefail
cd /app

echo "[entrypoint] Resolving HOCON config → build/config/atelier.env"
python bin/resolve-config.py

# Export the materialized env so the servicer + gateway see resolved values.
set -a
while IFS='=' read -r key value; do
  [[ -z "$key" || "$key" =~ ^# ]] && continue
  export "$key=$value"
done < build/config/atelier.env
set +a

echo "[entrypoint] Waiting for Postgres at ${ATELIER_DB_URL%%\?*}"
python - <<'PY'
import os, sys, time
from sqlalchemy import create_engine, text
url = os.environ["ATELIER_DB_URL"]
for _ in range(60):
    try:
        eng = create_engine(url, connect_args={"connect_timeout": 3})
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
        eng.dispose()
        print("[entrypoint] Postgres ready")
        break
    except Exception as exc:  # noqa: BLE001
        last = exc
        time.sleep(2)
else:
    print(f"[entrypoint] Postgres never came up: {last}", file=sys.stderr)
    sys.exit(1)
PY

echo "[entrypoint] Running database migrations (atelier.db.bootstrap)"
python -m atelier.db.bootstrap

echo "[entrypoint] Starting gRPC servicer on :50051"
python -m atelier.server &
GRPC_PID=$!

python - <<'PY'
import sys, grpc
ch = grpc.insecure_channel("127.0.0.1:50051")
try:
    grpc.channel_ready_future(ch).result(timeout=30)
    print("[entrypoint] gRPC ready")
except Exception as exc:  # noqa: BLE001
    print(f"[entrypoint] gRPC never became ready: {exc}", file=sys.stderr)
    sys.exit(1)
PY

# If the gRPC servicer dies, take the container down so compose restarts it.
trap 'kill -TERM "$GRPC_PID" 2>/dev/null || true' EXIT

echo "[entrypoint] Starting HTTP gateway on 0.0.0.0:${ATELIER_GATEWAY_PORT:-8090}"
exec python -m uvicorn atelier.gateway:app \
  --host 0.0.0.0 --port "${ATELIER_GATEWAY_PORT:-8090}"
