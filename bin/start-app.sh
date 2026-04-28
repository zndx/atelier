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

# ── Decrypt SOPS-encrypted artifacts ──────────────────────────────
# bin/bootstrap-secrets.sh is the shared entry point (CAI here,
# devenv enterShell locally, `just bootstrap-secrets` on demand).
# It handles .env.cai.enc → .env.cai AND
# features/fixtures/curated_reference.csv.enc → build/data/curated_reference.csv.
# Idempotent + no-op on missing ciphertext, so a clean checkout
# without encrypted defaults still boots.
if [ -f bin/bootstrap-secrets.sh ]; then
  bash bin/bootstrap-secrets.sh || true
fi
if [ -f .env.cai ]; then
  echo "Loading CAI defaults from .env.cai..."
  set -a
  source .env.cai
  set +a
  # Post-source audit: print which high-signal defaults actually
  # reached the environment.  Helps operators tell "the encrypted
  # defaults loaded cleanly" apart from "bootstrap-secrets ran but
  # sops / age-key failed and the file is empty".
  _report() {
    local name="$1" val="${!1:-}"
    if [ -n "$val" ]; then
      # ARNs and URLs get truncated so the log stays readable;
      # short values print whole.
      local disp="$val"
      [ "${#disp}" -gt 60 ] && disp="${disp:0:57}..."
      echo "  ✓ ${name}=${disp}"
    else
      echo "  ✗ ${name} not set"
    fi
  }
  _report ATELIER_AGENT_MODEL
  _report ATELIER_CLASSIFY_MODEL
  _report ATELIER_DATA_CONNECTIONS
  _report ATELIER_CLASSIFY_CONNECTION
  _report ATELIER_CLASSIFY_AUTO_START
  _report ATELIER_OVERWATCH_MODEL
  _report AWS_REGION
  unset -f _report
else
  echo "WARNING: .env.cai not present after bootstrap-secrets — encrypted"
  echo "WARNING: deployment defaults did NOT decrypt.  You'll need every"
  echo "WARNING: ATELIER_* value set via the AMP env form.  Check the"
  echo "WARNING: install-deps job output for scripts/install_sops.sh and"
  echo "WARNING: confirm SOPS_AGE_KEY reached the application environment."
fi
# Point the pipeline at the materialized curated-reference CSV unless
# the operator overrode ATELIER_CLASSIFY_REFERENCE_URI directly (env-var
# wins).
if [ -z "${ATELIER_CLASSIFY_REFERENCE_URI:-}" ] && [ -f build/data/curated_reference.csv ]; then
  export ATELIER_CLASSIFY_REFERENCE_URI=build/data/curated_reference.csv
  echo "curated reference: $ATELIER_CLASSIFY_REFERENCE_URI"
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
  # Bump V8's old-space heap.  PGlite + pgvector hold result sets and
  # index pages in the Node heap; the ~4 GB default cap is tight for
  # runs that touch thousands of columns and the WASM allocator
  # aborts (kernel/cgroup OOM-kill, no Node-level error trace) when
  # it can't satisfy a request.  CAI hosts have plenty of headroom —
  # override via PGLITE_NODE_MAX_OLD_SPACE_MB if needed.
  NODE_MAX_OLD_SPACE_MB="${PGLITE_NODE_MAX_OLD_SPACE_MB:-8192}"
  PGLITE_DATA_DIR=.app/pgdata PGLITE_PORT=$PGLITE_PORT \
    node --max-old-space-size="$NODE_MAX_OLD_SPACE_MB" \
      scripts/pglite-server.mjs > .app/pglite.log 2>&1 &
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

# ── Embedding model warmup + offline-mode lock ────────────────────
# Load the SentenceTransformer once and probe-encode it before
# starting the server.  On a healthy CAI deploy the model is already
# in the HF cache (pre-downloaded by scripts/install_deps.py) — this
# is a fast confirmation.  On a broken deploy it fails LOUDLY with a
# clear remediation path, rather than letting the pipeline stall on
# the first cosine-evidence call hours into a run.
#
# After warmup confirms the cache is populated, we export
# HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 so that any subsequent
# load attempt CANNOT silently re-download — a missing-cache failure
# becomes loud instead of silent.
echo "Warming up embedding model..."
if python -c "from atelier.classify.embedding import warmup; warmup()"; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    echo "Embedding model ready; HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1 locked in"
else
    echo "Embedding model warmup FAILED — refusing to start gateway." >&2
    echo "Remediation:" >&2
    echo "  - On CAI: re-run the install job to populate the HF cache" >&2
    echo "    (scripts/install_deps.py downloads all-MiniLM-L6-v2)." >&2
    echo "  - Locally: ensure HF_HUB_OFFLINE is unset on first run so" >&2
    echo "    the model can download." >&2
    exit 1
fi

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
GRPC_PID=$!

wait_for_service "gRPC server" \
    "python -c \"import grpc; ch = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(ch).result(timeout=2)\"" \
    30

# Start HTTP gateway serving React build + REST-to-gRPC bridge.
# Backgrounded (was foreground) so the shell can monitor every
# critical child and exit when any one dies — see the wait-n loop
# below.
echo "Starting HTTP gateway on $HOST:$PORT..."
python -m uvicorn atelier.gateway:app --host "$HOST" --port "$PORT" &
UVICORN_PID=$!

# ── Self-healing on critical-child death ─────────────────────────
# If PGlite, the gRPC server, or uvicorn dies, we exit so
# scripts/startup_app.py restarts the whole stack with backoff.
# Without this, a PGlite OOM-kill leaves uvicorn alive serving 500s
# against a dead database until the AMP is manually restarted —
# pool_pre_ping in dao.py will reconnect to a fresh PGlite, so a
# clean restart is enough to recover.
#
# `wait -n` returns when any backgrounded child exits.  We protect
# against `set -e` by routing the exit code through `||`.
wait -n || exit_code=$?
exit_code="${exit_code:-0}"

echo "" >&2
echo "Critical child exited (code $exit_code) — taking down the stack so scripts/startup_app.py can restart." >&2
for spec in "PGlite:${PGLITE_PID:-}" "gRPC server:${GRPC_PID:-}" "uvicorn gateway:${UVICORN_PID:-}"; do
    name="${spec%%:*}"; pid="${spec##*:}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
        echo "  → $name (PID $pid) is gone" >&2
    fi
done
if [ -f .app/pglite.log ]; then
    echo "  → tail of .app/pglite.log:" >&2
    tail -n 10 .app/pglite.log 2>&1 | sed 's/^/      /' >&2
fi
exit "$exit_code"
