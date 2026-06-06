#!/bin/bash
# PGlite supervisor — keeps scripts/pglite-server.mjs running.
#
# The CAI Application supervisor (scripts/startup_app.py) only watches the
# foreground gateway, so a sibling-process pglite that dies mid-session
# leaves the gateway 500-ing on every DB call until the AMP is restarted.
# This wrapper closes that gap: it respawns pglite with exponential
# backoff and a circuit-breaker after repeated fast failures, and
# publishes machine-readable state to .app/pglite-supervisor.state so
# the gateway can surface health to the UI.
#
# Inputs (env):
#   PGLITE_PORT              — TCP port for pglite (default 5440)
#   PGLITE_DATA_DIR          — pgdata path (default .app/pgdata)
#   PGLITE_NODE_MAX_OLD_SPACE_MB — V8 heap cap (default 8192)
#   PGLITE_MAX_CONNECTIONS   — accept-queue size (passed through to pglite)
#   PGLITE_SUPERVISOR_MAX_BACKOFF — cap on backoff seconds (default 60)
#   PGLITE_SUPERVISOR_BREAK_THRESHOLD — consecutive fast failures before
#                                       circuit opens (default 5)
#
# Files:
#   .app/pglite.log                    — child stdout/stderr
#   .app/pglite-supervisor.state       — JSON health snapshot
#   .app/pglite-supervisor.restart     — touch this to request a restart
#                                        (gateway POST /api/pglite/restart)

set -uo pipefail

PGLITE_PORT="${PGLITE_PORT:-5440}"
PGLITE_DATA_DIR="${PGLITE_DATA_DIR:-.app/pgdata}"
NODE_MAX_OLD_SPACE_MB="${PGLITE_NODE_MAX_OLD_SPACE_MB:-8192}"
MAX_BACKOFF="${PGLITE_SUPERVISOR_MAX_BACKOFF:-60}"
BREAK_THRESHOLD="${PGLITE_SUPERVISOR_BREAK_THRESHOLD:-5}"
HEALTHY_UPTIME_SECONDS="${PGLITE_SUPERVISOR_HEALTHY_UPTIME:-30}"

LOG_FILE=".app/pglite.log"
STATE_FILE=".app/pglite-supervisor.state"
RESTART_SENTINEL=".app/pglite-supervisor.restart"

mkdir -p "$(dirname "$STATE_FILE")"

write_state() {
    # Atomic state-file update.  Writes via tmp+mv so the gateway
    # never reads a half-written file.  Values passed to python via
    # argv (not heredoc interpolation) so the bash sentinel "null"
    # maps cleanly onto Python ``None`` — the previous version of
    # this function interpolated literal "null" into python source
    # where it's an undefined name.
    local state="$1" pid="${2:-null}" exit_code="${3:-null}" \
          started_at="${4:-null}" backoff_until="${5:-null}" \
          consecutive_failures="${6:-0}" restart_count="${7:-0}" \
          circuit_broken="${8:-false}"

    python - "$state" "$pid" "$exit_code" "$started_at" \
              "$backoff_until" "$consecutive_failures" \
              "$restart_count" "$circuit_broken" \
              "$$" "$PGLITE_PORT" <<'PY' > "${STATE_FILE}.tmp"
import json, sys, time
def _num(s):
    if s in ("null", "", "None"):
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None
def _bool(s):
    return s.lower() == "true"
(state, pid, exit_code, started_at, backoff_until,
 consecutive_failures, restart_count, circuit_broken,
 supervisor_pid, port) = sys.argv[1:11]
doc = {
    "state": state,
    "pid": _num(pid),
    "supervisor_pid": _num(supervisor_pid),
    "started_at": _num(started_at),
    "last_exit_code": _num(exit_code),
    "consecutive_failures": _num(consecutive_failures) or 0,
    "restart_count": _num(restart_count) or 0,
    "backoff_until": _num(backoff_until),
    "circuit_broken": _bool(circuit_broken),
    "updated_at": time.time(),
    "port": _num(port),
}
print(json.dumps(doc, indent=2))
PY
    mv "${STATE_FILE}.tmp" "$STATE_FILE"
}

iso_or_null() {
    # Echo a JSON-safe numeric epoch or null (we use epoch seconds —
    # the UI converts to local time).
    if [ -z "${1:-}" ]; then echo null; else echo "$1"; fi
}

# Trap so a Ctrl-C / SIGTERM to the supervisor doesn't leave a zombie
# pglite child behind.
CHILD_PID=""
shutdown() {
    if [ -n "$CHILD_PID" ] && kill -0 "$CHILD_PID" 2>/dev/null; then
        echo "[supervisor] shutting down child pid=$CHILD_PID" >&2
        kill -TERM "$CHILD_PID" 2>/dev/null || true
        # Give it a moment, then SIGKILL if still alive
        for _ in 1 2 3 4 5; do
            kill -0 "$CHILD_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -KILL "$CHILD_PID" 2>/dev/null || true
    fi
    write_state "stopped" null null null null 0 "$RESTART_COUNT" false
    exit 0
}
trap shutdown INT TERM

RESTART_COUNT=0
CONSECUTIVE_FAILURES=0
BACKOFF=1

while true; do
    # ── Circuit-breaker gate ───────────────────────────────────
    if [ "$CONSECUTIVE_FAILURES" -ge "$BREAK_THRESHOLD" ]; then
        echo "[supervisor] circuit broken after $CONSECUTIVE_FAILURES consecutive fast failures — waiting for restart sentinel" >&2
        write_state "circuit_broken" null null null null \
            "$CONSECUTIVE_FAILURES" "$RESTART_COUNT" true
        # Wait for operator-requested restart (touch sentinel).  Poll
        # every 2s; this is a degraded state — rare in practice.
        local_ref="$(stat -c %Y "$RESTART_SENTINEL" 2>/dev/null || echo 0)"
        while true; do
            current_ref="$(stat -c %Y "$RESTART_SENTINEL" 2>/dev/null || echo 0)"
            if [ "$current_ref" -gt "$local_ref" ]; then
                echo "[supervisor] restart sentinel touched — clearing circuit" >&2
                CONSECUTIVE_FAILURES=0
                BACKOFF=1
                break
            fi
            sleep 2
        done
    fi

    # ── Launch child ──────────────────────────────────────────
    START_TS=$(date +%s)
    write_state "starting" null null "$START_TS" null \
        "$CONSECUTIVE_FAILURES" "$RESTART_COUNT" false

    echo "[supervisor] launching pglite (attempt $((RESTART_COUNT + 1)))" | tee -a "$LOG_FILE"
    PGLITE_DATA_DIR="$PGLITE_DATA_DIR" \
    PGLITE_PORT="$PGLITE_PORT" \
    PGLITE_MAX_CONNECTIONS="${PGLITE_MAX_CONNECTIONS:-32}" \
        node --max-old-space-size="$NODE_MAX_OLD_SPACE_MB" \
            scripts/pglite-server.mjs >> "$LOG_FILE" 2>&1 &
    CHILD_PID=$!

    write_state "running" "$CHILD_PID" null "$START_TS" null \
        "$CONSECUTIVE_FAILURES" "$RESTART_COUNT" false

    # ── Watch loop: wait for child OR restart sentinel ────────
    SENTINEL_TS_AT_START="$(stat -c %Y "$RESTART_SENTINEL" 2>/dev/null || echo 0)"
    while kill -0 "$CHILD_PID" 2>/dev/null; do
        # Operator-requested restart: SIGTERM the child to break out.
        current_sentinel="$(stat -c %Y "$RESTART_SENTINEL" 2>/dev/null || echo 0)"
        if [ "$current_sentinel" -gt "$SENTINEL_TS_AT_START" ]; then
            echo "[supervisor] restart sentinel touched — terminating child pid=$CHILD_PID" >&2
            kill -TERM "$CHILD_PID" 2>/dev/null || true
            # Treat operator restart as healthy so backoff doesn't kick in
            CONSECUTIVE_FAILURES=0
            BACKOFF=1
            break
        fi
        sleep 2
    done
    wait "$CHILD_PID" 2>/dev/null
    EXIT_CODE=$?
    EXIT_TS=$(date +%s)
    UPTIME=$((EXIT_TS - START_TS))
    RESTART_COUNT=$((RESTART_COUNT + 1))

    echo "[supervisor] child exited code=$EXIT_CODE uptime=${UPTIME}s" | tee -a "$LOG_FILE"

    # ── Healthy-uptime resets backoff ────────────────────────
    if [ "$UPTIME" -ge "$HEALTHY_UPTIME_SECONDS" ]; then
        CONSECUTIVE_FAILURES=0
        BACKOFF=1
    else
        CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        BACKOFF=$((BACKOFF * 2))
        [ "$BACKOFF" -gt "$MAX_BACKOFF" ] && BACKOFF=$MAX_BACKOFF
    fi

    # ── Backoff state visible to UI ──────────────────────────
    NEXT_TS=$((EXIT_TS + BACKOFF))
    write_state "backoff" null "$EXIT_CODE" null "$NEXT_TS" \
        "$CONSECUTIVE_FAILURES" "$RESTART_COUNT" false

    echo "[supervisor] backing off ${BACKOFF}s (consecutive_failures=$CONSECUTIVE_FAILURES)" >&2
    sleep "$BACKOFF"
done
