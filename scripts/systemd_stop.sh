#!/usr/bin/env bash
# Soft stop for atelier.service (systemd).
# Lattice-safe: only this project's capability engine on :50251.
# Does NOT stop product servicer (:50071), gateway, or just teardown / GPU wipe.
# Does NOT touch /tmp/zndx-gpu-leases (Gaius :50051 / Ægir :50151 sibling leases).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"

GRPC_PORT="${ATELIER_ENGINE_PORT:-50251}"
LOG_DIR="${ATELIER_ENGINE_LOG_DIR:-/tmp/atelier-engine}"
PID_FILE="${LOG_DIR}/unit_server.pid"

echo "atelier.service: stopping capability engine on :${GRPC_PORT}"

stop_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  local cmd
  cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
  if [[ "$cmd" != *atelier.engine* ]]; then
    return 0
  fi
  echo "atelier.service: TERM pid=$pid"
  kill -TERM "$pid" 2>/dev/null || true
}

if [[ -f "$PID_FILE" ]]; then
  stop_pid "$(cat "$PID_FILE")"
  rm -f "$PID_FILE"
fi

if command -v ss >/dev/null 2>&1; then
  for pid in $(ss -ltnp 2>/dev/null | grep ":${GRPC_PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u); do
    stop_pid "$pid"
  done
fi

sleep 2
if command -v ss >/dev/null 2>&1; then
  for pid in $(ss -ltnp 2>/dev/null | grep ":${GRPC_PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u); do
    cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
    if [[ "$cmd" == *atelier.engine* ]]; then
      echo "atelier.service: KILL pid=$pid after grace"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
fi
