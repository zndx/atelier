#!/usr/bin/env bash
# Shared helpers for scripts/systemd_{start,stop}.sh.
# Source after ROOT is set. Co-tenant-safe: only touch atelier.engine / this
# checkout's devenv compose. Never teardown / GPU-wipe sibling leases.

: "${ROOT:?systemd-unit.sh: ROOT must be set}"

GRPC_PORT="${ATELIER_ENGINE_PORT:-50251}"

info() { echo "atelier.service: $*"; }

status_ok() {
  local py="${ROOT}/.devenv/state/venv/bin/python"
  [[ -x "$py" ]] || py="${ROOT}/.venv/bin/python"
  [[ -x "$py" ]] || py=python3
  ATELIER_ENGINE_PORT="${GRPC_PORT}" "$py" "$ROOT/scripts/zndx_status_ok.py" >/dev/null 2>&1
}

listener_pids() {
  command -v ss >/dev/null 2>&1 || return 0
  ss -ltnpH 2>/dev/null | awk -v p=":${GRPC_PORT}" '
    $4 ~ p"$" {
      if (match($0, /pid=[0-9]+/)) print substr($0, RSTART+4, RLENGTH-4)
    }' | sort -u
}

listener_count() {
  command -v ss >/dev/null 2>&1 || { echo 0; return; }
  ss -ltnH 2>/dev/null | grep -cE ":${GRPC_PORT}[[:space:]]" || true
}

is_atelier_engine() {
  local cmd
  cmd=$(ps -p "$1" -o args= 2>/dev/null || true)
  [[ "$cmd" == *atelier.engine* ]]
}

# True if pid is a descendant of this checkout's process-compose / devenv up.
owned_by_compose() {
  local pid="$1" p cmd cwd
  p="$pid"
  local i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    [[ -n "$p" && "$p" != 0 ]] || return 1
    cmd=$(ps -p "$p" -o args= 2>/dev/null || true)
    if [[ "$cmd" == *process-compose* || "$cmd" == *devenv-wrapped*daemon-processes* ]]; then
      cwd=$(readlink "/proc/${p}/cwd" 2>/dev/null || true)
      [[ -z "$cwd" || "$cwd" == "$ROOT" ]] && return 0
    fi
    p=$(ps -p "$p" -o ppid= 2>/dev/null | tr -d ' ')
  done
  return 1
}

unit_already_ready() {
  status_ok || return 1
  local n pid
  n=$(listener_count)
  [[ "${n:-0}" -eq 1 ]] || return 1
  pid=$(listener_pids | head -1)
  [[ -n "$pid" ]] || return 1
  owned_by_compose "$pid"
}

# TERM atelier.engine listeners that process-compose does not own (old setsid).
reap_foreign_engines() {
  local pid
  for pid in $(listener_pids); do
    is_atelier_engine "$pid" || continue
    if owned_by_compose "$pid"; then
      continue
    fi
    info "TERM foreign atelier.engine pid=$pid (not process-compose)"
    kill -TERM "$pid" 2>/dev/null || true
  done
  sleep 2
  for pid in $(listener_pids); do
    is_atelier_engine "$pid" || continue
    owned_by_compose "$pid" && continue
    info "KILL foreign atelier.engine pid=$pid after grace"
    kill -KILL "$pid" 2>/dev/null || true
  done
}

# Same graph as a laptop `devenv up -d`. Do not use `just up` — that is
# foreground `devenv up` and would block the oneshot forever.
lattice_up() {
  /bin/bash -lc "cd \"$ROOT\" && export PATH=\"/usr/local/bin:\$PATH\" && \
    export NIXPKGS_ALLOW_INSECURE=\"\${NIXPKGS_ALLOW_INSECURE:-1}\" && \
    devenv up -d"
}

lattice_down() {
  /bin/bash -lc "cd \"$ROOT\" && export PATH=\"/usr/local/bin:\$PATH\" && \
    (just down 2>/dev/null || devenv processes down || true)" || true
}
