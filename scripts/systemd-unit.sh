#!/usr/bin/env bash
# Shared helpers for scripts/systemd_{start,stop}.sh.
# Source after ROOT is set. Co-tenant-safe: only touch atelier.engine / this
# checkout's devenv compose. Never teardown / GPU-wipe sibling leases.

: "${ROOT:?systemd-unit.sh: ROOT must be set}"

GRPC_PORT="${ATELIER_ENGINE_PORT:-50251}"

info() { echo "atelier.service: $*"; }

# Optional: only if the unit still exports ATELIER_DEVENV_RUNTIME (legacy).
# Default is devenv's own runtime so systemd and a login-shell `devenv up`
# share one process-compose graph. A dedicated runtime makes
# `devenv processes restart gateway` miss the unit's stack.
export_unit_runtime() {
  if [[ -n "${ATELIER_DEVENV_RUNTIME:-}" ]]; then
    mkdir -p "$ATELIER_DEVENV_RUNTIME"
    export DEVENV_RUNTIME="$ATELIER_DEVENV_RUNTIME"
  fi
  if [[ -d "/run/user/$(id -u)" ]]; then
    export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
  fi
}

_devenv_lc() {
  export_unit_runtime
  /bin/bash -lc "cd \"$ROOT\" && export PATH=\"/usr/local/bin:\$PATH\" && \
    export NIXPKGS_ALLOW_INSECURE=\"\${NIXPKGS_ALLOW_INSECURE:-1}\" && \
    export XDG_RUNTIME_DIR=\"${XDG_RUNTIME_DIR:-}\" && \
    ${DEVENV_RUNTIME:+export DEVENV_RUNTIME=\"$DEVENV_RUNTIME\" &&} \
    $*"
}

# Same graph as a laptop `devenv up -d`. Do not use `just up` — that is
# foreground `devenv up` and would block the oneshot forever.
lattice_up() {
  _devenv_lc "devenv up -d"
}

lattice_down() {
  _devenv_lc "just down 2>/dev/null || devenv processes down || true" || true
}

# Login-shell `devenv processes` can see this checkout's compose.
compose_visible() {
  _devenv_lc "devenv processes list" >/dev/null 2>&1
}

# PIDs of devenv process-compose daemons whose cwd is this checkout
# (any runtime — leftover /tmp/devenv-* from a unit that lacked
# XDG_RUNTIME_DIR, or a stale session stack).
atelier_compose_pids() {
  local pid cwd args
  while read -r pid args; do
    [[ "$args" == *devenv-wrapped*daemon-processes* ]] || continue
    cwd=$(readlink "/proc/${pid}/cwd" 2>/dev/null || true)
    [[ "$cwd" == "$ROOT" ]] || continue
    printf '%s\n' "$pid"
  done < <(ps -eo pid=,args=)
}

term_pid() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
}

kill_pid() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null || return 0
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

# Reap leftover compose daemons for THIS checkout only (cwd match).
reap_atelier_compose() {
  local pid still i
  for pid in $(atelier_compose_pids); do
    info "TERM devenv compose pid=$pid (cwd=$ROOT)"
    term_pid "$pid"
  done
  for i in $(seq 1 20); do
    still=$(atelier_compose_pids || true)
    [[ -z "${still}" ]] && break
    sleep 1
  done
  for pid in $(atelier_compose_pids); do
    info "KILL devenv compose pid=$pid after grace"
    kill_pid "$pid"
  done
}

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


