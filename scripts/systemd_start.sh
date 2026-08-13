#!/usr/bin/env bash
# Idempotent oneshot start for atelier.service (systemd).
# Brings the capability engine on :50251, then blocks until lattice accept
# passes: codegen zndx.engine.v1.Engine/Status project=atelier.
# Does NOT start product servicer (:50071) / gateway / just up.
# Does NOT wait for vLLM cold-load (Status at gRPC bind is enough).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"

GRPC_PORT="${ATELIER_ENGINE_PORT:-50251}"
POLL_ITERS="${ATELIER_SYSTEMD_POLL_ITERS:-180}"
POLL_SLEEP="${ATELIER_SYSTEMD_POLL_SLEEP:-5}"
LOG_DIR="${ATELIER_ENGINE_LOG_DIR:-/tmp/atelier-engine}"
PID_FILE="${LOG_DIR}/unit_server.pid"
LOG_FILE="${LOG_DIR}/unit_server.log"

info() { echo "atelier.service: $*"; }

# Status via generated signals-protocol stubs (proto = specification / codegen).
# Reflection remains required for external bare grpcurl (lattice-ci checks it).
status_ok() {
  local py="${ROOT}/.devenv/state/venv/bin/python"
  [[ -x "$py" ]] || py="${ROOT}/.venv/bin/python"
  [[ -x "$py" ]] || py=python3
  ATELIER_ENGINE_PORT="${GRPC_PORT}" "$py" "$ROOT/scripts/zndx_status_ok.py" >/dev/null 2>&1
}

if status_ok; then
  info "already READY (Engine/Status :${GRPC_PORT}) — skip up"
  exit 0
fi

# Dual listeners on :50251 make Status flaky — prefer one owner.
if command -v ss >/dev/null 2>&1; then
  nlisten=$(ss -ltn 2>/dev/null | grep -cE ":${GRPC_PORT}\\s" || true)
  if [[ "${nlisten:-0}" -gt 1 ]]; then
    info "WARN: ${nlisten} listeners on :${GRPC_PORT} — killing foreign atelier.engine PIDs"
    for pid in $(ss -ltnp 2>/dev/null | grep ":${GRPC_PORT}" | grep -oP 'pid=\K[0-9]+' | sort -u); do
      cmd=$(ps -p "$pid" -o args= 2>/dev/null || true)
      if [[ "$cmd" == *atelier.engine* ]]; then
        info "  TERM pid=$pid ($cmd)"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    sleep 2
  fi
fi

if ! status_ok; then
  info "starting capability engine (python -m atelier.engine.server) on :${GRPC_PORT}"
  mkdir -p "$LOG_DIR"
  py="${ROOT}/.devenv/state/venv/bin/python"
  [[ -x "$py" ]] || py="${ROOT}/.venv/bin/python"
  [[ -x "$py" ]] || py=python3
  # Prefer aegir cuda-driver-libs if atelier has none (shared lab layout).
  CUDA_LIBS="${ROOT}/build/cuda-driver-libs"
  if [[ ! -d "$CUDA_LIBS" && -d "${ROOT}/../aegir/build/cuda-driver-libs" ]]; then
    CUDA_LIBS="${ROOT}/../aegir/build/cuda-driver-libs"
  elif [[ ! -d "$CUDA_LIBS" && -d /home/rch/local/src/zndx/aegir/build/cuda-driver-libs ]]; then
    CUDA_LIBS=/home/rch/local/src/zndx/aegir/build/cuda-driver-libs
  fi
  # New session so the oneshot can exit while the engine stays up.
  setsid env \
    LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    ATELIER_ENGINE_PORT="${GRPC_PORT}" \
    PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m atelier.engine.server \
    >>"$LOG_FILE" 2>&1 < /dev/null &
  echo $! > "$PID_FILE"
  info "engine pid=$(cat "$PID_FILE") log=$LOG_FILE"
fi

for i in $(seq 1 "$POLL_ITERS"); do
  if status_ok; then
    info "Engine/Status ready on :${GRPC_PORT} (iter=$i)"
    exit 0
  fi
  sleep "$POLL_SLEEP"
done

info "timed out waiting for zndx.engine.v1.Engine/Status on :${GRPC_PORT}" >&2
info "  Codegen probe: scripts/zndx_status_ok.py" >&2
info "  External DX:  grpcurl -plaintext 127.0.0.1:${GRPC_PORT} list" >&2
info "  External DX:  grpcurl -plaintext 127.0.0.1:${GRPC_PORT} zndx.engine.v1.Engine/Status" >&2
info "  Log: $LOG_FILE" >&2
info "  Product stack (optional): just up  # servicer :50071 / gateway — not accept gate" >&2
exit 1
