#!/usr/bin/env bash
# Lattice / capability engine — devenv process-compose child.
# Status is live at gRPC bind; do not wait for vLLM SERVING.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export ATELIER_ENGINE_PORT="${ATELIER_ENGINE_PORT:-50251}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"

# CUDA unmask when present (Linux GPU lab). Harmless no-op on a laptop.
for d in \
  "${ROOT}/build/cuda-driver-libs" \
  "${ROOT}/../aegir/build/cuda-driver-libs"
do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="${d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    break
  fi
done

# Refuse to sit beside a foreign binder (gRPC SO_REUSEPORT would dual-bind).
if command -v ss >/dev/null 2>&1 && ss -ltnH 2>/dev/null | grep -qE ":${ATELIER_ENGINE_PORT}[[:space:]]"; then
  echo "capability-engine: :${ATELIER_ENGINE_PORT} already bound — refuse dual-bind" >&2
  exit 1
fi

py="${ROOT}/.devenv/state/venv/bin/python"
[[ -x "$py" ]] || py="${ROOT}/.venv/bin/python"
[[ -x "$py" ]] || py=python3
exec "$py" -m atelier.engine.server
