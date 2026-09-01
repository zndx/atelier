#!/usr/bin/env bash
# Oneshot start for atelier.service (systemd).
#
# Gaius-style wrap: systemd is the signals.target membership hook.
# The engine is a devenv process (`capability-engine`). `devenv up -d`
# is the same graph on Linux GPU hosts and macOS laptops — no
# machine-type gate. Accept is Engine/Status at bind, not vLLM SERVING.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"
# shellcheck source=systemd-unit.sh
source "$ROOT/scripts/systemd-unit.sh"

POLL_ITERS="${ATELIER_SYSTEMD_POLL_ITERS:-180}"
POLL_SLEEP="${ATELIER_SYSTEMD_POLL_SLEEP:-5}"

export_unit_runtime

# Skip only when Engine/Status is up AND a login-shell `devenv processes`
# sees the same compose. A /tmp leftover from an old unit is not enough.
if unit_already_ready && compose_visible; then
  info "already READY (login-visible compose, Engine/Status :${GRPC_PORT}) — skip up"
  exit 0
fi

if ! compose_visible; then
  info "login-shell devenv cannot see this checkout's compose — reaping leftover daemons"
  reap_atelier_compose
fi

if [[ "$(listener_count)" -gt 0 ]]; then
  info "reaping foreign :${GRPC_PORT} listeners before devenv up"
  reap_foreign_engines
fi

info "starting devenv graph (devenv up -d; capability-engine :${GRPC_PORT})"
if ! lattice_up; then
  info "up reported failure — will still poll (stack may already be live)"
fi

for i in $(seq 1 "$POLL_ITERS"); do
  if [[ "$(listener_count)" -gt 1 ]]; then
    info "WARN: $(listener_count) listeners on :${GRPC_PORT} (iter=$i)"
  elif unit_already_ready; then
    info "compose-owned Engine/Status ready on :${GRPC_PORT} (iter=$i)"
    # Warm the varnish-fronted waffle roster: the malloc store is empty
    # after a restart. Fire-and-forget; the public route primes the cache.
    (
      sleep 5
      curl -sf --max-time 60 -o /dev/null \
        "http://127.0.0.1:${CDSW_APP_PORT:-8090}/api/atelier/v1/federation/surfaces" || true
    ) >/dev/null 2>&1 &
    exit 0
  fi
  if (( i % 6 == 0 )); then
    info "waiting… iter=$i status=$(status_ok && echo ok || echo no) listeners=$(listener_count)"
  fi
  sleep "$POLL_SLEEP"
done

info "timed out waiting for compose-owned zndx.engine.v1.Engine/Status on :${GRPC_PORT}" >&2
info "  devenv processes status   # capability-engine should be running" >&2
info "  grpcurl -plaintext 127.0.0.1:${GRPC_PORT} list" >&2
info "  grpcurl -plaintext 127.0.0.1:${GRPC_PORT} zndx.engine.v1.Engine/Status" >&2
exit 1
