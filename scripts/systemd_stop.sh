#!/usr/bin/env bash
# Peer-scoped stop for atelier.service (systemd).
#
# Gaius-style wrap: take down THIS checkout's devenv graph (product
# servicer, gateway, vite, capability-engine). Co-tenant-safe: no
# host teardown / GPU wipe of Gaius or Ægir leases.
# Does NOT touch /tmp/zndx-gpu-leases (Gaius :50051 / Ægir :50151).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="/usr/local/bin:/usr/bin:/bin:${HOME}/.nix-profile/bin:${PATH:-}"
# shellcheck source=systemd-unit.sh
source "$ROOT/scripts/systemd-unit.sh"

info "peer-scoped devenv down (capability-engine :${GRPC_PORT})"
lattice_down
reap_foreign_engines

if [[ "$(listener_count)" -gt 0 ]]; then
  info "WARN :${GRPC_PORT} still listening after stop" >&2
  ss -ltnpH 2>/dev/null | grep -E ":${GRPC_PORT}[[:space:]]" >&2 || true
else
  info ":${GRPC_PORT} free"
fi
