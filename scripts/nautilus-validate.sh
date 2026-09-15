#!/usr/bin/env bash
# Follow-only: validate config/supervision/atelier.textproto with the local
# nautilus binary (same crate Gaius/Hermes run).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTANCE="${1:-$ROOT/config/supervision/atelier.textproto}"
BIN="${NAUTILUS_BIN:-}"
if [[ -z "$BIN" ]]; then
  for c in \
    "$HOME/local/src/zndx/gaius/external/nautilus/target/release/nautilus" \
    "$HOME/local/src/zndx/nautilus/target/release/nautilus"
  do
    [[ -x "$c" ]] && BIN="$c" && break
  done
fi
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "nautilus binary not found; set NAUTILUS_BIN" >&2
  exit 1
fi
exec "$BIN" validate "$INSTANCE"
