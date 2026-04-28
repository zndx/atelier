#!/usr/bin/env bash
# bin/bootstrap-secrets.sh
#
# Materialize SOPS-encrypted artifacts so a bootable Atelier has the
# config defaults and curated-reference CSV it expects on disk.  Called
# from bin/start-app.sh (CAI deploy), justfile (ad-hoc), and
# devenv.nix enterShell (local dev) so every entrypoint gets the
# same state.
#
# Idempotent: re-running overwrites the plaintext artifacts with a
# fresh decrypt.  No-ops on missing ciphertext so the script is safe
# to call even when nothing is encrypted yet (fresh clone before
# first just encrypt-secrets).
#
# Requires either SOPS_AGE_KEY (the private key string) or
# SOPS_AGE_KEY_FILE (path to a file containing it).  On local
# maintainer machines the key usually lives at
# ~/.config/sops/age/keys.txt, which SOPS picks up without an env
# var.

set -euo pipefail

# Resolve repo root from this script's location (bin/ ↪ ..) so the
# script works from any CWD.
cd "$(dirname "$0")/.."

# Look for sops on PATH *and* in the user-local bin that
# scripts/install_sops.sh writes to.  On CAI the deploy hook installs
# sops to $HOME/.local/bin before start-app.sh runs; a fresh shell
# that hasn't re-sourced .profile / .bashrc won't have that dir on
# PATH yet, so prepend it defensively.
export PATH="$HOME/.local/bin:$PATH"

have_sops=0
if command -v sops >/dev/null 2>&1; then
  have_sops=1
fi

if [ "$have_sops" -eq 0 ]; then
  # Only noisy when encrypted defaults exist but sops to decrypt
  # them does not — that's a misconfigured deploy, not a clean
  # no-secrets checkout.
  if [ -f .env.cai.enc ] || [ -f features/fixtures/curated_reference.csv.enc ]; then
    echo "bootstrap-secrets: ERROR — sops is required to decrypt shipped deployment defaults but is not on PATH." >&2
    echo "bootstrap-secrets:   CAI: scripts/install_sops.sh should have installed it at \$HOME/.local/bin/sops — check the AMP install job output." >&2
    echo "bootstrap-secrets:   Local: install sops via devenv shell, 'brew install sops', or 'apt install sops'." >&2
    echo "bootstrap-secrets:   Continuing without the encrypted defaults — you'll need to set every ATELIER_* env var manually." >&2
  else
    echo "bootstrap-secrets: sops not on PATH and no encrypted artifacts present; nothing to do."
  fi
  exit 0
fi

# 1) Deployment defaults (dotenv shape) → .env.cai
if [ -f .env.cai.enc ]; then
  if sops --decrypt --output-type dotenv \
        .env.cai.enc > .env.cai 2>/dev/null; then
    echo "bootstrap-secrets: decrypted .env.cai ($(wc -l < .env.cai) lines)"
  else
    # Decrypt failed — no SOPS_AGE_KEY, wrong recipient, etc.
    # Clean up any partial file so start-app.sh's source check
    # doesn't trip on garbage, then continue.
    rm -f .env.cai
    echo "bootstrap-secrets: could not decrypt .env.cai.enc — is SOPS_AGE_KEY set?" >&2
  fi
fi

# 2) Curated-reference CSV (binary shape) → build/data/curated_reference.csv
#    Encrypted file lives with the BDD corpus it validates; plaintext
#    lands under the gitignored build/ tree.
REF_ENC="features/fixtures/curated_reference.csv.enc"
if [ -f "$REF_ENC" ]; then
  mkdir -p build/data
  if sops --decrypt --input-type binary --output-type binary \
        "$REF_ENC" > build/data/curated_reference.csv 2>/dev/null; then
    echo "bootstrap-secrets: materialized build/data/curated_reference.csv"
  else
    rm -f build/data/curated_reference.csv
    echo "bootstrap-secrets: could not decrypt $REF_ENC — is SOPS_AGE_KEY set?" >&2
  fi
fi
