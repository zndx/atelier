#!/usr/bin/env bash
# bin/bootstrap-secrets.sh
#
# Materialize SOPS-encrypted artifacts so a bootable Atelier has the
# config defaults and ground-truth CSV it expects on disk.  Called
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

have_sops=0
if command -v sops >/dev/null 2>&1; then
  have_sops=1
fi

if [ "$have_sops" -eq 0 ]; then
  echo "bootstrap-secrets: sops not on PATH; skipping decrypts" >&2
  exit 0
fi

# 1) Deployment defaults (dotenv shape) → .env.cai
if [ -f .env.cai.enc ]; then
  if sops --decrypt --input-type dotenv --output-type dotenv \
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

# 2) Ground-truth CSV (binary shape) → build/data/ground_truth.csv
#    Encrypted file lives with the BDD corpus it validates; plaintext
#    lands under the gitignored build/ tree.
GT_ENC="features/fixtures/ground_truth.csv.enc"
if [ -f "$GT_ENC" ]; then
  mkdir -p build/data
  if sops --decrypt --input-type binary --output-type binary \
        "$GT_ENC" > build/data/ground_truth.csv 2>/dev/null; then
    echo "bootstrap-secrets: materialized build/data/ground_truth.csv"
  else
    rm -f build/data/ground_truth.csv
    echo "bootstrap-secrets: could not decrypt $GT_ENC — is SOPS_AGE_KEY set?" >&2
  fi
fi
