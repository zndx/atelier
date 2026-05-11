#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# Download and install the SOPS binary for CAI deployment.
# bin/bootstrap-secrets.sh requires sops on PATH; without it, the
# encrypted deployment defaults (.env.cai.enc) are never decrypted
# and the operator has to re-enter every model ARN, data connection,
# and governance URL in the AMP env form.  Installing sops here
# makes the encrypted-defaults channel actually functional on CAI.
#
# Same pattern as install_qdrant.sh: pre-cached binary from /app if
# the base image provides one, otherwise download from GitHub releases.
set -euo pipefail

SOPS_VERSION="v3.9.2"
ARCH="$(uname -m)"
case "${ARCH}" in
  x86_64) SOPS_ARCH="amd64" ;;
  aarch64|arm64) SOPS_ARCH="arm64" ;;
  *) echo "install_sops: unsupported arch ${ARCH}" >&2; exit 1 ;;
esac

SOPS_BIN="sops-${SOPS_VERSION}.linux.${SOPS_ARCH}"
DL_URL="https://github.com/getsops/sops/releases/download/${SOPS_VERSION}/${SOPS_BIN}"
DEST="$HOME/.local/bin/sops"

mkdir -p "$HOME/.local/bin"

# Skip the download if sops already runs and reports a matching-major
# version — speeds up idempotent re-installs during restart loops.
if command -v sops >/dev/null 2>&1; then
  echo "sops already on PATH: $(sops --version | head -1)"
  exit 0
fi

if [ -f "/app/${SOPS_BIN}" ]; then
  cp "/app/${SOPS_BIN}" "$DEST"
else
  wget --no-verbose -O "$DEST" "$DL_URL"
fi
chmod +x "$DEST"
echo "sops ${SOPS_VERSION} installed to $DEST ($("$DEST" --version | head -1))"
