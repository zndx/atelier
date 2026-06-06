#!/bin/bash
# Download and install Qdrant binary for CAI deployment.
# Follows the RAG Studio pattern: pre-compiled binary from GitHub releases.
set -euo pipefail

QDRANT_VERSION="v1.13.2"
QDRANT_TGZ=qdrant.tar.gz
DL_URL="https://github.com/qdrant/qdrant/releases/download/${QDRANT_VERSION}/qdrant-x86_64-unknown-linux-musl.tar.gz"

mkdir -p qdrant
cd qdrant

# Use pre-cached binary from base image if available, otherwise download
if [ -f /app/${QDRANT_TGZ} ]; then
    cp /app/${QDRANT_TGZ} ${QDRANT_TGZ}
else
    wget --no-verbose -O ${QDRANT_TGZ} ${DL_URL}
fi

tar xzf ${QDRANT_TGZ} && rm ${QDRANT_TGZ}
echo "Qdrant ${QDRANT_VERSION} installed."
