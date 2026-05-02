#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
