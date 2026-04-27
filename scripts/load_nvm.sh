#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# Source nvm into the current shell so node/npm are on PATH.
# Called by bin/start-app.sh and other scripts that need Node.js.
#
# Deliberately avoids set -e / set -o pipefail — .bashrc on CAI runtimes
# often contains commands that exit non-zero and we must not propagate those.
set +x

touch ~/.bashrc
source ~/.bashrc > /dev/null 2>&1 || true

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" > /dev/null 2>&1 || true
