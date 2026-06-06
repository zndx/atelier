#!/bin/bash
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
