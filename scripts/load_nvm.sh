#!/bin/bash
# Source nvm into the current shell so node/npm are on PATH.
# Called by bin/start-app.sh and other scripts that need Node.js.
set -o pipefail
set +x

touch ~/.bashrc
source ~/.bashrc > /dev/null

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" > /dev/null
