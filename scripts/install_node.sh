#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# Install Node.js via nvm for CAI deployment.
# Follows the RAG Studio pattern: nvm into ~/.nvm, Node.js 22.
# Idempotent — skips if node is already available.
set +x

node --version 2>/dev/null
if [ $? -eq 0 ]; then
  echo "Node.js is already installed. Exiting."
  exit 0
fi

touch ~/.bashrc

# NVM installer updates .bashrc
wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash

source ~/.bashrc > /dev/null

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh" > /dev/null

nvm install v22.15.0
nvm use 22

echo "Node: $(which node) $(node -v)"
echo "npm: $(which npm) $(npm -v)"
