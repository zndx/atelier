#!/bin/bash
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
