#!/bin/bash
# Atelier CML build hook — runs during Docker image builds
# for model endpoints and experiments. Installs into system Python
# (no virtualenv) so packages are available in the built image.
set -eox pipefail

pip3 install -e .
bash scripts/install_node.sh
source scripts/load_nvm.sh
cd scripts && npm install && cd ..
cd ui && npm install && npm run build
