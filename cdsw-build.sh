#!/bin/bash
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

# Atelier CML build hook — runs during Docker image builds
# for model endpoints and experiments. Installs into system Python
# (no virtualenv) so packages are available in the built image.
set -eox pipefail

pip3 install -e .
bash scripts/install_node.sh
source scripts/load_nvm.sh
cd scripts && npm install && cd ..
cd ui && npm install && npm run build
