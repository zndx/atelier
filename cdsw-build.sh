#!/bin/bash
# Atelier CML build hook — runs during project import.
set -eox pipefail

pip3 install uv
uv sync --frozen
cd ui && npm install && npm run build
