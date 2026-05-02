<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Deployment

## Cloudera AI (CML)

Atelier deploys as a CAI Application from the Git URL `https://github.com/zndx/atelier`.

The `.project-metadata.yaml` defines two tasks:

1. **Install Dependencies** — Installs Python (via uv) and Node.js dependencies, builds the React frontend
2. **Start Atelier** — Launches the gRPC server and HTTP gateway on `CDSW_APP_PORT`

## Local Development

```bash
devenv shell          # Enter dev environment (loads .env automatically)
just install          # Install Python + Node dependencies
just proto            # Generate proto stubs
just resolve-config   # Materialize HOCON → build/config/atelier.env
just up               # Start gRPC + Vite dev server via devenv processes
```
