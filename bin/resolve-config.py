#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Resolve HOCON config from the current environment.

Reads config/base.conf with env var substitution and writes hydrated
build artifacts. Called by devenv resolve-config process, bin/start-app.sh,
and `just resolve-config`.

Trusts whatever environment it's invoked in:
- devenv shell: dotenv.enable loads .env, env.* provides defaults
- AMP/CML: platform sets ANTHROPIC_API_KEY, AWS_*, CDSW_* at runtime
- CI: env vars injected by the pipeline

Outputs:
  build/config/atelier.env   — flat KEY=value (sourceable by shell)
  build/config/atelier.json  — JSON with redacted secrets (for conftest)
"""

from atelier.config import (
    load_config,
    materialize_config,
    materialize_config_json,
)

cfg = load_config()
materialize_config(cfg, "build/config/atelier.env")
materialize_config_json(cfg, "build/config/atelier.json")

# Summary
providers = []
if cfg.has_anthropic:
    providers.append("anthropic")
if cfg.has_bedrock:
    providers.append("bedrock")
creds = ", ".join(providers) if providers else "none"
print(f"[resolve-config] build/config/atelier.{{env,json}} — credentials: {creds}")
