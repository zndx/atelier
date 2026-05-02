<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Hive Data Source Auto-Discovery

**Date:** 2026-04-14

## Problem

When `ATELIER_DATA_CONNECTIONS` is configured in CAI, the gateway knows about
Hive connections but doesn't probe them for annotations tables. The OOTB sample
source seeds at startup, but Hive connections sat inert — no data source was
registered, so the UI dropdown only showed the sample.

## Solution

Added auto-discovery at gateway startup: for each configured connection, probe
databases for `annotations` tables matching the known schema (legacy or
universal format), validate columns, and register a data source via
`get_or_create_data_source()`. Idempotent on restart.

## Files Changed

- `src/atelier/data/connections.py` — `discover_hive_sources()` + helpers
- `src/atelier/gateway.py` — `_discover_and_register_hive_sources()` in lifespan
- `features/infra/config_lifecycle.feature` — tier-0 config parsing scenario
- `features/deployment/runtime_profile.feature` — tier-cai discovery scenario
- `features/infra/step_defs/config_steps.py` — step definitions

## Verification

- 98 tier-0 BDD scenarios pass (0 failed)
- New config parsing scenario validates `cml_data_connection_names` property
- CAI verification: restart gateway → check logs → /api/data-sources shows Hive source
