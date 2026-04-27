<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# AMP Deployment Fix: embedding-atlas Fork

## Problem

AMP deployment failed at `scripts/install_deps.py` line 51 — the npm workspace
build of embedding-atlas from source requires tools not available on bare CAI
runtimes:

- **viewer** package: `uv run python scripts/download_duckdb_extensions.py` (needs uv + Python)
- **umap-wasm**: needs Emscripten (`emcc`) for C++ → WASM compilation
- **density-clustering**: needs Rust + `wasm-bindgen` for Rust → WASM

## Root Cause Analysis

The fork's `.gitignore` globally ignores `dist/`, so `git submodule update --init`
on CAI produces a clean checkout with no pre-built artifacts. The build step then
tries to compile everything from source, which fails.

## Fix

1. **Fork (.gitignore)**: Added exceptions for `packages/embedding-atlas/dist/`
   and `packages/embedding-atlas/svelte/` so pre-built artifacts are committed

2. **Fork (commit)**: Committed the locally-built dist/ (4.5MB, 19 files including
   bundled JS chunks, workers, type declarations) to `rch/devenv` branch

3. **install_deps.py**: Replaced the 11-line npm workspace build block with a
   simple existence check — if `dist/react.js` is present, proceed; otherwise
   warn that the build may fail

4. **Documentation**: Updated CLAUDE.md, README.md, and memory to correctly
   characterize embedding-atlas as NOT dev-only — it's a production dependency
   with important modifications that must be carried forward

## Workflow for Future Fork Updates

1. Make changes in `external/embedding-atlas/`
2. Build locally: `cd external/embedding-atlas && npm install && npm run package -w ...`
3. Commit dist/ to the fork: `git add packages/embedding-atlas/dist/ && git commit && git push`
4. Update submodule pointer in main repo: `git add external/embedding-atlas && git commit`

## Files Changed

| File | Change |
|------|--------|
| `external/embedding-atlas/.gitignore` | Added `!packages/embedding-atlas/dist/` and `svelte/` |
| `external/embedding-atlas/packages/embedding-atlas/dist/*` | Committed 19 pre-built files |
| `scripts/install_deps.py` | Replaced npm build with dist/ existence check |
| `CLAUDE.md` | Corrected submodule description |
| `README.md` | Corrected submodule and deployment docs (3 locations) |
| `MEMORY.md` | Added embedding-atlas fork section |
