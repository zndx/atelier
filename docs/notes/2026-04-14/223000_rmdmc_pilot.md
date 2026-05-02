<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# R-MDMC Pilot Implementation

## Summary

Row-level Monte Carlo sampling for bootstrap iteration diversity. Addresses the
"50 fetched, 5 used" gap: the system discards 90-95% of available row data before
any classifier sees it. Now stores all fetched rows in a reservoir and rotates
subsets across bootstrap iterations.

## Changes

### New files
- `src/atelier/classify/row_sampler.py` — RowMCConfig + 3 selection strategies
  (head/random/stratified with pattern-frequency tie-breaker)
- `features/agent/step_defs/row_mc_steps.py` — 5 BDD scenario step definitions

### Modified files
- `src/atelier/classify/sampler.py` — `ColumnSample.all_values` field, 3 loaders updated
- `config/base.conf` — `classify.row_mc {}` section (6 params)
- `src/atelier/config.py` — 6 HOCON mappings + 6 dataclass fields
- `src/atelier/classify/pipeline.py` — Row MC rotation in convergence loop,
  row-stability recording, adaptive escalation
- `src/atelier/classify/bootstrap.py` — `row_labels_history` in BootstrapState,
  `row_stability()` helper
- `src/atelier/classify/llm_backend.py` — Total values count annotation in prompt
- `features/agent/classification.feature` — 5 new @row-mc scenarios
- `features/steps/__init__.py` — row_mc_steps import

## Design Decisions

- **Grok review adopted selectively**: pattern-frequency tie-breaker (adopt),
  weighted mean pooling (defer), distinct-only values (reject — preserves frequency info)
- **Zero-cost when disabled**: `enabled = false` default, all_values still populated
- **Backward compatible**: `col.values` remains `all_values[:5]` unless row MC rotates it
- **Adaptive escalation**: row-unstable columns (stability < 0.5) get full reservoir
- **Pilot scope only**: multi-view embedding, mean-pooled propagation, adaptive
  max_values deferred to integration phase

## Test Results

97 tier-0 non-slow scenarios passing (5 new row-MC scenarios included).
