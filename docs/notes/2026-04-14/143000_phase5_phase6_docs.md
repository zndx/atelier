<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Phase 5+6 Documentation

Documented the proposed MLflow and Hive integrations in mdbook architecture pages
instead of implementing them directly (hard to test without CAI services).

## New Pages

- `docs/src/architecture/data-sources.md` — documents the completed data source +
  versioning model (Phases 1-4): OOTB sample source, expanded ontology (300 leaves),
  mixed-domain tables, auto-import, vocabulary routing, API endpoints, UI integration

- `docs/src/architecture/integrations.md` — documents proposed Phase 5 (MLflow bridge)
  and Phase 6 (Hive data source):
  - MLflow: write-then-reconcile pattern, queue format, experiment structure,
    module design, gating, configuration
  - Hive: data flow, vocabulary composition, source creation, pipeline routing,
    no new modules needed (just wiring)

## Other Changes

- Updated `SUMMARY.md` with new page links
- Updated classification.md milestones: M5 (done), M6 (proposed → integrations.md)

## Commits This Session

- `e44ce06` — source-aware pipeline routing with OOTB sample auto-import
  (sampler, taxonomy, pipeline, gateway, dao, Landing.tsx, Status.tsx)
