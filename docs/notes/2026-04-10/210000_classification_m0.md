<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# M0: Classification Pipeline Implementation

## Summary

Implemented the complete Milestone 0 classification pipeline: DST evidence
fusion with 3 independent sources (cosine similarity, pattern detection,
column name matching), AgentFSM background state machine, and end-to-end
pipeline from vocabulary loading through evaluation.

## What was built

### Core classify module (10 files)
- `belief.py` — DST primitives (FocalElement, BeliefAssignment, dempster_combine, FrameOfDiscernment)
- `features.py` — 12 discrete features + 8 regex pattern detectors
- `taxonomy.py` — HierarchicalCategorySet with hive/JSON/mock loaders
- `mass_functions.py` — 3 active evidence converters (cosine, pattern, name_match) + 2 stubs (catboost, svm)
- `embedding.py` — Sentence-transformer cosine classifier (all-MiniLM-L6-v2)
- `sampler.py` — Hive metadata sampling with mock fixture fallback
- `synth.py` — Synthetic data generation stub (M1)
- `pipeline.py` — End-to-end orchestration driving FSM states
- `fsm.py` — AgentFSM with DB persistence
- `fixtures/` — 24-category mock vocabulary + 8 realistic tables (50 columns)

### Infrastructure
- New DB migration: fsm_runs, classification_runs, vocabularies tables
- ORM models: FSMRun, ClassificationRun, Vocabulary
- DAO: upsert_fsm_run, get_fsm_run, list_fsm_runs
- Proto: GetFSMStatus, StartClassification RPCs
- Gateway: /api/fsm/status, /api/fsm/start, /api/fsm/runs
- Config: classify{} HOCON section with connection, database, sample_size, etc.

### Frontend
- ClassificationPipelineCard on Status page with auto-polling
- New agent role themes (sampler, synth_generator) on Agents page
- Updated canvas layout for 5-agent topology

### Skills & Agents
- 5 new skills: sample-metadata, discover-tables, load-annotations, generate-synth-tables, classify-columns
- 2 new keystone agents: Metadata Sampler, Synthetic Generator
- Updated agent seeding in bin/start-app.sh

### BDD
- 8 new tier-0 scenarios covering DST, features, patterns, name matching, pipeline E2E, FSM transitions
- All 41 tier-0 scenarios pass (0 failures)

### Documentation
- `docs/src/architecture/classification.md` — Full methodology and architecture doc
- Session notes

## Test Results

Pipeline E2E with mock data:
- 16 leaf categories loaded
- 8 tables discovered, 50 columns sampled
- Name match + pattern detection classify most columns correctly
- Accuracy against ground truth exceeds threshold
- Pipeline reaches CONVERGED state in ~0.2s

## Next Steps (M1)
- CatBoost + SVM classifiers with synthetic training data
- Claude Agent SDK-driven synthetic data generation
- Bootstrap convergence loop (LLM sweep → ML validation → targeted revisit)
- pyarrow dependency for parquet output
- sentence-transformers dependency for cosine classification on CAI
