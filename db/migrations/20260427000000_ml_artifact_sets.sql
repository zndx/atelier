-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
--
-- This file contains material proprietary to Cloudera, Inc., and is provided
-- to authorized licensees solely for use in connection with the Cloudera AI
-- (CAI) Application from which it was obtained.  It may not be copied,
-- modified, redistributed, or used in any other manner without the express
-- written consent of Cloudera, Inc.

-- migrate:up

-- ML artifact sets: bundles of trained-model artifacts produced by an
-- FSM run.  A row indexes the on-disk files (CatBoost, SVM, UMAP, +
-- sidecars) so an Extend Classification run can replay them on new
-- data without re-running the full training pipeline.
--
-- Lineage shape borrows from OpenLineage: this row is the link between
-- the producing Run (fsm_runs.id) and any downstream Run that consumes
-- it (parent_artifact_set_id self-FK + datasets.artifact_set_id back-ref).
CREATE TABLE IF NOT EXISTS ml_artifact_sets (
    id TEXT PRIMARY KEY,
    source_id TEXT REFERENCES data_sources(id),
    fsm_run_id TEXT REFERENCES fsm_runs(id),
    parent_artifact_set_id TEXT REFERENCES ml_artifact_sets(id),

    -- On-disk paths (relative to the project root for portability across
    -- devenv / CAI deploys).  The DAO resolves them through cfg.results_dir.
    catboost_path TEXT NOT NULL,
    catboost_classes_path TEXT NOT NULL,
    svm_path TEXT,
    svm_classes_path TEXT,
    umap_path TEXT,

    -- Compatibility metadata.  vocab_signature is sha256 of sorted
    -- json-encoded classes; cheap equality + dedup check.
    classes TEXT NOT NULL,
    feature_groups TEXT,
    vocab_signature TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dim INTEGER NOT NULL,

    -- UX surface.
    display_name TEXT,
    summary TEXT,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    is_archived BOOLEAN NOT NULL DEFAULT FALSE,

    -- OpenLineage projection (schema, parent_run, custom ml-artifact facet).
    facets TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ml_artifact_sets_source
    ON ml_artifact_sets(source_id, created_at DESC);

-- Partial unique index: at most one globally active artifact set.
-- The DAO method set_active_artifact_set runs the deactivation +
-- activation in a single transaction; this index is the Postgres-side
-- invariant that catches any miswiring.
CREATE UNIQUE INDEX IF NOT EXISTS idx_ml_artifact_sets_one_active
    ON ml_artifact_sets((is_active)) WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_ml_artifact_sets_not_archived
    ON ml_artifact_sets(is_archived) WHERE is_archived = FALSE;

-- Extend datasets to record run-kind + lineage:
--  - artifact_set_id: which artifact set produced (for classify) OR was
--    consumed (for extend) by the run that produced this dataset.
--  - parent_dataset_id: for extend runs, the dataset of the source
--    classify run we're extending against.  Always NULL for classify runs.
--  - run_kind: 'classify' or 'extend'.  Used by the UI to render
--    differently and by the Embeddings page for context.
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS
    artifact_set_id TEXT REFERENCES ml_artifact_sets(id);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS
    parent_dataset_id TEXT REFERENCES datasets(id);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS
    run_kind TEXT NOT NULL DEFAULT 'classify';

CREATE INDEX IF NOT EXISTS idx_datasets_artifact_set
    ON datasets(artifact_set_id);

-- migrate:down

DROP INDEX IF EXISTS idx_datasets_artifact_set;
ALTER TABLE datasets DROP COLUMN IF EXISTS run_kind;
ALTER TABLE datasets DROP COLUMN IF EXISTS parent_dataset_id;
ALTER TABLE datasets DROP COLUMN IF EXISTS artifact_set_id;

DROP INDEX IF EXISTS idx_ml_artifact_sets_not_archived;
DROP INDEX IF EXISTS idx_ml_artifact_sets_one_active;
DROP INDEX IF EXISTS idx_ml_artifact_sets_source;
DROP TABLE IF EXISTS ml_artifact_sets;
