-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
--
-- This file contains material proprietary to Cloudera, Inc., and is provided
-- to authorized licensees solely for use in connection with the Cloudera AI
-- (CAI) Application from which it was obtained.  It may not be copied,
-- modified, redistributed, or used in any other manner without the express
-- written consent of Cloudera, Inc.

-- migrate:up

-- Data sources: OOTB sample, hive connections, etc.
CREATE TABLE IF NOT EXISTS data_sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,                -- 'sample' | 'hive'
    source_uri TEXT NOT NULL DEFAULT '',      -- '' for sample, 'conn/db' for hive
    display_name TEXT NOT NULL,
    vocabulary_mode TEXT NOT NULL DEFAULT 'universal',  -- 'universal' | 'hive'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                             -- JSON: table_count, column_count, etc.
);

-- Extend datasets with versioning columns
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS source_id TEXT REFERENCES data_sources(id);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS version_number INTEGER NOT NULL DEFAULT 1;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS fsm_run_id TEXT;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Extend fsm_runs with source tracking
ALTER TABLE fsm_runs ADD COLUMN IF NOT EXISTS source_id TEXT REFERENCES data_sources(id);

CREATE INDEX IF NOT EXISTS idx_datasets_source_version ON datasets(source_id, version_number DESC);

-- Seed the built-in sample source.  The 'ootb-' id prefix is retained as
-- an internal marker (distinguishes shipped sources from user-registered
-- connections); the user-facing display name is simply "Sample".
INSERT INTO data_sources (id, source_type, source_uri, display_name, vocabulary_mode)
VALUES ('ootb-sample', 'sample', '', 'Sample', 'universal')
ON CONFLICT (id) DO NOTHING;

-- migrate:down

DROP INDEX IF EXISTS idx_datasets_source_version;

ALTER TABLE fsm_runs DROP COLUMN IF EXISTS source_id;

ALTER TABLE datasets DROP COLUMN IF EXISTS created_at;
ALTER TABLE datasets DROP COLUMN IF EXISTS fsm_run_id;
ALTER TABLE datasets DROP COLUMN IF EXISTS summary;
ALTER TABLE datasets DROP COLUMN IF EXISTS is_active;
ALTER TABLE datasets DROP COLUMN IF EXISTS version_number;
ALTER TABLE datasets DROP COLUMN IF EXISTS source_id;

DROP TABLE IF EXISTS data_sources;
