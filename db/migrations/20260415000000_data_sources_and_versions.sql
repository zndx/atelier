-- migrate:up

-- Data sources: OOTB sample, hive connections, etc.
CREATE TABLE data_sources (
    id TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,                -- 'sample' | 'hive'
    source_uri TEXT NOT NULL DEFAULT '',      -- '' for sample, 'conn/db' for hive
    display_name TEXT NOT NULL,
    vocabulary_mode TEXT NOT NULL DEFAULT 'universal',  -- 'universal' | 'hive'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata TEXT                             -- JSON: table_count, column_count, etc.
);

-- Extend datasets with versioning columns
ALTER TABLE datasets ADD COLUMN source_id TEXT REFERENCES data_sources(id);
ALTER TABLE datasets ADD COLUMN version_number INTEGER NOT NULL DEFAULT 1;
ALTER TABLE datasets ADD COLUMN is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE datasets ADD COLUMN summary TEXT;
ALTER TABLE datasets ADD COLUMN fsm_run_id TEXT;
ALTER TABLE datasets ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;

-- Extend fsm_runs with source tracking
ALTER TABLE fsm_runs ADD COLUMN source_id TEXT REFERENCES data_sources(id);

CREATE INDEX idx_datasets_source_version ON datasets(source_id, version_number DESC);

-- Seed the OOTB sample source
INSERT INTO data_sources (id, source_type, source_uri, display_name, vocabulary_mode)
VALUES ('ootb-sample', 'sample', '', 'OOTB Sample', 'universal');

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
