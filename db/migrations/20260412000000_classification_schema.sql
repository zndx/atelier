-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
--
-- This file contains material proprietary to Cloudera, Inc., and is provided
-- to authorized licensees solely for use in connection with the Cloudera AI
-- (CAI) Application from which it was obtained.  It may not be copied,
-- modified, redistributed, or used in any other manner without the express
-- written consent of Cloudera, Inc.

-- migrate:up
CREATE TABLE IF NOT EXISTS fsm_runs (
    id TEXT PRIMARY KEY,
    state TEXT NOT NULL DEFAULT 'IDLE',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    config TEXT,
    progress TEXT,
    error TEXT,
    result_path TEXT
);

CREATE TABLE IF NOT EXISTS classification_runs (
    id TEXT PRIMARY KEY,
    fsm_run_id TEXT REFERENCES fsm_runs(id),
    table_name TEXT NOT NULL,
    column_name TEXT NOT NULL,
    predicted_code TEXT,
    predicted_label TEXT,
    confidence REAL,
    belief REAL,
    plausibility REAL,
    conflict REAL,
    evidence_sources TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vocabularies (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    category_count INTEGER,
    categories TEXT
);

-- migrate:down
DROP TABLE IF EXISTS classification_runs;
DROP TABLE IF EXISTS vocabularies;
DROP TABLE IF EXISTS fsm_runs;
