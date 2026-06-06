-- migrate:up
CREATE TABLE IF NOT EXISTS agents (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    role TEXT,
    tool_ids TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id TEXT PRIMARY KEY,
    name TEXT,
    parquet_path TEXT,
    description TEXT,
    row_count BIGINT
);

-- migrate:down
DROP TABLE datasets;
DROP TABLE agents;
