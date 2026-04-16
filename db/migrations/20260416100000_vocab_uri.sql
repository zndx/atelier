-- migrate:up
ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS vocab_uri TEXT NOT NULL DEFAULT '';

-- migrate:down
ALTER TABLE data_sources DROP COLUMN IF EXISTS vocab_uri;
