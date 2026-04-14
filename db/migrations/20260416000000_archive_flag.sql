-- migrate:up

-- Archive flag for data sources and datasets (soft-hide, not delete).
-- Archived items are excluded from default list queries but remain
-- accessible via include_archived=true for audit and restore.
ALTER TABLE data_sources ADD COLUMN is_archived BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE datasets ADD COLUMN is_archived BOOLEAN NOT NULL DEFAULT FALSE;

-- Partial indexes: most queries filter on is_archived=FALSE
CREATE INDEX idx_data_sources_not_archived ON data_sources(is_archived) WHERE is_archived = FALSE;
CREATE INDEX idx_datasets_not_archived ON datasets(is_archived) WHERE is_archived = FALSE;

-- migrate:down

DROP INDEX IF EXISTS idx_datasets_not_archived;
DROP INDEX IF EXISTS idx_data_sources_not_archived;

ALTER TABLE datasets DROP COLUMN IF EXISTS is_archived;
ALTER TABLE data_sources DROP COLUMN IF EXISTS is_archived;
