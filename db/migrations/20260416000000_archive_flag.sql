-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
--
-- This file contains material proprietary to Cloudera, Inc., and is provided
-- to authorized licensees solely for use in connection with the Cloudera AI
-- (CAI) Application from which it was obtained.  It may not be copied,
-- modified, redistributed, or used in any other manner without the express
-- written consent of Cloudera, Inc.

-- migrate:up

-- Archive flag for data sources and datasets (soft-hide, not delete).
-- Archived items are excluded from default list queries but remain
-- accessible via include_archived=true for audit and restore.
ALTER TABLE data_sources ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS is_archived BOOLEAN NOT NULL DEFAULT FALSE;

-- Partial indexes: most queries filter on is_archived=FALSE
CREATE INDEX IF NOT EXISTS idx_data_sources_not_archived ON data_sources(is_archived) WHERE is_archived = FALSE;
CREATE INDEX IF NOT EXISTS idx_datasets_not_archived ON datasets(is_archived) WHERE is_archived = FALSE;

-- migrate:down

DROP INDEX IF EXISTS idx_datasets_not_archived;
DROP INDEX IF EXISTS idx_data_sources_not_archived;

ALTER TABLE datasets DROP COLUMN IF EXISTS is_archived;
ALTER TABLE data_sources DROP COLUMN IF EXISTS is_archived;
