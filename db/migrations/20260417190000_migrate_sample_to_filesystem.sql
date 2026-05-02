-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
--
-- This file contains material proprietary to Cloudera, Inc., and is provided
-- to authorized licensees solely for use in connection with the Cloudera AI
-- (CAI) Application from which it was obtained.  It may not be copied,
-- modified, redistributed, or used in any other manner without the express
-- written consent of Cloudera, Inc.

-- migrate:up
-- Flip the retired 'sample' source_type value to 'filesystem'.  The three
-- rows that historically used 'sample' (ootb-sample, synthetic, meta-tagging)
-- are all architecturally identical mount-based sources; the gateway
-- seeders re-upsert them with source_type='filesystem' + scheme'd URIs
-- on every startup, so this migration is primarily for clean state on
-- long-lived databases that boot without a seeder sweep.
UPDATE data_sources
SET source_type = 'filesystem'
WHERE source_type = 'sample';

-- migrate:down
UPDATE data_sources
SET source_type = 'sample'
WHERE source_type = 'filesystem';
