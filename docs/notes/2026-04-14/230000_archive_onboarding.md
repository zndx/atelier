# Archive/Unarchive + Onboarding BDD Flow

## Summary

Added `is_archived` boolean to both `data_sources` and `datasets` tables, with
full CRUD through DAO → gRPC → gateway. Default list queries exclude archived
items; `?include_archived=true` overrides. Archiving a data source cascades to
all its datasets.

Also created a BDD onboarding feature that exercises the new-user flow: archive
all existing data for a clean slate, unarchive just the OOTB sample, run the
classification pipeline, and verify embeddings data becomes available.

## Changes

### Database
- Migration `20260416000000_archive_flag.sql` — adds `is_archived BOOLEAN NOT NULL DEFAULT FALSE` to both tables with partial indexes

### Model + DAO
- `model.py`: `is_archived` column on `DataSource` and `Dataset`
- `dao.py`: `include_archived` parameter on `list_data_sources()`, `list_datasets()`, `list_dataset_versions()`
- New DAO methods: `archive_data_source()`, `unarchive_data_source()`, `archive_dataset()`, `unarchive_dataset()`
- Extracted `_source_to_dict()` helper (mirrors `_dataset_to_dict()`)

### Proto + Service
- `atelier.proto`: `is_archived` field on `DataSource` (8) and `ClassificationDataset` (12); `include_archived` on both list requests
- `service.py`: threads `include_archived` from request to DAO

### Gateway
- Updated `list_data_sources()` and `list_datasets()` to accept `include_archived` query param
- 4 new endpoints: `POST /api/data-sources/{id}/archive`, `/unarchive`, `POST /api/datasets/{id}/archive`, `/unarchive`
- Fixed `get_dataset_data()` to use `include_archived=True` (archived datasets' parquet files should still be servable)

### BDD
- `features/gateway/onboarding.feature` — 2 scenarios (tier-1, @slow)
- `features/gateway/step_defs/onboarding_steps.py` — archive setup with `context.add_cleanup()` teardown

## Also Fixed This Session
- Embeddings page HTTP 500 → graceful "No Embeddings Data Yet" info alert
- FSM SAMPLING→SAMPLING invalid transition → allow same-state advances as progress updates

## Verification
- 97 tier-0 scenarios pass (0 failures)
- Migration applied successfully
- Proto stubs regenerated
