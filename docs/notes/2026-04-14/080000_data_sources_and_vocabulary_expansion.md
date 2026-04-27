<!--
Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

This file contains material proprietary to Cloudera, Inc., and is provided
to authorized licensees solely for use in connection with the Cloudera AI
(CAI) Application from which it was obtained.  It may not be copied,
modified, redistributed, or used in any other manner without the express
written consent of Cloudera, Inc.
-->

# Data Sources, Versioning, and Vocabulary Expansion

Date: 2026-04-14

## Phase 2: Data Model (Complete)

### Data Sources + Versioning
- Created `data_sources` table: id, source_type (sample|hive), display_name, vocabulary_mode
- Extended `datasets` with: source_id, version_number, is_active, summary, fsm_run_id, created_at
- Extended `fsm_runs` with: source_id
- Seeded OOTB sample source (`ootb-sample`)

### Proto + Gateway
- New `DataSource` message, `ListDataSources` RPC
- `ListDatasetsRequest.source_id` filter
- `StartClassificationRequest.source_id`
- REST: `GET /api/data-sources`, `GET /api/datasets?source_id=`, `POST /api/datasets/{id}/activate`

### UI
- DatasetContext refactored: source-aware with `sources`, `activeSourceId`, source-filtered dataset fetching
- Status page: source dropdown + version table (replaces flat dataset dropdown)
- Landing page: sources count suffix on Terms card

Commit: `0bc7035` → pushed

## Phase 1: Vocabulary Expansion (Complete)

### scripts/expand_vocabulary.py
- One-time developer script building the expanded ICE.* vocabulary
- 325 total categories: 300 leaves + 25 internal nodes
- Output: `data/sample/ontology.json`

### Breakdown by CCO ICE Trichotomy
| Subtree | Leaves | Examples |
|---------|--------|---------|
| ICE.NONSENSITIVE.DESIGNATIVE | 68 | NAME.PERSON, CODE.ISBN, GEO.COUNTRY, REF.URL |
| ICE.NONSENSITIVE.DESCRIPTIVE | 120 | TEXT.DESCRIPTION, MEASUREMENT.TEMPERATURE, TEMPORAL.DATE, STATISTICAL.MEAN |
| ICE.NONSENSITIVE.PRESCRIPTIVE | 21 | FORMULA, RULE, CONFIG, REGEX, SLA, POLICY |
| ICE.SENSITIVE | 66 | PID.CONTACT.EMAIL, PID.HEALTH.DIAGNOSIS, BUSINESS.TRADE_SECRET, TECHNICAL.API_KEY |
| ICE.METADATA | 25 | TIMESTAMP, RECID, LINEAGE, ETL_BATCH, TENANT_ID |

### Design Principle (Reiterated)
Every category uses our ICE.* coding scheme with BFO/CCO-grounded descriptions.
External sources (GitTables, meta-tagging, DBpedia) inform conceptual coverage.
The mapping goes OUTWARD via `atelier-vocab.ttl`, not inward.

### TTL Updates
- New internal nodes: ICE.NONSENSITIVE.DESIGNATIVE (⊑ cco:DesignativeICE),
  ICE.NONSENSITIVE.DESCRIPTIVE (⊑ cco:DescriptiveICE),
  ICE.NONSENSITIVE.PRESCRIPTIVE (⊑ cco:PrescriptiveICE),
  ICE.SENSITIVE.PID.HEALTH, ICE.SENSITIVE.BUSINESS
- Version bumped to 0.2.0

## Phase 3: Sample Data Generation (Complete)

### scripts/generate_sample_source.py
- 25 realistic mixed-domain tables, 300 columns total, 100 rows each
- Tables mix columns from different ontology subtrees (like real GitTables data)
  - `customers`: identity + contact + metadata + categorical
  - `orders`: financial + temporal + categorical + measurement
  - `dataset_7`: opaque table name, opaque column names
- ~25% of columns get opaque/abbreviated names (field_42, var_abc, etc.)
- Column order shuffled within tables to prevent position-based patterns
- Output: `data/sample/tables/*.csv` + `data/sample/ground_truth.json`
- All 300 leaf categories have generators, 100% coverage

### Design Principles
- Mixed-domain tables: no table is purely from one ontology subtree
- Opaque naming: 75/300 columns (~25%) use coded names
- Realistic patterns: inspired by GitTables organic relational table layouts
- Deterministic: seed=42, fully reproducible

## Next
- Pre-classification for OOTB Embeddings page
- Gateway auto-import on first boot
