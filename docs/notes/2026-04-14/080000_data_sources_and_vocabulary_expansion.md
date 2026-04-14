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

## Next: Phase 3 (Sample Data Generation)
- `scripts/generate_sample_source.py`: generate ~20 domain tables from ontology
- `data/sample/tables/*.csv` + `data/sample/ground_truth.json`
- Pre-classification for OOTB Embeddings page
- Gateway auto-import on first boot
