-- migrate:up

-- Taxonomy registry: administrative pointers to enriched annotation
-- collections living in Qdrant.  PGlite holds *only* the pointer
-- (where the collection lives, at which version, in which status) —
-- never the vectors or payload themselves.  Vectors + structured
-- annotation metadata are first-class data in the Qdrant collection
-- named here.
--
-- Multiple versions of a single taxonomy_id may coexist:
--   - 'building' — enrichment in progress
--   - 'current' — the version classify pipelines should consume
--   - 'stale'   — superseded but kept queryable for A/B / rollback
--   - 'archived'— retained for audit only
--
-- A partial unique index enforces at most one 'current' row per
-- taxonomy_id (same invariant pattern as ml_artifact_sets.is_active).
--
-- See docs/src/architecture/late-interaction-cosine.md for the design
-- doc this schema is derived from.
CREATE TABLE IF NOT EXISTS taxonomy_registry (
    id                   TEXT PRIMARY KEY,

    -- Logical taxonomy identifier, e.g., "default", "hive-poc/synth".
    -- Stable across versions; the version is captured separately below.
    taxonomy_id          TEXT NOT NULL,

    -- Where the source rows came from.  E.g., "default.annotations".
    -- Captured for lineage; not used at runtime.
    source_table         TEXT NOT NULL,

    -- Globally unique Qdrant collection name.  Convention:
    --   annotations_<taxonomy_id>_<augmentation_version>
    qdrant_collection    TEXT NOT NULL UNIQUE,

    -- URL of the Qdrant instance hosting the collection.  Nullable for
    -- environments that resolve the URL through cfg / env.
    qdrant_url           TEXT,

    -- Augmentation function version: prompt template + verifier suite +
    -- enrichment-script semantic version.  Bumping invalidates caches.
    augmentation_version TEXT NOT NULL,

    -- Embedding model identity.  Captured here so consumers can fail
    -- fast on mismatch (cf. ml_artifact_sets.embedding_model).
    embedding_model      TEXT NOT NULL,
    embedding_dim        INTEGER NOT NULL,

    -- Build lifecycle.  Transitions: building -> current -> stale -> archived.
    -- Only one 'current' row per taxonomy_id (partial index below).
    status               TEXT NOT NULL DEFAULT 'building',

    -- UX surface.
    summary              TEXT,

    -- Audit.
    built_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- (taxonomy_id, augmentation_version) is the natural compound key;
    -- enforced via a unique constraint so rebuilds of the same version
    -- are idempotent on the DAO side.
    UNIQUE (taxonomy_id, augmentation_version)
);

-- Partial unique index: at most one 'current' row per taxonomy_id.
-- Mirrors the ml_artifact_sets one-active invariant; the DAO runs
-- demote-prior + promote-new in a single transaction so this index
-- never sees two 'current' rows for the same taxonomy.
CREATE UNIQUE INDEX IF NOT EXISTS idx_taxonomy_registry_one_current
    ON taxonomy_registry(taxonomy_id) WHERE status = 'current';

-- Lookup by status (UI listings, build-orchestration queries).
CREATE INDEX IF NOT EXISTS idx_taxonomy_registry_status
    ON taxonomy_registry(taxonomy_id, status, built_at DESC);

-- Per-run lineage: record which enriched annotation collection an
-- FSM run consumed.  NULL = legacy single-vector cosine path; non-NULL
-- = late-interaction cosine path.  Allows A/B comparison and rollback
-- across runs without schema bifurcation.
ALTER TABLE fsm_runs ADD COLUMN IF NOT EXISTS
    taxonomy_collection TEXT REFERENCES taxonomy_registry(qdrant_collection);

CREATE INDEX IF NOT EXISTS idx_fsm_runs_taxonomy_collection
    ON fsm_runs(taxonomy_collection) WHERE taxonomy_collection IS NOT NULL;

-- migrate:down

DROP INDEX IF EXISTS idx_fsm_runs_taxonomy_collection;
ALTER TABLE fsm_runs DROP COLUMN IF EXISTS taxonomy_collection;

DROP INDEX IF EXISTS idx_taxonomy_registry_status;
DROP INDEX IF EXISTS idx_taxonomy_registry_one_current;
DROP TABLE IF EXISTS taxonomy_registry;
