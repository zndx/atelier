-- migrate:up

-- NHSVM head registry: administrative pointers to trained factorized
-- NHSVM head artifacts living in build/cache/nhsvm/{head_sig}/.  Like
-- taxonomy_registry, PGlite holds *only* the pointer — the serialized
-- head (state_dict.pt + manifests) is first-class artifact data on
-- disk, content-addressable by head_sig.
--
-- Multiple versions of a single (taxonomy_id, encoder) may coexist:
--   - 'building' — training in progress, gate not yet evaluated
--   - 'current' — the version classify pipelines should consume
--   - 'stale'   — superseded but kept queryable for A/B / rollback
--   - 'archived'— retained for audit only
--
-- A partial unique index enforces at most one 'current' row per
-- (taxonomy_id, encoder) tuple — different encoders are independent
-- deployment lanes (a ModernBERT-base bump invalidates heads trained
-- against the prior encoder).
--
-- See docs/notes/2026-05-25/phase_gate_brief.md "Training protocol"
-- section + .claude/plans/lovely-doodling-badger.md for the design
-- this schema is derived from.

CREATE TABLE IF NOT EXISTS nhsvm_head_registry (
    id                   TEXT PRIMARY KEY,

    -- Logical taxonomy identifier (joins to taxonomy_registry.taxonomy_id
    -- semantically; not a FK because taxonomy_registry rows can be
    -- archived independently).
    taxonomy_id          TEXT NOT NULL,

    -- Content-addressable head identifier — first 16 hex chars of
    -- sha1(vocab_sig | reference_hash | corpus_hash | encoder |
    -- embed_dim | training_mode | fold_seed | augment_floor).  Stable
    -- across machines: two heads with the same head_sig are
    -- byte-identical operationally.
    head_sig             TEXT NOT NULL UNIQUE,

    -- Vocabulary signature: hash of the category_set codes the head
    -- was trained over.  Required for runtime compatibility check
    -- (head's taxonomy must match the pipeline's runtime taxonomy).
    vocab_sig            TEXT NOT NULL,

    -- Audit-derived reference content hash; tracks when the
    -- training_weights.json changed.  NULL for synth-only training
    -- mode (no reference dependency).
    reference_hash       TEXT,

    -- Synth corpus content hash.  NULL for reference-only training
    -- (a future mode); non-NULL for synth-only and reference-primary
    -- modes that consume the corpus.
    corpus_hash          TEXT,

    -- Training protocol that produced this head: 'synth-only',
    -- 'reference-primary', 'reference+synth'.  See
    -- VALID_TRAINING_MODES in atelier.optimize.svm.train.
    training_mode        TEXT NOT NULL,

    -- K-fold seed (NULL for non-k-fold modes).
    fold_seed            INTEGER,

    -- Encoder identity, e.g. 'answerdotai/ModernBERT-base'.  Encoder
    -- changes invalidate heads (different embedding spaces).
    encoder              TEXT NOT NULL,
    embedding_dim        INTEGER NOT NULL,

    -- Synth augmentation policy (NULL for synth-only training).
    augment_floor        INTEGER,

    -- Filesystem path of the saved adapter artifact.  Convention:
    --   build/cache/nhsvm/{head_sig}/{state_dict.pt, head_manifest.json,
    --                                  adapter_manifest.json}
    artifact_path        TEXT NOT NULL,

    -- Build lifecycle.  Transitions: building -> current -> stale -> archived.
    status               TEXT NOT NULL DEFAULT 'building',

    -- Quality metrics surface — Gate A held-out top-1 + fold variance
    -- + Gate B uplift/regression-band, etc.  JSONB so new metrics can
    -- land without schema migrations.  Promotion utilities check
    -- variance thresholds against this column before flipping to
    -- 'current' (see fold-1 underperformance task #271).
    metrics              JSONB,

    -- UX surface.
    summary              TEXT,

    -- Audit.
    built_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Partial unique index: at most one 'current' row per (taxonomy_id, encoder).
-- The DAO runs demote-prior + promote-new in a single transaction so this
-- index never sees two 'current' rows for the same (taxonomy_id, encoder).
CREATE UNIQUE INDEX IF NOT EXISTS idx_nhsvm_head_registry_one_current
    ON nhsvm_head_registry(taxonomy_id, encoder) WHERE status = 'current';

-- Lookup by status (UI listings, promotion queries, runtime current-head
-- resolution).
CREATE INDEX IF NOT EXISTS idx_nhsvm_head_registry_status
    ON nhsvm_head_registry(taxonomy_id, encoder, status, built_at DESC);

-- Per-run lineage: record which NHSVM head an FSM run consumed.
-- NULL = legacy per-vocab TF-IDF SVM path; non-NULL = registered
-- factorized NHSVM head path.  Allows A/B comparison and rollback
-- across runs without schema bifurcation.  Mirrors the
-- fsm_runs.taxonomy_collection pattern.
ALTER TABLE fsm_runs ADD COLUMN IF NOT EXISTS
    nhsvm_head TEXT REFERENCES nhsvm_head_registry(head_sig);

CREATE INDEX IF NOT EXISTS idx_fsm_runs_nhsvm_head
    ON fsm_runs(nhsvm_head) WHERE nhsvm_head IS NOT NULL;

-- migrate:down

DROP INDEX IF EXISTS idx_fsm_runs_nhsvm_head;
ALTER TABLE fsm_runs DROP COLUMN IF EXISTS nhsvm_head;

DROP INDEX IF EXISTS idx_nhsvm_head_registry_status;
DROP INDEX IF EXISTS idx_nhsvm_head_registry_one_current;
DROP TABLE IF EXISTS nhsvm_head_registry;
