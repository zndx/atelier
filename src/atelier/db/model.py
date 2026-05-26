# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""SQLAlchemy ORM models for Atelier state."""

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class DataSource(Base):
    """A data source (OOTB sample, hive connection, etc.)."""

    __tablename__ = "data_sources"

    id = Column(String, primary_key=True, nullable=False)
    # 'filesystem' (local dir/zip mount, optional annotations.csv)
    # | 'hive' (Cloudera data-platform connection + database).
    # Future schemes — 's3', 'jdbc' — slot in without further model
    # changes; source_uri carries the scheme prefix
    # (file:///…, hive://…, s3://…).  'sample' is retired from this
    # field but the migration keeps the row intact.
    source_type = Column(String, nullable=False)
    source_uri = Column(String, nullable=False, default="")
    display_name = Column(String, nullable=False)
    vocabulary_mode = Column(String, nullable=False, default="universal")
    vocab_uri = Column(String, nullable=False, default="")  # e.g. "meta.vocab"
    created_at = Column(DateTime, server_default=func.now())
    source_metadata = Column("metadata", Text, nullable=True)  # JSON
    is_archived = Column(Boolean, nullable=False, default=False)


class Agent(Base):
    """Persisted keystone agent definition."""

    __tablename__ = "agents"

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    role = Column(String, nullable=True)
    tool_ids = Column(Text, nullable=True)  # JSON array


class Dataset(Base):
    """Reference to a classification parquet dataset (versioned under a source)."""

    __tablename__ = "datasets"

    id = Column(String, primary_key=True, nullable=False)
    name = Column(String, nullable=True)
    parquet_path = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    row_count = Column(BigInteger, nullable=True)
    source_id = Column(String, nullable=True)          # FK → data_sources.id
    version_number = Column(Integer, nullable=False, default=1)
    is_active = Column(Boolean, nullable=False, default=True)
    summary = Column(Text, nullable=True)
    fsm_run_id = Column(String, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    is_archived = Column(Boolean, nullable=False, default=False)
    # Artifact-set lineage (added 20260427).  classify runs set
    # artifact_set_id to the row they produced; extend runs set it to
    # the row they consumed AND parent_dataset_id to the source dataset.
    artifact_set_id = Column(String, nullable=True)    # FK → ml_artifact_sets.id
    parent_dataset_id = Column(String, nullable=True)  # FK → datasets.id
    run_kind = Column(String, nullable=False, default="classify")  # 'classify'|'extend'


class MLArtifactSet(Base):
    """Bundle of trained-model artifacts produced by an FSM run.

    Indexes the on-disk paths of CatBoost / SVM / UMAP + sidecars so an
    Extend Classification run can replay them on new data without
    re-running the full training pipeline.  Lineage shape borrows from
    OpenLineage: this row links the producing Run (``fsm_run_id``) to
    any downstream Run that consumes it via ``Dataset.artifact_set_id``.
    """

    __tablename__ = "ml_artifact_sets"

    id = Column(String, primary_key=True, nullable=False)
    source_id = Column(String, nullable=True)          # FK → data_sources.id
    fsm_run_id = Column(String, nullable=True)         # FK → fsm_runs.id
    parent_artifact_set_id = Column(String, nullable=True)  # self-FK

    catboost_path = Column(Text, nullable=False)
    catboost_classes_path = Column(Text, nullable=False)
    svm_path = Column(Text, nullable=True)
    svm_classes_path = Column(Text, nullable=True)
    umap_path = Column(Text, nullable=True)

    classes = Column(Text, nullable=False)             # JSON array
    feature_groups = Column(Text, nullable=True)       # JSON array
    vocab_signature = Column(String, nullable=False)   # sha256(sorted(classes))
    embedding_model = Column(String, nullable=False)
    embedding_dim = Column(Integer, nullable=False)

    display_name = Column(String, nullable=True)
    summary = Column(Text, nullable=True)
    is_active = Column(Boolean, nullable=False, default=False)
    is_archived = Column(Boolean, nullable=False, default=False)

    facets = Column(Text, nullable=True)               # JSON: OpenLineage projection

    created_at = Column(DateTime, server_default=func.now())


class TaxonomyRegistry(Base):
    """Administrative pointer to an enriched annotation collection in Qdrant.

    PGlite holds *only* the pointer — Qdrant holds the vectors + payload
    (one multi-vector point per annotation with structured JSON metadata).
    Multiple versions of a single ``taxonomy_id`` may coexist; the
    partial unique index ``idx_taxonomy_registry_one_current`` enforces
    at most one ``status='current'`` row per taxonomy.

    See ``docs/src/architecture/late-interaction-cosine.md``.
    """

    __tablename__ = "taxonomy_registry"

    id = Column(String, primary_key=True, nullable=False)

    # Logical taxonomy identifier (e.g., "default", "hive-poc/synth").
    # Stable across versions; version captured separately below.
    taxonomy_id = Column(String, nullable=False)

    source_table = Column(String, nullable=False)
    qdrant_collection = Column(String, nullable=False, unique=True)
    qdrant_url = Column(String, nullable=True)

    # Prompt template + verifier suite + enrichment-script semantic version.
    augmentation_version = Column(String, nullable=False)

    embedding_model = Column(String, nullable=False)
    embedding_dim = Column(Integer, nullable=False)

    # 'building' | 'current' | 'stale' | 'archived'
    status = Column(String, nullable=False, default="building")

    summary = Column(Text, nullable=True)

    built_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class NhsvmHeadRegistry(Base):
    """Administrative pointer to a trained factorized NHSVM head artifact.

    PGlite holds the pointer; serialized head + manifests live on disk at
    ``build/cache/nhsvm/{head_sig}/``.  Multiple versions of a single
    (taxonomy_id, encoder) may coexist; the partial unique index
    ``idx_nhsvm_head_registry_one_current`` enforces at most one
    ``status='current'`` row per (taxonomy_id, encoder) tuple.

    See ``.claude/plans/lovely-doodling-badger.md`` Step 4 for the
    design this schema is derived from.
    """

    __tablename__ = "nhsvm_head_registry"

    id = Column(String, primary_key=True, nullable=False)

    taxonomy_id = Column(String, nullable=False)
    head_sig = Column(String, nullable=False, unique=True)
    vocab_sig = Column(String, nullable=False)

    reference_hash = Column(String, nullable=True)
    corpus_hash = Column(String, nullable=True)
    training_mode = Column(String, nullable=False)
    fold_seed = Column(Integer, nullable=True)

    encoder = Column(String, nullable=False)
    embedding_dim = Column(Integer, nullable=False)
    augment_floor = Column(Integer, nullable=True)

    artifact_path = Column(String, nullable=False)

    # 'building' | 'current' | 'stale' | 'archived'
    status = Column(String, nullable=False, default="building")

    # Per-fold variance + Gate A/B headline numbers.  SQLAlchemy JSON
    # maps to JSONB on Postgres/PGlite (where queryability is useful)
    # and to plain TEXT on SQLite; consumers pass + receive dicts.
    metrics = Column(JSON, nullable=True)

    summary = Column(Text, nullable=True)

    built_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())


class FSMRun(Base):
    """Classification pipeline run state."""

    __tablename__ = "fsm_runs"

    id = Column(String, primary_key=True, nullable=False)
    state = Column(String, nullable=False, default="IDLE")
    started_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now())
    config = Column(Text, nullable=True)       # JSON
    progress = Column(Text, nullable=True)     # JSON
    error = Column(Text, nullable=True)
    result_path = Column(Text, nullable=True)
    source_id = Column(String, nullable=True)  # FK → data_sources.id

    # Late-interaction cosine lineage: which enriched annotation collection
    # this run consumed.  NULL = legacy single-vector cosine path.
    taxonomy_collection = Column(String, nullable=True)


class ClassificationRun(Base):
    """Per-column classification result."""

    __tablename__ = "classification_runs"

    id = Column(String, primary_key=True, nullable=False)
    fsm_run_id = Column(String, nullable=True)
    table_name = Column(String, nullable=False)
    column_name = Column(String, nullable=False)
    predicted_code = Column(String, nullable=True)
    predicted_label = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    belief = Column(Float, nullable=True)
    plausibility = Column(Float, nullable=True)
    conflict = Column(Float, nullable=True)
    evidence_sources = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, server_default=func.now())


class Vocabulary(Base):
    """Cached controlled vocabulary metadata."""

    __tablename__ = "vocabularies"

    id = Column(String, primary_key=True, nullable=False)
    source = Column(String, nullable=False)
    loaded_at = Column(DateTime, server_default=func.now())
    category_count = Column(BigInteger, nullable=True)
    categories = Column(Text, nullable=True)  # JSON
