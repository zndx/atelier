"""SQLAlchemy ORM models for Atelier state."""

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, Integer, String, Text
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
