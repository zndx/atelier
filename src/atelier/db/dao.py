"""Data access object for Atelier state persistence.

Follows the Fine Tuning Studio DAO pattern: SQLAlchemy engine
with context-managed sessions.

Schema is managed by dbmate (db/migrations/). Do NOT use
Base.metadata.create_all() — run ``just migrate`` instead.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Default engine settings for PGlite resilience. Callers can override
# individual keys via engine_args (e.g. tests may set pool_size=1).
_DEFAULT_ENGINE_ARGS = {
    "pool_pre_ping": True,
    "pool_recycle": 300,
    "pool_size": 3,
    "max_overflow": 2,
    "connect_args": {"connect_timeout": 10},
}


class AtelierDao:
    """Database access for Atelier application state."""

    def __init__(
        self,
        engine_url: str | None = None,
        echo: bool = False,
        engine_args: dict | None = None,
    ):
        if engine_url is None:
            from atelier.config import load_config
            engine_url = load_config().db_url

        merged = {**_DEFAULT_ENGINE_ARGS, **(engine_args or {})}
        self.engine = create_engine(engine_url, echo=echo, **merged)
        self.Session = sessionmaker(
            bind=self.engine, autoflush=True, autocommit=False,
        )

    @contextmanager
    def get_session(self):
        """Context manager for a database session with auto commit/rollback."""
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    # ── Data source operations ────────────────────────────────────

    def list_data_sources(self, include_archived: bool = False) -> list[dict]:
        """Return data sources as dicts. Excludes archived by default."""
        from atelier.db.model import DataSource
        with self.get_session() as session:
            q = session.query(DataSource).order_by(DataSource.created_at)
            if not include_archived:
                q = q.filter_by(is_archived=False)
            rows = q.all()
            return [self._source_to_dict(r) for r in rows]

    def get_data_source(self, source_id: str) -> dict | None:
        """Return a data source by ID as dict, or None."""
        from atelier.db.model import DataSource
        with self.get_session() as session:
            r = session.query(DataSource).filter_by(id=source_id).first()
            if r is None:
                return None
            return self._source_to_dict(r)

    def get_or_create_data_source(self, source_id: str, source_type: str,
                                  display_name: str, source_uri: str = "",
                                  vocabulary_mode: str = "universal",
                                  vocab_uri: str = "",
                                  metadata: str | None = None) -> dict:
        """Get existing or create a new data source. Returns the source dict."""
        from atelier.db.model import DataSource
        with self.get_session() as session:
            r = session.query(DataSource).filter_by(id=source_id).first()
            if r is None:
                r = DataSource(
                    id=source_id, source_type=source_type,
                    source_uri=source_uri, display_name=display_name,
                    vocabulary_mode=vocabulary_mode, vocab_uri=vocab_uri,
                    source_metadata=metadata,
                )
                session.add(r)
                session.flush()
            return self._source_to_dict(r)

    def force_upsert_data_source(self, source_id: str, source_type: str,
                                 display_name: str, source_uri: str = "",
                                 vocabulary_mode: str = "universal",
                                 vocab_uri: str = "") -> dict:
        """Upsert canonical fields on a data source row.

        Unlike :meth:`get_or_create_data_source`, this refreshes
        ``source_type`` / ``source_uri`` / ``display_name`` /
        ``vocabulary_mode`` / ``vocab_uri`` on every call.  Used by
        startup seeders that want the DB to converge to the runtime
        mount state — e.g. flipping legacy ``source_type="sample"``
        rows to ``"filesystem"`` and stamping the URI with its scheme
        prefix.  ``source_metadata`` is untouched (the seeder maintains
        it via :meth:`update_data_source_metadata`).
        """
        from atelier.db.model import DataSource
        with self.get_session() as session:
            r = session.query(DataSource).filter_by(id=source_id).first()
            if r is None:
                r = DataSource(
                    id=source_id, source_type=source_type,
                    source_uri=source_uri, display_name=display_name,
                    vocabulary_mode=vocabulary_mode, vocab_uri=vocab_uri,
                )
                session.add(r)
            else:
                r.source_type = source_type
                r.source_uri = source_uri
                r.display_name = display_name
                r.vocabulary_mode = vocabulary_mode
                r.vocab_uri = vocab_uri
            session.flush()
            return self._source_to_dict(r)

    @staticmethod
    def _source_to_dict(r) -> dict:
        return {
            "id": r.id, "source_type": r.source_type,
            "source_uri": r.source_uri, "display_name": r.display_name,
            "vocabulary_mode": r.vocabulary_mode,
            "vocab_uri": getattr(r, "vocab_uri", "") or "",
            "created_at": str(r.created_at or ""),
            "metadata": r.source_metadata,
            "is_archived": r.is_archived,
        }

    def update_data_source_metadata(self, source_id: str, metadata: str) -> None:
        """Update the metadata JSON on a data source."""
        from atelier.db.model import DataSource
        with self.get_session() as session:
            r = session.query(DataSource).filter_by(id=source_id).first()
            if r is not None:
                r.source_metadata = metadata

    def update_data_source(self, source_id: str, **fields) -> dict | None:
        """Update mutable fields on a data source. Returns updated dict or None."""
        from atelier.db.model import DataSource
        _MUTABLE = {"vocab_uri", "display_name", "vocabulary_mode", "source_metadata"}
        with self.get_session() as session:
            r = session.query(DataSource).filter_by(id=source_id).first()
            if r is None:
                return None
            for k, v in fields.items():
                if k in _MUTABLE and v is not None:
                    setattr(r, k, v)
            session.flush()
            return self._source_to_dict(r)

    # ── Dataset operations (version-aware) ─────────────────────

    def list_datasets(self, source_id: str | None = None,
                      include_archived: bool = False) -> list[dict]:
        """Return datasets. Excludes archived by default."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            q = session.query(Dataset)
            if source_id is not None:
                q = q.filter_by(source_id=source_id)
            if not include_archived:
                q = q.filter_by(is_archived=False)
            rows = q.order_by(Dataset.created_at.desc()).all()
            return [self._dataset_to_dict(r) for r in rows]

    def get_dataset(self, dataset_id: str) -> dict | None:
        """Return a dataset by ID as dict, or None."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            r = session.query(Dataset).filter_by(id=dataset_id).first()
            if r is None:
                return None
            return self._dataset_to_dict(r)

    def upsert_dataset(self, dataset_id: str, name: str,
                       parquet_path: str, description: str = "",
                       row_count: int = 0, source_id: str | None = None,
                       version_number: int = 1, is_active: bool = True,
                       summary: str | None = None,
                       fsm_run_id: str | None = None):
        """Insert or update a dataset record."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                ds = Dataset(
                    id=dataset_id, name=name, parquet_path=parquet_path,
                    description=description, row_count=row_count,
                    source_id=source_id, version_number=version_number,
                    is_active=is_active, summary=summary,
                    fsm_run_id=fsm_run_id,
                )
                session.add(ds)
            else:
                ds.name = name
                ds.parquet_path = parquet_path
                ds.description = description
                ds.row_count = row_count
                if source_id is not None:
                    ds.source_id = source_id
                ds.version_number = version_number
                ds.is_active = is_active
                ds.summary = summary
                ds.fsm_run_id = fsm_run_id

    def list_dataset_versions(self, source_id: str,
                              include_archived: bool = False) -> list[dict]:
        """Return dataset versions for a source, newest first."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            q = (session.query(Dataset)
                 .filter_by(source_id=source_id))
            if not include_archived:
                q = q.filter_by(is_archived=False)
            rows = q.order_by(Dataset.version_number.desc()).all()
            return [self._dataset_to_dict(r) for r in rows]

    def next_version_number(self, source_id: str) -> int:
        """Return the next version number for a source."""
        from sqlalchemy import func as sa_func
        from atelier.db.model import Dataset
        with self.get_session() as session:
            max_v = (session.query(sa_func.max(Dataset.version_number))
                     .filter_by(source_id=source_id)
                     .scalar())
            return (max_v or 0) + 1

    def get_active_version(self, source_id: str) -> dict | None:
        """Return the active dataset version for a source, or None."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            r = (session.query(Dataset)
                 .filter_by(source_id=source_id, is_active=True)
                 .order_by(Dataset.version_number.desc())
                 .first())
            if r is None:
                return None
            return self._dataset_to_dict(r)

    def set_active_version(self, source_id: str, dataset_id: str):
        """Set one version as active, deactivating all others for the source."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            session.query(Dataset).filter_by(
                source_id=source_id
            ).update({"is_active": False})
            session.query(Dataset).filter_by(
                id=dataset_id
            ).update({"is_active": True})

    @staticmethod
    def _dataset_to_dict(r) -> dict:
        return {
            "id": r.id, "name": r.name, "parquet_path": r.parquet_path,
            "description": r.description, "row_count": r.row_count,
            "source_id": r.source_id, "version_number": r.version_number,
            "is_active": r.is_active, "summary": r.summary,
            "fsm_run_id": r.fsm_run_id,
            "created_at": str(r.created_at or ""),
            "is_archived": r.is_archived,
        }

    # ── Archive operations ─────────────────────────────────────────

    def archive_data_source(self, source_id: str) -> bool:
        """Archive a data source and all its datasets. Returns True if found."""
        from atelier.db.model import DataSource, Dataset
        with self.get_session() as session:
            source = session.query(DataSource).filter_by(id=source_id).first()
            if source is None:
                return False
            source.is_archived = True
            session.query(Dataset).filter_by(source_id=source_id).update(
                {"is_archived": True}
            )
            return True

    def unarchive_data_source(self, source_id: str) -> bool:
        """Unarchive a data source and all its datasets. Returns True if found."""
        from atelier.db.model import DataSource, Dataset
        with self.get_session() as session:
            source = session.query(DataSource).filter_by(id=source_id).first()
            if source is None:
                return False
            source.is_archived = False
            session.query(Dataset).filter_by(source_id=source_id).update(
                {"is_archived": False}
            )
            return True

    def archive_dataset(self, dataset_id: str) -> bool:
        """Archive a single dataset. Returns True if found."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            count = session.query(Dataset).filter_by(id=dataset_id).update(
                {"is_archived": True}
            )
            return count > 0

    def unarchive_dataset(self, dataset_id: str) -> bool:
        """Unarchive a single dataset. Returns True if found."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            count = session.query(Dataset).filter_by(id=dataset_id).update(
                {"is_archived": False}
            )
            return count > 0

    # ── Agent operations ──────────────────────────────────────────

    def list_agents(self) -> list[dict]:
        """Return all registered agents as dicts."""
        from atelier.db.model import Agent
        with self.get_session() as session:
            rows = session.query(Agent).all()
            return [
                {"id": r.id, "name": r.name, "description": r.description,
                 "role": r.role, "tool_ids": r.tool_ids}
                for r in rows
            ]

    def get_agent(self, agent_id: str) -> dict | None:
        """Return an agent by ID as dict, or None."""
        from atelier.db.model import Agent
        with self.get_session() as session:
            r = session.query(Agent).filter_by(id=agent_id).first()
            if r is None:
                return None
            return {"id": r.id, "name": r.name, "description": r.description,
                    "role": r.role, "tool_ids": r.tool_ids}

    def upsert_agent(self, agent_id: str, name: str, description: str = "",
                     role: str = "", tool_ids: str = "[]"):
        """Insert or update an agent record."""
        from atelier.db.model import Agent
        with self.get_session() as session:
            agent = session.query(Agent).filter_by(id=agent_id).first()
            if agent is None:
                agent = Agent(
                    id=agent_id, name=name, description=description,
                    role=role, tool_ids=tool_ids,
                )
                session.add(agent)
            else:
                agent.name = name
                agent.description = description
                agent.role = role
                agent.tool_ids = tool_ids

    # ── FSM operations ───────────────────────────────────────────

    def upsert_fsm_run(self, run_id: str, state: str, started_at: str,
                       updated_at: str, config: str = "", progress: str = "",
                       error: str | None = None, result_path: str | None = None,
                       source_id: str | None = None):
        """Insert or update an FSM run record."""
        from atelier.db.model import FSMRun
        with self.get_session() as session:
            run = session.query(FSMRun).filter_by(id=run_id).first()
            if run is None:
                run = FSMRun(
                    id=run_id, state=state, config=config,
                    progress=progress, error=error, result_path=result_path,
                    source_id=source_id,
                )
                session.add(run)
            else:
                run.state = state
                run.progress = progress
                run.error = error
                run.result_path = result_path
                if source_id is not None:
                    run.source_id = source_id

    def get_fsm_run(self, run_id: str) -> dict | None:
        """Return an FSM run by ID as dict, or None."""
        from atelier.db.model import FSMRun
        with self.get_session() as session:
            r = session.query(FSMRun).filter_by(id=run_id).first()
            if r is None:
                return None
            return {"id": r.id, "state": r.state,
                    "started_at": str(r.started_at or ""),
                    "updated_at": str(r.updated_at or ""),
                    "config": r.config, "progress": r.progress,
                    "error": r.error, "result_path": r.result_path,
                    "source_id": r.source_id}

    def list_fsm_runs(self) -> list[dict]:
        """Return all FSM runs as dicts."""
        from atelier.db.model import FSMRun
        with self.get_session() as session:
            rows = session.query(FSMRun).order_by(FSMRun.started_at.desc()).all()
            return [
                {"id": r.id, "state": r.state,
                 "started_at": str(r.started_at or ""),
                 "updated_at": str(r.updated_at or ""),
                 "config": r.config, "progress": r.progress,
                 "error": r.error, "result_path": r.result_path,
                 "source_id": r.source_id}
                for r in rows
            ]
