"""Data access object for Atelier state persistence.

Follows the Fine Tuning Studio DAO pattern: SQLAlchemy engine
with context-managed sessions.

Schema is managed by dbmate (db/migrations/). Do NOT use
Base.metadata.create_all() — run ``just migrate`` instead.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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

        self.engine = create_engine(
            engine_url, echo=echo, **(engine_args or {}),
        )
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

    def list_datasets(self) -> list[dict]:
        """Return all registered datasets as dicts."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            rows = session.query(Dataset).all()
            return [
                {"id": r.id, "name": r.name, "parquet_path": r.parquet_path,
                 "description": r.description, "row_count": r.row_count}
                for r in rows
            ]

    def get_dataset(self, dataset_id: str) -> dict | None:
        """Return a dataset by ID as dict, or None."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            r = session.query(Dataset).filter_by(id=dataset_id).first()
            if r is None:
                return None
            return {"id": r.id, "name": r.name, "parquet_path": r.parquet_path,
                    "description": r.description, "row_count": r.row_count}

    def upsert_dataset(self, dataset_id: str, name: str,
                       parquet_path: str, description: str = "",
                       row_count: int = 0):
        """Insert or update a dataset record."""
        from atelier.db.model import Dataset
        with self.get_session() as session:
            ds = session.query(Dataset).filter_by(id=dataset_id).first()
            if ds is None:
                ds = Dataset(
                    id=dataset_id, name=name, parquet_path=parquet_path,
                    description=description, row_count=row_count,
                )
                session.add(ds)
            else:
                ds.name = name
                ds.parquet_path = parquet_path
                ds.description = description
                ds.row_count = row_count
