"""Data access object for Atelier state persistence.

Follows the Fine Tuning Studio DAO pattern: SQLAlchemy engine
with context-managed sessions.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from atelier.db.model import Base


class AtelierDao:
    """Database access for Atelier application state."""

    def __init__(
        self,
        engine_url: str | None = None,
        echo: bool = False,
        engine_args: dict | None = None,
    ):
        if engine_url is None:
            engine_url = "sqlite+pysqlite:///.app/state.db"

        self.engine = create_engine(
            engine_url, echo=echo, **(engine_args or {}),
        )
        self.Session = sessionmaker(
            bind=self.engine, autoflush=True, autocommit=False,
        )
        Base.metadata.create_all(self.engine)

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
