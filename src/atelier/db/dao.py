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
