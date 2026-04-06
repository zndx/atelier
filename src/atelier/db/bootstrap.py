"""Bootstrap embedded PostgreSQL for CAI deployment.

Local dev: devenv services.postgres provides PostgreSQL 16 + pgvector.
CAI: pgserver (pip-installed embedded PG) auto-starts a PostgreSQL process.

This module detects the environment and returns the appropriate DB URI.
The bootstrap is only activated when no ATELIER_DB_URL is explicitly set
and we detect a CML environment (CDSW_APP_PORT is present).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_server = None


def ensure_database(data_dir: str = ".app/pgdata") -> str:
    """Start embedded PostgreSQL if needed, return SQLAlchemy connection URI.

    Uses pgserver to manage an embedded PostgreSQL process. The server
    persists data in *data_dir* and stays alive for the lifetime of the
    process. If another process is already using the same data directory,
    pgserver attaches to the existing server.

    Returns:
        SQLAlchemy-compatible connection URI
        (``postgresql+psycopg://...``).
    """
    import pgserver

    global _server

    if _server is None:
        data_path = Path(data_dir)
        data_path.mkdir(parents=True, exist_ok=True)
        log.info("Starting embedded PostgreSQL in %s", data_path)
        _server = pgserver.get_server(str(data_path))
        log.info("Embedded PostgreSQL ready: %s", _server.get_uri())

    uri = _server.get_uri()
    # pgserver returns postgresql:// — convert to SQLAlchemy psycopg format
    return uri.replace("postgresql://", "postgresql+psycopg://", 1)


def run_migrations(db_url: str, migrations_dir: str = "db/migrations") -> None:
    """Run dbmate migrations against the given database URL.

    Args:
        db_url: SQLAlchemy-style URI. The ``+psycopg`` driver suffix is
            stripped for dbmate compatibility.
        migrations_dir: Path to dbmate migration files.
    """
    # dbmate expects bare postgresql:// without the +psycopg driver
    dbmate_url = db_url.replace("+psycopg", "")
    log.info("Running dbmate migrations from %s", migrations_dir)
    result = subprocess.run(
        ["dbmate", "--url", dbmate_url, "--migrations-dir", migrations_dir, "up"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error("dbmate migration failed: %s", result.stderr)
        raise RuntimeError(f"dbmate migration failed: {result.stderr}")
    log.info("Migrations applied successfully")
