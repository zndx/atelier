"""Bootstrap embedded PostgreSQL for CAI deployment.

Local dev: devenv services.postgres provides PostgreSQL 16 + pgvector.
CAI: pgserver (pip-installed embedded PG) auto-starts a PostgreSQL process.

This module detects the environment and returns the appropriate DB URI.
The bootstrap is only activated when no ATELIER_DB_URL is explicitly set
and we detect a CML environment (CDSW_APP_PORT is present).
"""

from __future__ import annotations

import logging
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
    """Apply SQL migrations from the migrations directory.

    Reads ``-- migrate:up`` blocks from migration files and executes them
    in filename order. Tracks applied migrations in a ``schema_migrations``
    table (compatible with dbmate's tracking table).

    This avoids requiring dbmate CLI in the CML environment while staying
    compatible with dbmate for local development.

    Args:
        db_url: SQLAlchemy-style connection URI.
        migrations_dir: Path to directory containing ``.sql`` migration files.
    """
    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    migrations_path = Path(migrations_dir)

    if not migrations_path.exists():
        log.warning("Migrations directory %s not found, skipping", migrations_dir)
        return

    with engine.begin() as conn:
        # Create tracking table (dbmate-compatible)
        conn.execute(text(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  version VARCHAR(128) PRIMARY KEY"
            ")"
        ))

        # Get already-applied migrations
        result = conn.execute(text("SELECT version FROM schema_migrations"))
        applied = {row[0] for row in result}

        # Apply pending migrations in order
        for sql_file in sorted(migrations_path.glob("*.sql")):
            version = sql_file.stem
            if version in applied:
                continue

            log.info("Applying migration: %s", version)
            content = sql_file.read_text()

            # Extract the -- migrate:up section and execute each statement
            up_sql = _extract_up_block(content)
            if up_sql:
                for stmt in _split_statements(up_sql):
                    conn.execute(text(stmt))
                conn.execute(
                    text("INSERT INTO schema_migrations (version) VALUES (:v)"),
                    {"v": version},
                )
                log.info("Applied: %s", version)

    engine.dispose()
    log.info("Migrations complete")


def _extract_up_block(content: str) -> str | None:
    """Extract SQL between ``-- migrate:up`` and ``-- migrate:down``."""
    lines = content.splitlines()
    in_up = False
    up_lines: list[str] = []

    for line in lines:
        stripped = line.strip().lower()
        if stripped == "-- migrate:up":
            in_up = True
            continue
        elif stripped == "-- migrate:down":
            break
        elif in_up:
            up_lines.append(line)

    sql = "\n".join(up_lines).strip()
    return sql if sql else None


def _split_statements(sql: str) -> list[str]:
    """Split SQL text on semicolons, returning non-empty statements."""
    return [s.strip() for s in sql.split(";") if s.strip()]
