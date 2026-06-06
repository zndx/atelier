"""Database bootstrap — migrations and seed data.

Provides the complete bootstrap sequence for both ``devenv up`` (local dev)
and ``bin/start-app.sh`` (CAI deployment):

1. Apply SQL migrations (dbmate-compatible tracking)
2. Seed keystone agents (idempotent upserts)

Run as a module for one-liner process exec::

    python -m atelier.db.bootstrap
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


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

    engine = create_engine(db_url, connect_args={"connect_timeout": 10})
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
    """Split SQL text on semicolons, ignoring ``;`` inside comments and strings.

    Semicolons that appear inside ``--`` line comments, ``/* ... */`` block
    comments, or single-quoted string literals are NOT statement terminators.
    Previous naive ``sql.split(';')`` broke on comments like
    ``-- internal marker; only the display changes`` by turning the latter
    half into a bogus statement.
    """
    out: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""

        # Line comment: skip to newline
        if ch == "-" and nxt == "-":
            buf.append(ch)
            i += 1
            while i < n and sql[i] != "\n":
                buf.append(sql[i])
                i += 1
            continue

        # Block comment: skip to closing */
        if ch == "/" and nxt == "*":
            buf.append(ch); buf.append(nxt)
            i += 2
            while i < n and not (sql[i] == "*" and i + 1 < n and sql[i + 1] == "/"):
                buf.append(sql[i])
                i += 1
            if i < n:
                buf.append("*"); buf.append("/")
                i += 2
            continue

        # Single-quoted string: skip to closing ', handling '' escapes
        if ch == "'":
            buf.append(ch)
            i += 1
            while i < n:
                c = sql[i]
                buf.append(c)
                i += 1
                if c == "'":
                    if i < n and sql[i] == "'":
                        buf.append("'"); i += 1
                        continue
                    break
            continue

        if ch == ";":
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue

        buf.append(ch)
        i += 1

    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


# ── Keystone agents ──────────────────────────────────────────────
# Single source of truth for agent definitions. Both devenv and CAI
# use this constant via bootstrap(). Upsert semantics keep
# descriptions and tool_ids in sync after code changes.

KEYSTONE_AGENTS: list[tuple[str, str, str, str, str]] = [
    (
        "sampler",
        "Metadata Sampler",
        "Discovers tables and samples column metadata from production "
        "databases via CAI Data Platform connections",
        "sampler",
        '["sample-metadata", "discover-tables", "load-annotations"]',
    ),
    (
        "classifier",
        "Column Classifier",
        "Extracts 12 discrete features per column and runs cosine, "
        "CatBoost, and SVM classifiers with regex pattern detection",
        "classifier",
        '["extract-features", "run-classifiers", "detect-patterns", "classify-columns"]',
    ),
    (
        "evidence-fuser",
        "Evidence Fuser",
        "Converts 5 evidence sources into Dempster-Shafer mass functions, "
        "fuses via conjunctive combination, and diagnoses conflicts",
        "evidence_fuser",
        '["build-mass-functions", "apply-dempster-rule", "diagnose-conflicts"]',
    ),
    (
        "synth-generator",
        "Synthetic Generator",
        "Creates deterministic synthetic training tables with representative "
        "column data for classifier training",
        "synth_generator",
        '["generate-synth-tables", "load-annotations"]',
    ),
    (
        "viz-director",
        "Visualization Director",
        "Computes SAGE/SHAP feature explanations and prepares Atlas-compatible "
        "embedding projections",
        "visualization_director",
        '["compute-sage-importance", "generate-shap-explanations", "prepare-atlas-projection"]',
    ),
]


def seed_keystone_agents() -> None:
    """Upsert keystone agent definitions into the database.

    Idempotent — safe to call on every restart. Uses upsert semantics
    so descriptions and tool_ids stay in sync with the codebase.
    """
    from atelier.db.dao import AtelierDao

    dao = AtelierDao()
    for aid, name, desc, role, tools in KEYSTONE_AGENTS:
        dao.upsert_agent(aid, name, desc, role, tools)
    log.info("Upserted %d keystone agents", len(KEYSTONE_AGENTS))


def sync_orphaned_runs(cfg=None) -> None:
    """Reconcile DB state against on-disk run artifacts.

    Walks ``build/results/`` and registers missing ``fsm_runs``,
    ``ml_artifact_sets``, and ``datasets`` rows for every complete
    run whose DB writes failed during the pipeline (transient PGlite
    hiccup, FK ordering bug, etc.).

    Idempotent — runs already represented in the DB cost one read
    per row.  Safe to call on every restart.
    """
    from pathlib import Path
    from atelier.db.sync import sync_filesystem_to_db

    results_root = Path("build/results")
    if not results_root.is_dir():
        log.info("No build/results directory — sync skipped")
        return

    report = sync_filesystem_to_db(results_root, cfg=cfg)
    log.info(report.summary_line())
    for outcome in report.outcomes:
        if outcome.artifact_set == "registered" or outcome.dataset == "registered":
            log.info(
                "sync: %s — artifact_set=%s dataset=%s source_id=%s%s",
                outcome.run_id, outcome.artifact_set, outcome.dataset,
                outcome.source_id,
                f" ({outcome.note})" if outcome.note else "",
            )
        elif outcome.artifact_set == "error" or outcome.dataset == "error":
            log.warning(
                "sync: %s — artifact_set=%s dataset=%s%s",
                outcome.run_id, outcome.artifact_set, outcome.dataset,
                f" ({outcome.note})" if outcome.note else "",
            )


def bootstrap() -> None:
    """Run migrations, seed agents, and sync orphaned runs.

    Shared entry point for ``devenv up`` and ``bin/start-app.sh``.
    Loads config from HOCON to get the database URL.

    The sync step runs **synchronously** after migrations, while the
    DB is guaranteed reachable — this is more reliable than the
    gateway's async lifespan sync which can silently fail if PGlite
    is still warming up.
    """
    from atelier.config import load_config

    cfg = load_config()

    log.info("Running database migrations...")
    run_migrations(cfg.db_url)

    log.info("Seeding keystone agents...")
    seed_keystone_agents()

    log.info("Syncing orphaned runs...")
    try:
        sync_orphaned_runs(cfg=cfg)
    except Exception as exc:
        # Non-fatal — the gateway lifespan sync is a fallback.
        log.warning("Orphaned-run sync failed (will retry in gateway): %s", exc)

    log.info("Bootstrap complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    bootstrap()
