#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Load synth CSV tables into a Hive 'synth' database via CAI Data Platform.

Reads every CSV from data/synth/tables/ and materializes it as a Hive
table inside a database called ``synth``.  All columns are STRING — the
classifier treats everything as text anyway.

Requires a CAI runtime with ``cml.data_v1`` and a configured data
connection (ATELIER_DATA_CONNECTIONS / ATELIER_CLASSIFY_CONNECTION).

Usage (on CAI):
    python scripts/load_synth_to_hive.py                        # auto-detect connection
    python scripts/load_synth_to_hive.py --connection prod-hive  # explicit
    python scripts/load_synth_to_hive.py --dry-run               # print DDL only
    python scripts/load_synth_to_hive.py --drop                  # recreate tables
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_DIR = PROJECT_ROOT / "data" / "synth"
TABLES_DIR = SYNTH_DIR / "tables"

DATABASE = "synth"
BATCH_SIZE = 50  # rows per INSERT statement (Hive has query-size limits)

log = logging.getLogger("load_synth_to_hive")


# ── Helpers ──────────────────────────────────────────────────────────

def _escape(value: str) -> str:
    """Escape a string value for HiveQL single-quoted literal."""
    if not value:
        return "NULL"
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _backtick(name: str) -> str:
    """Backtick-quote a column/table name for Hive."""
    return "`" + name.replace("`", "``") + "`"


def _execute(conn, sql: str, *, dry_run: bool = False) -> None:
    """Execute a SQL statement via the CAI data connection.

    Uses a raw cursor instead of ``get_pandas_dataframe`` so that DDL
    and DML statements (CREATE, DROP, INSERT) work correctly — they
    return no result set, which ``as_pandas`` cannot handle.
    """
    if dry_run:
        # Truncate long INSERT statements for readability
        display = sql if len(sql) < 400 else sql[:400] + " ..."
        print(display)
        return
    base = conn.get_base_connection()
    try:
        cursor = base.cursor()
        try:
            cursor.execute(sql)
        finally:
            cursor.close()
    finally:
        base.close()


# ── Core ─────────────────────────────────────────────────────────────

def _dedup_columns(header: list[str]) -> list[str]:
    """Make column names unique for Hive (case-insensitive).

    Hive treats column names as case-insensitive, so 'channel' and
    'CHANNEL' collide.  Lowercase everything and append _N suffixes
    when collisions occur, making sure the suffixed name itself
    doesn't collide with any other column.
    """
    # First pass: collect all lowered names so we know the full set
    all_lower = [col.lower() for col in header]
    taken: set[str] = set()
    result: list[str] = []
    for key in all_lower:
        if key not in taken:
            taken.add(key)
            result.append(key)
        else:
            # Find a suffix that doesn't collide
            n = 2
            while f"{key}_{n}" in taken:
                n += 1
            new_name = f"{key}_{n}"
            taken.add(new_name)
            result.append(new_name)
    return result


def copy_annotations_table(
    conn,
    *,
    drop: bool = False,
    dry_run: bool = False,
) -> None:
    """Copy default.annotations directly into synth.annotations via CTAS."""
    fq_table = f"{DATABASE}.annotations"
    if drop:
        _execute(conn, f"DROP TABLE IF EXISTS {fq_table}", dry_run=dry_run)
    _execute(
        conn,
        f"CREATE TABLE IF NOT EXISTS {fq_table} AS SELECT * FROM default.annotations",
        dry_run=dry_run,
    )


def load_csv_to_hive(
    conn,
    csv_path: Path,
    *,
    drop: bool = False,
    dry_run: bool = False,
) -> int:
    """Load a single CSV into synth.<table_name>. Returns row count."""
    table_name = csv_path.stem

    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            log.warning("Skipping empty CSV: %s", csv_path.name)
            return 0
        rows = list(reader)

    # Hive column names are case-insensitive — deduplicate
    header = _dedup_columns(header)

    fq_table = f"{DATABASE}.{_backtick(table_name)}"

    # DROP if requested
    if drop:
        _execute(conn, f"DROP TABLE IF EXISTS {fq_table}", dry_run=dry_run)

    # CREATE TABLE — all STRING columns
    col_defs = ", ".join(f"{_backtick(c)} STRING" for c in header)
    _execute(
        conn,
        f"CREATE TABLE IF NOT EXISTS {fq_table} ({col_defs}) STORED AS PARQUET",
        dry_run=dry_run,
    )

    # INSERT data in batches
    if not rows:
        return 0

    inserted = 0
    for batch_start in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        values_clauses = []
        for row in batch:
            # Pad/truncate row to match header length
            padded = row + [""] * (len(header) - len(row))
            vals = ", ".join(_escape(padded[i]) for i in range(len(header)))
            values_clauses.append(f"({vals})")

        insert_sql = (
            f"INSERT INTO {fq_table} VALUES\n"
            + ",\n".join(values_clauses)
        )
        _execute(conn, insert_sql, dry_run=dry_run)
        inserted += len(batch)

    return inserted


def get_connection(connection_name: str | None = None):
    """Obtain a cml.data_v1 connection handle.

    Falls back to ATELIER_CLASSIFY_CONNECTION → first entry in
    ATELIER_DATA_CONNECTIONS → error.
    """
    try:
        import cml.data_v1 as cmldata  # type: ignore[import-not-found]
    except ImportError:
        print(
            "ERROR: cml.data_v1 not available.\n"
            "This script must run on a CAI runtime with a configured data connection.",
            file=sys.stderr,
        )
        sys.exit(1)

    if connection_name:
        return cmldata.get_connection(connection_name)

    # Auto-detect from config env vars
    import os

    for env_var in ("ATELIER_CLASSIFY_CONNECTION", "ATELIER_DATA_CONNECTIONS"):
        val = os.environ.get(env_var, "").strip()
        if val:
            name = val.split(",")[0].strip()
            if name:
                log.info("Using connection '%s' from %s", name, env_var)
                return cmldata.get_connection(name)

    print(
        "ERROR: No connection name provided and none found in env.\n"
        "Set ATELIER_DATA_CONNECTIONS or pass --connection <name>.",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load synth CSV tables into Hive 'synth' database."
    )
    parser.add_argument(
        "--connection", "-c",
        help="CAI data connection name (auto-detected if omitted).",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Print SQL statements without executing.",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop and recreate tables if they already exist.",
    )
    parser.add_argument(
        "--tables", "-t",
        nargs="*",
        help="Load only these tables (stem names, e.g. 'accounts employees').",
    )
    parser.add_argument(
        "--synth-dir",
        type=Path,
        default=SYNTH_DIR,
        help=f"Path to synth data directory (default: {SYNTH_DIR}).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    tables_dir = args.synth_dir / "tables"
    if not tables_dir.is_dir():
        print(f"ERROR: Tables directory not found: {tables_dir}", file=sys.stderr)
        sys.exit(1)

    csv_files = sorted(tables_dir.glob("*.csv"))
    if args.tables:
        allowed = set(args.tables)
        csv_files = [f for f in csv_files if f.stem in allowed]

    if not csv_files:
        print("No CSV files to load.", file=sys.stderr)
        sys.exit(1)

    log.info("Found %d synth tables to load into Hive '%s' database", len(csv_files), DATABASE)

    # Connect (or skip for dry-run)
    conn = None
    if not args.dry_run:
        conn = get_connection(args.connection)

    # Create database
    create_db_sql = f"CREATE DATABASE IF NOT EXISTS {DATABASE}"
    _execute(conn, create_db_sql, dry_run=args.dry_run)
    log.info("Ensured database '%s' exists", DATABASE)

    # Copy default.annotations → synth.annotations
    try:
        copy_annotations_table(conn, drop=args.drop, dry_run=args.dry_run)
        log.info("Copied default.annotations → %s.annotations", DATABASE)
    except Exception as exc:
        log.warning("Failed to copy default.annotations: %s", exc)

    # Load each table
    t0 = time.monotonic()
    total_rows = 0
    total_cols = 0
    errors: list[str] = []

    for i, csv_path in enumerate(csv_files, 1):
        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader, [])
                n_cols = len(header)

            row_count = load_csv_to_hive(
                conn, csv_path, drop=args.drop, dry_run=args.dry_run,
            )
            total_rows += row_count
            total_cols += n_cols
            log.info(
                "[%3d/%d] %-30s %4d cols  %4d rows",
                i, len(csv_files), csv_path.stem, n_cols, row_count,
            )
        except Exception as exc:
            log.error("[%3d/%d] %-30s FAILED: %s", i, len(csv_files), csv_path.stem, exc)
            errors.append(f"{csv_path.stem}: {exc}")

    elapsed = time.monotonic() - t0
    log.info(
        "Done: %d tables, %d columns, %d rows in %.1fs",
        len(csv_files) - len(errors), total_cols, total_rows, elapsed,
    )
    if errors:
        log.warning("%d tables failed:", len(errors))
        for e in errors:
            log.warning("  %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
