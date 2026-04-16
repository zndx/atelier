"""CAI Data Platform connection wrapper.

``cml.data_v1`` ships only with CAI runtimes and is NOT pip-installable.
All imports are deferred so this module loads fine in devenv / CI, where
:func:`list_connections` still works (it reads HOCON) and
:func:`test_connection` returns a soft failure.

Follows the project convention: take :class:`AtelierConfig` by argument,
never read ``os.environ`` directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.config import AtelierConfig


def list_connections(cfg: "AtelierConfig") -> list[str]:
    """Return configured connection names (HOCON source of truth)."""
    return cfg.cml_data_connection_names


def _cml_available() -> bool:
    try:
        import cml.data_v1  # noqa: F401
        return True
    except ImportError:
        return False


def test_connection(cfg: "AtelierConfig", name: str) -> dict:
    """Run ``show databases`` against the named CAI connection.

    Returns a JSON-serializable dict. On success::

        {
          "ok": True,
          "connection": "prod-impala",
          "query": "show databases",
          "row_count": 4,
          "columns": ["database_name"],
          "rows": [["airlines"], ["default"], ...],
          "latency_ms": 1234,
        }

    On failure::

        {"ok": False, "connection": name, "error": "..."}
    """
    import time

    if name not in cfg.cml_data_connection_names:
        return {
            "ok": False,
            "connection": name,
            "error": (
                f"Connection '{name}' not in configured list. "
                "Set ATELIER_DATA_CONNECTIONS."
            ),
        }

    if not _cml_available():
        return {
            "ok": False,
            "connection": name,
            "error": "cml.data_v1 not available — only present on CAI runtimes.",
        }

    try:
        import cml.data_v1 as cmldata  # type: ignore[import-not-found]

        t0 = time.monotonic()
        conn = cmldata.get_connection(name)
        df = conn.get_pandas_dataframe("show databases")
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        columns = [str(c) for c in df.columns]
        rows = [
            [
                None if _is_nan(v) else (
                    v if isinstance(v, (str, int, float, bool)) else str(v)
                )
                for v in row
            ]
            for row in df.head(50).itertuples(index=False, name=None)
        ]
        return {
            "ok": True,
            "connection": name,
            "query": "show databases",
            "row_count": int(len(df)),
            "columns": columns,
            "rows": rows,
            "latency_ms": elapsed_ms,
        }
    except Exception as exc:
        return {"ok": False, "connection": name, "error": str(exc)}


def _is_nan(v) -> bool:
    try:
        import math
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


# ── Hive data source discovery ───────────────────────────────────

# Minimum columns that identify a valid annotations table.
_LEGACY_REQUIRED = {"id", "ontology", "annotation"}
_UNIVERSAL_REQUIRED = {"code", "label"}


def _normalize_key(k: str) -> str:
    """Strip Hive table-qualification, BOM, quotes; lowercase + underscores."""
    s = str(k)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    s = s.strip().strip("'\"\ufeff").lower().replace(" ", "_")
    return s


def _list_databases(conn) -> list[str]:
    """Return database names from a cml.data_v1 connection."""
    df = conn.get_pandas_dataframe("SHOW DATABASES")
    return [str(v) for v in df.iloc[:, 0].tolist()]


def _has_annotations_table(conn, database: str) -> bool:
    """Check whether *database* contains an ``annotations`` table."""
    df = conn.get_pandas_dataframe(f"SHOW TABLES IN {database}")
    tables = {str(v).lower() for v in df.iloc[:, 0].tolist()}
    return "annotations" in tables


def _validate_annotations_schema(conn, database: str) -> dict | None:
    """Fetch one row from ``{database}.annotations`` and validate schema.

    Returns a metadata dict on success, ``None`` if the schema doesn't match.
    """
    df = conn.get_pandas_dataframe(
        f"SELECT * FROM {database}.annotations LIMIT 1"
    )
    if df.empty:
        return None

    keys = {_normalize_key(c) for c in df.columns}
    if _LEGACY_REQUIRED <= keys:
        fmt = "legacy"
    elif _UNIVERSAL_REQUIRED <= keys:
        fmt = "universal"
    else:
        return None

    # Get row count for metadata
    count_df = conn.get_pandas_dataframe(
        f"SELECT count(*) AS cnt FROM {database}.annotations"
    )
    count = int(count_df.iloc[0, 0]) if not count_df.empty else 0

    return {"schema_format": fmt, "annotation_count": count}


def _table_count(conn, database: str) -> int:
    """Count tables in a database."""
    try:
        df = conn.get_pandas_dataframe(f"SHOW TABLES IN {database}")
        return len(df)
    except Exception:
        return 0


def refresh_connection(cfg: "AtelierConfig", name: str) -> dict:
    """Probe a connection: list databases with table counts and annotations.

    Returns a dict with per-database metadata including whether an
    annotations table exists, its row count, and schema format.
    """
    import logging
    import time
    log = logging.getLogger(__name__)

    if not _cml_available():
        return {"ok": False, "connection": name, "error": "cml.data_v1 not available"}

    import cml.data_v1 as cmldata  # type: ignore[import-not-found]

    t0 = time.monotonic()
    try:
        conn = cmldata.get_connection(name)
        databases = _list_databases(conn)
    except Exception as exc:
        return {"ok": False, "connection": name, "error": str(exc)}

    db_infos = []
    for db in databases:
        info: dict = {"name": db, "table_count": 0, "has_annotations": False,
                      "annotation_count": 0, "schema_format": None}
        try:
            info["table_count"] = _table_count(conn, db)
            if _has_annotations_table(conn, db):
                info["has_annotations"] = True
                meta = _validate_annotations_schema(conn, db)
                if meta:
                    info["annotation_count"] = meta["annotation_count"]
                    info["schema_format"] = meta["schema_format"]
        except Exception as exc:
            log.debug("refresh_connection: error probing %s/%s: %s", name, db, exc)
        db_infos.append(info)

    latency = int((time.monotonic() - t0) * 1000)
    return {"ok": True, "connection": name, "latency_ms": latency, "databases": db_infos}


def discover_hive_sources(cfg: "AtelierConfig") -> list[dict]:
    """Probe configured connections for annotations tables.

    For each connection × database, checks for an ``annotations`` table
    with the expected schema (legacy or universal format).  Returns a list
    of discovery results suitable for data source registration.

    Gracefully skips connections/databases that fail.
    Only runs on CAI runtimes (``cml.data_v1`` required).
    """
    import logging

    log = logging.getLogger(__name__)

    if not _cml_available():
        return []

    import cml.data_v1 as cmldata  # type: ignore[import-not-found]

    results: list[dict] = []
    for conn_name in cfg.cml_data_connection_names:
        try:
            conn = cmldata.get_connection(conn_name)
            databases = _list_databases(conn)
        except Exception as exc:
            log.warning("Hive discovery: connection '%s' failed: %s", conn_name, exc)
            continue

        for db in databases:
            try:
                if not _has_annotations_table(conn, db):
                    continue
                meta = _validate_annotations_schema(conn, db)
                if meta is None:
                    log.debug(
                        "Hive discovery: %s/%s has annotations table but wrong schema",
                        conn_name, db,
                    )
                    continue
                source_id = f"{conn_name}/{db}"
                results.append({
                    "connection": conn_name,
                    "database": db,
                    "source_id": source_id,
                    "display_name": f"Hive: {conn_name}/{db}",
                    "annotation_count": meta["annotation_count"],
                    "schema_format": meta["schema_format"],
                })
                log.info(
                    "Hive discovery: found %d annotations in %s/%s (%s format)",
                    meta["annotation_count"], conn_name, db, meta["schema_format"],
                )
            except Exception as exc:
                log.warning(
                    "Hive discovery: error probing %s/%s: %s", conn_name, db, exc
                )
    return results
