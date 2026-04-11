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
