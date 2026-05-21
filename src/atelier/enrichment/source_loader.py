# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Source-taxonomy loaders for the enrichment pipeline.

The enrichment loop accepts a list of dicts with at least ``code``,
``label``, ``mnemonic``, ``description``, ``parent_code``.  Two loader
paths are supported:

- **JSON file** — pre-exported snapshot (offline, dev, CI).
- **CAI Data connection** (``cml.data_v1``) — live read against the
  active Hive/Impala annotations table on a CAI runtime.

The vocabulary identity is deliberately runtime-selected: the
operator names the ``(connection, database, table)`` triple per run
and the loader treats it as opaque.  No vocabulary content is
inferred from connection or database names — the loader normalizes
columns and returns rows as the architecture sees them, regardless
of which vocabulary happens to be loaded.

The two annotations schemas (legacy ``id/ontology/annotation`` and
universal ``code/label/...``) recognized by
``atelier.data.connections._validate_annotations_schema`` are both
accepted here.  Universal-format rows pass through; legacy-format
rows are renamed to the universal field names so downstream code
sees a single shape.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# Normalize column names: strip qualification, BOM, quotes; lowercase,
# replace spaces with underscores.  Mirrors
# atelier.data.connections._normalize_key for consistency.
def _normalize(key: str) -> str:
    s = str(key)
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return s.strip().strip("'\"﻿").lower().replace(" ", "_")


# Universal-schema field set, the shape the enrichment loop expects.
_UNIVERSAL_FIELDS = (
    "code", "label", "abbrev", "description",
    "common_names", "notation", "taxonomy", "parent_code",
)


# Legacy schema rename map — translates the on-disk Hive shape
# (id/ontology/annotation/definition) to the universal shape the
# enrichment loop expects (code/label/abbrev/description).  Empirical
# semantics verified against the cached export
# `build/data/annotations/hive-poc__default.json`:
#   - ``id``          carries the dotted code (e.g., "1.1.1.1.1.1.1") → code
#   - ``ontology``    carries the human label ("Payment Card Number") → label
#   - ``annotation``  carries the short mnemonic ("PAN")              → abbrev
#   - ``definition``  carries the descriptive prose                   → description
# ``parent_code`` is NOT present in the legacy schema; it is derived
# from the dotted-code structure (``"a.b.c"`` → ``"a.b"``) in
# :func:`_derive_parent_code`.
_LEGACY_RENAME = {
    "id": "code",
    "ontology": "label",
    "annotation": "abbrev",
    "definition": "description",
}


def _derive_parent_code(code: str) -> str:
    """Return the parent code for a dotted-hierarchy code.

    ``"1.1.1.1.1.1.7"`` → ``"1.1.1.1.1.1"``; root codes (``"0"``,
    ``"1"``) return ``""``.  This convention matches what the cached
    export emits and what downstream taxonomy-walking code expects.
    """
    if not code or "." not in code:
        return ""
    return code.rsplit(".", 1)[0]


def _row_to_dict(row, columns: list[str]) -> dict:
    """Convert one DataFrame row tuple to a normalized dict."""
    out: dict = {}
    for col, val in zip(columns, row):
        norm = _normalize(col)
        # Pandas NaN → None.  Import math locally to avoid a hard pandas dep.
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                val = None
        except Exception:
            pass
        out[norm] = val
    return out


def _legacy_to_universal(row: dict) -> dict:
    """Translate legacy-schema row to universal shape.

    Universal-shaped rows pass through unchanged (universal fields
    have priority over legacy keys when both are present).  Legacy
    rows are translated per ``_LEGACY_RENAME`` and the ``parent_code``
    is derived from the dotted-code structure since the legacy schema
    omits it.
    """
    out = dict(row)
    for legacy_key, universal_key in _LEGACY_RENAME.items():
        if legacy_key in out and not out.get(universal_key):
            out[universal_key] = out[legacy_key]
    # parent_code derivation: only fill if missing.  Universal exports
    # already carry it.
    if not out.get("parent_code"):
        out["parent_code"] = _derive_parent_code(out.get("code") or "")
    # mnemonic is used by some downstream code as a fallback id.
    # In the legacy schema the natural fallback is the abbrev
    # (e.g., "INOS", "PAN") which is short and unique within the
    # vocabulary.
    if not out.get("mnemonic"):
        out["mnemonic"] = out.get("abbrev") or out.get("code") or ""
    return out


def load_from_connection(
    *,
    connection_name: str,
    database: str,
    table_name: str = "annotations",
    limit: int | None = None,
) -> list[dict]:
    """Load source-taxonomy rows from a CAI Data Platform connection.

    Parameters
    ----------
    connection_name
        CAI Data connection identifier (e.g., ``"hive-poc"``).  No
        vocabulary identity is inferred from this name; it is opaque
        to the loader.
    database
        Database/schema containing the annotations table (e.g.,
        ``"default"``).
    table_name
        Annotations table name (default ``"annotations"``).
    limit
        Optional cap on the number of rows returned.  ``None`` (the
        default) returns the full table — appropriate for production
        enrichment runs.  Set a small integer for smoke tests.

    Returns
    -------
    list[dict]
        Each row carries at least ``code`` and ``label``; legacy
        ``id/ontology/annotation`` schemas are translated to the
        universal shape.

    Raises
    ------
    ImportError
        ``cml.data_v1`` is not available (this loader is only
        functional on a CAI runtime).
    RuntimeError
        Query against the connection failed.
    """
    try:
        import cml.data_v1 as cmldata  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "cml.data_v1 not available — load_from_connection only runs "
            "on CAI runtimes.  Use --source-json with a pre-exported "
            "snapshot for offline/dev environments."
        ) from exc

    qualified = f"{database}.{table_name}"
    query = f"SELECT * FROM {qualified}"
    if limit is not None:
        query += f" LIMIT {int(limit)}"

    conn = cmldata.get_connection(connection_name)
    try:
        df = conn.get_pandas_dataframe(query)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {qualified} from connection "
            f"{connection_name!r}: {exc}"
        ) from exc
    finally:
        try:
            conn.close()
        except Exception:
            pass

    columns = [str(c) for c in df.columns]
    rows = [
        _legacy_to_universal(_row_to_dict(t, columns))
        for t in df.itertuples(index=False, name=None)
    ]

    # Post-pass: prune orphan parent_code references.  Per-row legacy
    # translation derives parent_code from the dotted-code structure
    # (``"1.1"`` → ``"1"``); but the source vocabulary may use codes
    # like ``"1.1"`` as roots without a literal ``"1"`` row.  Convert
    # any unresolved parent reference into an empty string (root) so
    # downstream taxonomy walking doesn't follow dangling pointers.
    # This is vocabulary-agnostic: whatever convention the loaded
    # taxonomy uses, orphan refs become roots.
    code_set = {r.get("code") for r in rows if r.get("code")}
    orphan_count = 0
    for r in rows:
        pc = r.get("parent_code")
        if pc and pc not in code_set:
            r["parent_code"] = ""
            orphan_count += 1

    logger.info(
        "Loaded %d rows from %s/%s.%s; pruned %d orphan parent refs to root",
        len(rows), connection_name, database, table_name, orphan_count,
    )
    return rows
