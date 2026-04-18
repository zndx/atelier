"""Ground-truth CSV ingest for UAT-style evaluation.

The Hive-backed pipeline doesn't surface ground-truth codes per
column.  When an operator has expected labels from an external
source (e.g. a UAT reviewer's xlsx → CSV export), loading them
here lets the pipeline populate ``col.ground_truth`` on every
classified column and produce real accuracy metrics in
``evaluation_report.json`` / overwatch.

Activation: set ``classify.ground_truth_uri`` in HOCON (or
``ATELIER_GROUND_TRUTH_URI`` in the env) to a CSV path.  Relative
paths resolve against the repo root.  ``file://`` scheme is also
accepted for parity with ``vocab_uri``.

Accepted CSV schemas
--------------------

Either columns explicitly::

    table_name,column_name,code,annotation
    personal_data,payment_card_number,1.1.1.1.1.1.1,PAN
    personal_data,attr_1_1_1_1_1_1_1,1.1.1.1.1.1.1,PAN

Or a pre-qualified single column (``annotation`` optional)::

    column_name,code,annotation
    personal_data.payment_card_number,1.1.1.1.1.1.1,PAN

Keys are indexed both with and without the ``table_name.`` prefix so
pipelines that strip qualifiers (e.g. the local meta-tagging loader)
and pipelines that keep them (Hive) both resolve correctly.

The module intentionally holds no references to private UAT data —
the CSV path is configured per deployment and its contents never
land in git.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)


def _resolve_uri(uri: str, project_root: Path) -> Path | None:
    """Normalize a ``file://`` or raw path into a concrete :class:`Path`."""
    if not uri:
        return None
    raw = uri[len("file://"):] if uri.startswith("file://") else uri
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = project_root / p
    p = p.resolve()
    return p if p.is_file() else None


def load_ground_truth_csv(
    uri: str,
    project_root: Path,
) -> dict[str, str]:
    """Return a ``{key: code}`` map; empty dict if *uri* is missing.

    Each row in the CSV gets indexed under multiple keys so lookups
    from different pipeline loaders resolve:

    - bare ``column_name`` (e.g. ``payment_card_number``)
    - qualified ``{table_name}.{column_name}`` (e.g.
      ``personal_data.payment_card_number``)

    When both forms appear in the CSV, the last one wins (dicts
    behave deterministically).  ``code`` may be any non-empty string
    — the pipeline compares it to ``predicted_code`` verbatim.
    """
    path = _resolve_uri(uri, project_root)
    if path is None:
        log.info("No ground-truth CSV resolvable from uri=%r", uri)
        return {}

    mapping: dict[str, str] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = (row.get("code") or row.get("predicted_code") or "").strip()
            if not code:
                continue
            table = (row.get("table_name") or "").strip()
            col = (row.get("column_name") or "").strip()
            if not col:
                continue
            # If column_name already carries the table qualifier, both
            # keys collapse to the same string — that's fine, dict upsert.
            mapping[col] = code
            if table and not col.startswith(f"{table}."):
                mapping[f"{table}.{col}"] = code
            # Also index a stripped form when column_name IS qualified.
            if "." in col:
                bare = col.split(".", 1)[1]
                mapping.setdefault(bare, code)

    log.info(
        "Loaded %d ground-truth entries (%d unique codes) from %s",
        len(mapping), len({v for v in mapping.values()}), path,
    )
    return mapping


def apply_ground_truth(
    samples: Iterable,
    mapping: dict[str, str],
) -> int:
    """Populate ``ColumnSample.ground_truth`` on every match.

    Tries the column's bare name first, then the ``{table}.{column}``
    qualified form, then (when the name already contains a dot) the
    stripped-prefix form.  Overwrites any pre-existing ``ground_truth``
    value because CSV-sourced labels are the intended reference.

    Returns the count of columns that got a ground-truth label —
    useful for logging / sanity checks.
    """
    if not mapping:
        return 0

    hits = 0
    for ts in samples:
        for col in ts.columns:
            key_bare = col.name
            key_qualified = f"{ts.name}.{col.name}" if not col.name.startswith(f"{ts.name}.") else col.name
            key_stripped = col.name.split(".", 1)[1] if "." in col.name else None

            code = (
                mapping.get(key_bare)
                or mapping.get(key_qualified)
                or (mapping.get(key_stripped) if key_stripped else None)
            )
            if code:
                col.ground_truth = code
                hits += 1
    return hits
