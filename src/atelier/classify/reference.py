# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Curated-reference CSV ingest for UAT-style evaluation.

The Hive-backed pipeline doesn't surface reference codes per column.
When an operator has expected labels from an external source (e.g. a
reviewer's xlsx → CSV export), loading them here lets the pipeline
populate ``col.reference_code`` on every classified column and produce
real accuracy metrics in ``evaluation_report.json`` / overwatch.

Activation: set ``classify.reference_uri`` in HOCON (or
``ATELIER_CLASSIFY_REFERENCE_URI`` in the env) to a CSV path.  Relative
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

Or mnemonic-only — shape emitted by ``ingest_reference`` when reading
a reviewer xlsx where each row has only ``(column_name, MNEMONIC)``
and the vocabulary code has to be resolved by looking the mnemonic up
in the loaded taxonomy::

    column_name,annotation
    personal_data.payment_card_number,PAN
    personal_data.attr_1_1_1_1_1_1_1,PAN

In the mnemonic-only case, ``load_reference_csv`` needs a
``category_set`` to resolve ``PAN → 1.1.1.1.1.1.1`` via
``category_set.by_abbrev``.  Without a category_set, mnemonic-only
rows are skipped with a warning — they can't be matched against
predictions that carry codes, not mnemonics.

Keys are indexed both with and without the ``table_name.`` prefix so
pipelines that strip qualifiers (e.g. the local meta-tagging loader)
and pipelines that keep them (Hive) both resolve correctly.

The module intentionally holds no references to private UAT data —
the CSV path is configured per deployment and its contents never
land in git.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.classify.taxonomy import CategorySet

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


def _build_abbrev_index(category_set: "CategorySet | None") -> dict[str, str]:
    """Case-insensitive mnemonic → code lookup from a category set.

    The xlsx → CSV ingest path (``ingest_reference`` CLI) emits
    reviewer mnemonics in the ``annotation`` column; this index is
    what turns those mnemonics into the codes the pipeline compares
    against ``predicted_code``.  Returns an empty dict when no
    category_set is provided, which preserves the historical
    "code or predicted_code" behavior for callers that don't plumb
    the vocabulary through.
    """
    if category_set is None:
        return {}
    idx: dict[str, str] = {}
    for cat in getattr(category_set, "categories", []):
        abbrev = getattr(cat, "abbrev", "") or ""
        if abbrev:
            idx[abbrev.upper()] = cat.code
    # ``by_abbrev`` and ``all_by_code`` cover the full taxonomy under
    # the unified ``categories`` semantics — walking ``all_by_code``
    # here is a redundant safety net that catches any caller still
    # passing a CategorySet built via the legacy flat-construction
    # path (which only carries terminal nodes).
    for cat in getattr(category_set, "all_by_code", {}).values():
        abbrev = getattr(cat, "abbrev", "") or ""
        if abbrev:
            idx.setdefault(abbrev.upper(), cat.code)
    return idx


def load_reference_csv(
    uri: str,
    project_root: Path,
    category_set: "CategorySet | None" = None,
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

    When a row carries no ``code`` column but does carry an
    ``annotation`` mnemonic AND a ``category_set`` is provided, the
    mnemonic is resolved to a code via ``category_set.by_abbrev``.
    This is the shape emitted by ``ingest_reference`` from a reviewer
    xlsx.
    """
    path = _resolve_uri(uri, project_root)
    if path is None:
        log.info("No reference CSV resolvable from uri=%r", uri)
        return {}

    abbrev_index = _build_abbrev_index(category_set)

    mapping: dict[str, str] = {}
    rows_total = 0
    rows_no_match = 0
    rows_via_annotation = 0
    unresolved_mnemonics: set[str] = set()

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_total += 1
            code = (
                row.get("reference_code")
                or row.get("code")
                or row.get("predicted_code")
                or ""
            ).strip()
            if not code:
                annotation = (row.get("annotation") or "").strip()
                if annotation and abbrev_index:
                    code = abbrev_index.get(annotation.upper(), "")
                    if code:
                        rows_via_annotation += 1
                    else:
                        unresolved_mnemonics.add(annotation)
                if not code:
                    rows_no_match += 1
                    continue

            table = (row.get("table_name") or row.get("table") or "").strip()
            col = (row.get("column_name") or row.get("column") or "").strip()
            if not col:
                continue
            mapping[col] = code
            if table and not col.startswith(f"{table}."):
                mapping[f"{table}.{col}"] = code
            if "." in col:
                bare = col.split(".", 1)[1]
                mapping.setdefault(bare, code)

    log.info(
        "Loaded %d reference entries (%d unique codes) from %s "
        "[%d rows total; %d resolved via annotation mnemonic; %d unresolved]",
        len(mapping), len({v for v in mapping.values()}), path,
        rows_total, rows_via_annotation, rows_no_match,
    )
    if unresolved_mnemonics:
        log.warning(
            "Reference CSV had %d mnemonic(s) missing from the vocabulary: %s",
            len(unresolved_mnemonics),
            ", ".join(sorted(unresolved_mnemonics)[:10])
            + ("…" if len(unresolved_mnemonics) > 10 else ""),
        )
    return mapping


def _is_dual_format_entry(v: object) -> bool:
    """Detect dual-format entry: dict with ``mnemonic`` and/or ``code`` keys.

    Distinguishes from the legacy nested-by-table form
    ``{table: {col: mnemonic}}``, whose values are plain strings.
    """
    return isinstance(v, dict) and ("mnemonic" in v or "code" in v)


def _resolve_entry(
    qkey: str,
    v: object,
    by_abbrev: dict,
    unresolved: set[str],
    drift_events: list[dict],
) -> str | None:
    """Resolve one reference entry to a code, honoring dual-format
    captured codes and recording structural drift.

    Dual-format entries store both ``mnemonic`` and ``code``; the
    captured ``code`` is authoritative because mnemonics can be moved
    within the ontology by curators (structural drift).  When the
    current mnemonic→code resolution disagrees with the captured
    code, the disagreement is recorded as a drift event for operator
    visibility — the captured code still wins.
    """
    if _is_dual_format_entry(v):
        captured_code = v.get("code")
        mnemonic = v.get("mnemonic")
        if captured_code:
            if mnemonic and by_abbrev:
                cat = by_abbrev.get(mnemonic.upper()) or by_abbrev.get(mnemonic)
                if cat is not None and cat.code != captured_code:
                    drift_events.append({
                        "qkey": qkey,
                        "mnemonic": mnemonic,
                        "captured_code": captured_code,
                        "current_code": cat.code,
                    })
            return captured_code
        if mnemonic:
            cat = by_abbrev.get(mnemonic.upper()) or by_abbrev.get(mnemonic)
            if cat is not None:
                return cat.code
            unresolved.add(mnemonic)
        return None
    if isinstance(v, str) and v:
        cat = by_abbrev.get(v.upper()) or by_abbrev.get(v)
        if cat is not None:
            return cat.code
        unresolved.add(v)
    return None


def load_reference_agent_mediated(
    path: Path,
    category_set: "CategorySet | None" = None,
) -> dict[str, str]:
    """Load an agent-mediated reference JSON and return ``{key: code}``.

    Supported entry shapes:

    * **Dual format** (preferred) — ``{"mnemonic": "EMAIL", "code":
      "1.1.1.9.3.1", "captured_at": "...", "source": "..."}``.  The
      ``code`` is authoritative; the ``mnemonic`` is the human anchor.
      A structural-drift event is logged when the current mnemonic→code
      resolution disagrees with the captured code (curator-induced
      ontology motion); the captured code still wins.
    * **Legacy mnemonic-only** — ``"EMAIL"``.  Resolved against the
      current category_set; carries no drift guard.
    * **Legacy nested-by-table** — ``{"table": {"col": "EMAIL"}}``.
      Flattened into qualified keys at load time.

    Keys are indexed in both qualified (``table.column``) and bare
    (``column``) forms so ``apply_reference`` resolves either shape.
    """
    if not path.is_file():
        log.warning(
            "Agent-mediated reference path %s does not exist — "
            "reference_code will not be populated",
            path,
        )
        return {}

    try:
        raw = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Agent-mediated reference load failed: %s", exc)
        return {}

    if not isinstance(raw, dict):
        log.warning(
            "Agent-mediated reference root must be a JSON object, got %s",
            type(raw).__name__,
        )
        return {}

    by_abbrev = getattr(category_set, "all_by_abbrev", None) or {}

    flat: dict[str, str] = {}
    unresolved: set[str] = set()
    drift_events: list[dict] = []

    for k, v in raw.items():
        # Legacy nested-by-table form: dict whose values are themselves
        # mnemonic strings (NOT a dual-format entry).
        if isinstance(v, dict) and not _is_dual_format_entry(v):
            for col, ann in v.items():
                if not ann:
                    continue
                qname = f"{k}.{col}"
                code = _resolve_entry(qname, ann, by_abbrev, unresolved, drift_events)
                if code is not None:
                    flat[qname] = code
                    flat.setdefault(col, code)
            continue
        # Either dual-format entry or legacy mnemonic-only string.
        code = _resolve_entry(k, v, by_abbrev, unresolved, drift_events)
        if code is not None:
            flat[k] = code
            if "." in k:
                bare = k.split(".", 1)[1]
                flat.setdefault(bare, code)

    resolved = len({v for v in flat.values()})
    total_entries = sum(
        len(v) if isinstance(v, dict) and not _is_dual_format_entry(v) else 1
        for v in raw.values()
    )
    log.info(
        "Agent-mediated reference: %d entries → %d resolved keys "
        "(%d unique codes) from %s",
        total_entries, len(flat), resolved, path,
    )
    if unresolved:
        log.warning(
            "Agent-mediated reference: %d mnemonic(s) not in vocabulary: %s",
            len(unresolved),
            ", ".join(sorted(unresolved)[:10])
            + ("…" if len(unresolved) > 10 else ""),
        )
    if drift_events:
        log.warning(
            "Agent-mediated reference: %d structural-drift event(s) — "
            "captured code disagrees with current mnemonic resolution. "
            "Using captured codes (per no-silent-drift policy). "
            "First 5: %s",
            len(drift_events), drift_events[:5],
        )
    return flat


def apply_reference(
    samples: Iterable,
    mapping: dict[str, str],
) -> int:
    """Populate ``ColumnSample.reference_code`` on every match.

    Tries the column's bare name first, then the ``{table}.{column}``
    qualified form, then (when the name already contains a dot) the
    stripped-prefix form.  Overwrites any pre-existing reference
    value because CSV-sourced labels are the intended reference.

    Returns the count of columns that got a reference label — useful
    for logging / sanity checks.
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
                col.reference_code = code
                hits += 1
    return hits
