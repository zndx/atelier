"""Meta-tagging data source — optional local directory mount.

Mounts a directory of CSV tables plus an ``annotations.csv`` vocabulary
file. This is the "sanctioned private dev source" for parity-testing
against UAT — the directory lives outside the repository
(``~/local/tmp/meta-tagging/`` by convention), its contents are never
committed, and nothing here persists values, labels, or codes back
into the repo.

Directory shape (matches the UAT reference layout)::

    <mount>/
      annotations.csv           # vocabulary: id, ontology, annotation, ...
      business_data.csv         # table — columns named + obfuscated pairs
      customer_data.csv
      ...

The tables follow a paired-column convention: each "clear" column
(e.g. ``personal_data.first_name``) has an obfuscated twin
(``personal_data.attr_1_1_1_9_2_1``) carrying the same data but whose
name *encodes* its ground-truth code (``1_1_1_9_2_1`` →
``1.1.1.9.2.1``).  Ground truth is thus derivable at load time:

- Obfuscated-named columns: parse code from the name suffix.
- Clear-named columns: map the name to the term's ontology label via
  a name→code lookup built from ``annotations.csv``.
- ``row_id`` columns: pinned to the ``0.1`` (non-sensitive) fallback.

Activation: mount resolution prefers, in order:
1. The ``ATELIER_META_TAGGING_DIR`` environment variable
2. ``cfg.classify_meta_tagging_dir`` (HOCON)
3. ``~/local/tmp/meta-tagging/`` (maintainer-convention default)

Returns ``None`` when no valid mount exists, at which point the source
is hidden from the UI and the pipeline won't accept ``source_id =
"meta-tagging"``.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path

from atelier.classify.sampler import ColumnSample, TableSample
from atelier.classify.taxonomy import (
    CategorySet,
    HierarchicalCategorySet,
    _build_category_set_from_records,
)


log = logging.getLogger(__name__)


# Obfuscated column-name prefixes seen in the UAT reference.  The code
# pattern is ``<prefix>_<digit>_<digit>_...`` where the digit tuple is
# the ontology path.
_OBFUSCATED_PREFIXES = (
    "attr", "field", "key", "code", "var", "ref", "data", "col",
)

_OBFUSCATED_RE = re.compile(
    r"^(?:" + "|".join(_OBFUSCATED_PREFIXES) + r")_(\d+(?:_\d+)*)$"
)


def _default_mount_candidate() -> Path:
    """Maintainer-convention location for the private meta-tagging dir."""
    return Path.home() / "local" / "tmp" / "meta-tagging"


def resolve_meta_tagging_mount(cfg=None) -> Path | None:  # type: ignore[no-untyped-def]
    """Return the directory Atelier should mount, or None.

    Probes (in precedence order):
      1. ``ATELIER_META_TAGGING_DIR`` env var
      2. ``cfg.classify_meta_tagging_dir`` (when a cfg is passed)
      3. ``~/local/tmp/meta-tagging/`` default

    A candidate passes only when it's an existing directory containing
    ``annotations.csv``.  Missing-or-malformed sources return None.
    """
    candidates: list[Path] = []
    env = os.environ.get("ATELIER_META_TAGGING_DIR", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    if cfg is not None:
        cfg_path = getattr(cfg, "classify_meta_tagging_dir", "") or ""
        if cfg_path:
            candidates.append(Path(cfg_path).expanduser())
    candidates.append(_default_mount_candidate())

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not (candidate / "annotations.csv").is_file():
            continue
        return candidate
    return None


def _load_annotations_records(mount: Path) -> list[dict]:
    """Read annotations.csv rows into normalized records for the taxonomy loader."""
    records: list[dict] = []
    with open(mount / "annotations.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys — first column is ``'ID`` with a leading apostrophe.
            code = (row.get("'ID") or row.get("ID") or row.get("id") or "").strip()
            if not code:
                continue
            records.append({
                "id": code,
                "ontology": (row.get("Ontology") or "").strip(),
                "annotation": (row.get("Annotation") or "").strip(),
                "definition": (row.get("Definition") or "").strip(),
                "common_names": (row.get("Common Names") or "").strip(),
                "specifics": (
                    row.get("Specifics, Examples and/or Additional Context")
                    or ""
                ).strip(),
                "deprecated": (row.get("Deprecated") or "").strip().lower(),
                # Sensitivity flags per data-subject role
                "non_corp": (row.get("NON_CORP") or "").strip(),
                "emp_contractor": (row.get("EMP, CONTRACTOR") or "").strip(),
                "individual": (row.get("INDIVIDUAL") or "").strip(),
                "corp": (row.get("CORP") or "").strip(),
                "taxonomy": "meta-tagging",
            })
    return records


def load_meta_tagging_vocabulary(mount: Path) -> HierarchicalCategorySet:
    """Build the hierarchical CategorySet from ``annotations.csv``.

    Delegates construction to :func:`taxonomy._build_reference_categories`
    — the same loader used for the Hive-sourced UAT path — so label
    normalization, leaf filtering, and parent derivation behave
    identically between environments.
    """
    records = _load_annotations_records(mount)
    cs = _build_category_set_from_records(records, hierarchical=True)
    if not isinstance(cs, HierarchicalCategorySet):
        cs = HierarchicalCategorySet(
            name="meta-tagging", categories=list(cs.categories),
        )
    log.info(
        "meta-tagging vocab: %d leaves (from %d annotation records)",
        len(cs.categories), len(records),
    )
    return cs


def _build_name_to_code_index(records: list[dict]) -> dict[str, str]:
    """Map human-readable names → ontology code for clear-named ground truth.

    For each annotation record we index: the ontology label, the
    annotation mnemonic, and any pipe-or-comma-separated common_names
    — all lowercased and underscore-normalized so ``"First Name"`` and
    ``"first_name"`` collide.
    """
    index: dict[str, str] = {}
    for r in records:
        code = r["id"]
        for key in (r.get("ontology"), r.get("annotation"), r.get("common_names")):
            if not key:
                continue
            for token in re.split(r"[|,]", key):
                norm = _normalize(token)
                if norm and norm not in index:
                    index[norm] = code
    return index


def _normalize(name: str) -> str:
    """Lowercase + replace separators with underscores + strip punctuation."""
    s = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _derive_ground_truth(
    col_name: str, name_index: dict[str, str], fallback_code: str
) -> str | None:
    """Resolve ground-truth code for a single column name.

    Obfuscated names (``attr_1_1_1_9_2_1``) → decode directly.
    Clear names (``first_name``) → look up in the name→code index.
    ``row_id``-style generic names → the fallback (non-sensitive).
    Anything else → None (unlabeled; evaluation counts as unknown).
    """
    # Obfuscated: prefix_1_1_1_9_2_1 → 1.1.1.9.2.1
    m = _OBFUSCATED_RE.match(col_name)
    if m:
        return m.group(1).replace("_", ".")

    # Generic pass-through names stay non-sensitive.
    if col_name in {"row_id"}:
        return fallback_code

    # Clear names → name index (ontology label or alias match).
    code = name_index.get(_normalize(col_name))
    return code


def _fallback_code(records: list[dict]) -> str:
    """Pick the taxonomy's non-sensitive catch-all for row_id etc.

    Prefers ``0.1`` when present (the convention in the UAT reference);
    otherwise the first depth-1 code in the records.
    """
    codes = {r["id"] for r in records}
    if "0.1" in codes:
        return "0.1"
    for r in records:
        code = r["id"]
        if code.count(".") == 1:
            return code
    # last resort — no non-sensitive anchor in this vocab
    return records[0]["id"] if records else ""


def load_meta_tagging_source(
    mount: Path | None = None,
) -> list[TableSample]:
    """Load every ``<table>.csv`` in the mount as a :class:`TableSample`.

    Ground truth per column is derived from the column-name convention
    described at module-level; ``ground_truth`` is set only when a
    mapping exists so evaluation can distinguish "classifier wrong" from
    "no reference label available".
    """
    if mount is None:
        mount = resolve_meta_tagging_mount()
    if mount is None:
        log.warning("meta-tagging: no mount resolved")
        return []

    records = _load_annotations_records(mount)
    name_index = _build_name_to_code_index(records)
    fallback = _fallback_code(records)

    samples: list[TableSample] = []
    for csv_path in sorted(mount.glob("*.csv")):
        if csv_path.name == "annotations.csv":
            continue
        table_name = csv_path.stem
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                continue
            rows = list(reader)

        # Strip "<table_name>." table-qualifier prefix on column names
        # if the CSV encodes them (the UAT reference does).
        qualifier = f"{table_name}."
        col_names = [
            (h[len(qualifier):] if h.startswith(qualifier) else h)
            for h in header
        ]

        columns: list[ColumnSample] = []
        for i, col_name in enumerate(col_names):
            all_vals = [row[i] for row in rows if i < len(row) and row[i]]
            values = all_vals[:5]
            total = len(rows)
            nulls = sum(1 for row in rows if i >= len(row) or not row[i])
            gt = _derive_ground_truth(col_name, name_index, fallback)

            columns.append(ColumnSample(
                name=col_name,
                column_type="object",
                values=values,
                all_values=all_vals,
                total_count=total,
                null_count=nulls,
                table_name=table_name,
                database="meta-tagging",
                siblings=col_names,
                ground_truth=gt,
                distinct_count=len(set(row[i] for row in rows if i < len(row))),
            ))

        samples.append(TableSample(
            name=table_name,
            database="meta-tagging",
            columns=columns,
        ))

    total_cols = sum(len(t.columns) for t in samples)
    labeled = sum(
        1 for t in samples for c in t.columns if c.ground_truth is not None
    )
    log.info(
        "meta-tagging source loaded: %d tables, %d columns (%d with ground truth) from %s",
        len(samples), total_cols, labeled, mount,
    )
    return samples


def meta_tagging_stats(mount: Path | None = None) -> dict:
    """Return summary stats for the source without loading all data."""
    if mount is None:
        mount = resolve_meta_tagging_mount()
    if mount is None:
        return {
            "table_count": 0, "column_count": 0,
            "has_data": False, "mount": None,
        }
    table_files = [
        p for p in mount.glob("*.csv")
        if p.name != "annotations.csv"
    ]
    col_count = 0
    for p in table_files:
        with open(p, newline="") as f:
            header = next(csv.reader(f), None)
            if header:
                col_count += len(header)
    return {
        "table_count": len(table_files),
        "column_count": col_count,
        "has_data": True,
        "mount": str(mount),
    }
