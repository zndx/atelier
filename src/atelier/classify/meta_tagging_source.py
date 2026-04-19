"""Meta-tagging data source — optional local directory mount.

Mounts a directory of CSV tables plus an ``annotations.csv`` vocabulary
file.  The UAT meta-tagging snapshot is the provisional corpus we load
here; it does **not** provide authoritative ground truth on its own
(see ``build/meta-tagging-clean/ground_truth.csv`` for the authoritative
reference — UAT's labels carry known obfuscation leaks and name-index
bugs which we correct at derivation time).  Directory contents are
gitignored and nothing here persists values, labels, or codes back
into the repo.

Directory shape (matches the UAT reference layout)::

    <mount>/
      annotations.csv           # vocabulary: id, ontology, annotation, ...
      business_data.csv         # table — natural-named + reference column pairs
      customer_data.csv
      ...

The tables follow a paired-column convention: each natural-named
column (e.g. ``personal_data.first_name``) has a *reference column*
twin (e.g. ``personal_data.attr_1_1_1_9_2_1``) whose name *encodes*
the paired column's ground-truth code (``1_1_1_9_2_1`` →
``1.1.1.9.2.1``).  **Reference columns are answer keys, not inputs**
— by invariant, they never appear in the sample set returned by this
loader.  Ground truth is derived at load time:

- Natural-named column with a paired reference twin: code from the
  twin's name suffix (authoritative — the synth generator guaranteed
  the pairing).
- Natural-named column without a twin: map the name to the term's
  ontology label via a name→code lookup built from
  ``annotations.csv`` with Ontology > Annotation > Common Names
  priority and a depth-winning tie-break (high/medium confidence).
- ``row_id`` columns: pinned to the ``0.1`` (non-sensitive) fallback.
- Reference-named columns: excluded; not classified and not in siblings.

Activation: mount resolution prefers, in order:
1. The ``ATELIER_META_TAGGING_DIR`` environment variable
2. ``cfg.classify_meta_tagging_dir`` (HOCON)
3. ``<repo>/build/meta-tagging/`` — in-repo, gitignored, UAT snapshot
4. ``~/local/tmp/meta-tagging/`` — legacy maintainer-convention default

The in-repo ``build/meta-tagging/`` slot is a symlink pointing at the
most recent dated UAT snapshot (e.g. ``meta-tagging-0418/``).  This
keeps the annotation vocabulary + table corpus co-located with the
runs that reference them while still respecting the "never commit
annotations.csv" privacy constraint (``build/`` is gitignored).

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


# Reference-column name pattern: ``<prefix>_<digit>_<digit>_...`` where
# the digit tuple encodes the ground-truth ontology code of the
# natural-named column immediately preceding it in schema order.
#
# Reference columns carry no independent classification signal — they
# are answer keys for their paired natural-named column, not data to
# classify.  By invariant, they never appear in train / test /
# validation / evaluation sample sets; they only contribute their
# encoded code as authoritative GT for the paired column.
#
# The prefix list is the union of prefixes observed across
# meta_tagging_source and real_data_loader historically, kept here as
# the single source of truth so both modules resolve the same pattern.
_REFERENCE_COL_PREFIXES = (
    "attr", "code", "col", "data", "field", "item", "key", "ref",
    "val", "var",
)

_REFERENCE_COL_RE = re.compile(
    r"^(?:" + "|".join(_REFERENCE_COL_PREFIXES) + r")_(\d+(?:_\d+)*)$"
)


def _repo_mount_candidate() -> Path:
    """In-repo gitignored slot for the UAT snapshot.

    Resolves to ``<repo_root>/build/meta-tagging/`` — typically a symlink
    to the most recent dated snapshot directory (``meta-tagging-0418/``
    at the time of writing).  ``build/`` is gitignored so the CSVs
    themselves never land in version control.
    """
    # <this file>: src/atelier/classify/meta_tagging_source.py
    # project_root = parents[3]  (classify → atelier → src → project)
    return Path(__file__).resolve().parents[3] / "build" / "meta-tagging"


def _legacy_mount_candidate() -> Path:
    """Maintainer-convention fallback for the private meta-tagging dir."""
    return Path.home() / "local" / "tmp" / "meta-tagging"


def resolve_meta_tagging_mount(cfg=None) -> Path | None:  # type: ignore[no-untyped-def]
    """Return the directory Atelier should mount, or None.

    Probes (in precedence order):
      1. ``ATELIER_META_TAGGING_DIR`` env var
      2. ``cfg.classify_meta_tagging_dir`` (when a cfg is passed)
      3. ``<repo>/build/meta-tagging/`` — in-repo, gitignored, UAT snapshot
      4. ``~/local/tmp/meta-tagging/`` — legacy fallback

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
    candidates.append(_repo_mount_candidate())
    candidates.append(_legacy_mount_candidate())

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not (candidate / "annotations.csv").is_file():
            continue
        return candidate
    return None


def _read_annotation_rows(annotations_csv: Path) -> list[dict]:
    """Read annotations.csv into normalized records (local helper).

    Parallels :func:`taxonomy.load_annotations_from_filesystem`'s row
    parsing but returns the intermediate ``list[dict]`` so callers can
    build auxiliary indices (name→code, fallback code) without having
    to walk a CategorySet.
    """
    from atelier.classify.taxonomy import _normalize_annotations_row
    records: list[dict] = []
    with open(annotations_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = _normalize_annotations_row(row)
            code = (row.get("'ID") or row.get("ID") or row.get("id") or "").strip()
            if not code:
                continue
            records.append({
                "id": code,
                "ontology": (row.get("Ontology") or row.get("ontology") or "").strip(),
                "annotation": (
                    row.get("Annotation") or row.get("annotation") or ""
                ).strip(),
                "common_names": (
                    row.get("Common Names") or row.get("common_names") or ""
                ).strip(),
            })
    return records


def load_meta_tagging_vocabulary(mount: Path) -> HierarchicalCategorySet:
    """Build the hierarchical CategorySet from ``annotations.csv``.

    Thin wrapper over :func:`taxonomy.load_annotations_from_filesystem`
    that pins the ``taxonomy`` namespace to ``"meta-tagging"`` so
    blended-vocabulary downstream code can tell this source apart from
    other filesystem-loaded vocabularies.
    """
    from atelier.classify.taxonomy import load_annotations_from_filesystem
    cs = load_annotations_from_filesystem(
        mount / "annotations.csv",
        hierarchical=True,
        taxonomy="meta-tagging",
    )
    if not isinstance(cs, HierarchicalCategorySet):
        cs = HierarchicalCategorySet(
            name="meta-tagging", categories=list(cs.categories),
        )
    log.info("meta-tagging vocab: %d leaves from %s", len(cs.categories), mount)
    return cs


def _build_name_to_code_index(records: list[dict]) -> dict[str, str]:
    """Map human-readable names → ontology code for clear-named ground truth.

    Each annotation record contributes up to three alias sources:
    ``ontology`` (the canonical label), ``annotation`` (the mnemonic),
    and ``common_names`` (pipe/comma-separated examples).  All are
    lowercased + underscore-normalized so ``"First Name"`` and
    ``"first_name"`` collide.

    Priority order — critical for correctness:

      1. ``ontology`` matches win over anything else.
      2. ``annotation`` matches win over ``common_names``.
      3. Within a tier, deeper (more specific) codes win over ancestors.

    Why this matters: annotations.csv parent entries often list their
    own children in ``common_names`` as examples (``First Name`` as a
    Common Name under ``Name (Full)``).  A naive first-writer-wins
    over CSV row order then incorrectly maps ``first_name`` → parent
    code.  Canonical Ontology matches and deeper codes take precedence
    so the leaf with ``Ontology = "First Name"`` wins as it should.
    """
    tiered: dict[str, tuple[int, int, str]] = {}
    TIER = {"ontology": 0, "annotation": 1, "common_names": 2}
    for r in records:
        code = r["id"]
        depth = code.count(".")
        for src_key, tier in TIER.items():
            key = r.get(src_key)
            if not key:
                continue
            for token in re.split(r"[|,]", key):
                norm = _normalize(token)
                if not norm:
                    continue
                existing = tiered.get(norm)
                # Replace when: no entry yet; strictly-higher-priority
                # tier; same tier with strictly-deeper code.
                if (
                    existing is None
                    or tier < existing[0]
                    or (tier == existing[0] and depth > existing[1])
                ):
                    tiered[norm] = (tier, depth, code)
    return {norm: code for norm, (_, _, code) in tiered.items()}


def _normalize(name: str) -> str:
    """Lowercase + replace separators with underscores + strip punctuation."""
    s = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _derive_ground_truth(
    col_name: str, name_index: dict[str, str], fallback_code: str
) -> str | None:
    """Resolve ground-truth code for a single column name.

    Reference names (``attr_1_1_1_9_2_1``) → decode directly.  Note:
    reference columns themselves are *not* classified — they exist to
    encode GT for their paired natural-named column.  This helper is
    still used to decode their embedded code when we need it as the
    paired column's ground truth.

    Natural names (``first_name``) → name→code index lookup.
    ``row_id`` generic names → the non-sensitive fallback.
    Anything else → None (unlabeled; evaluation counts as unknown).
    """
    # Reference-column form: prefix_1_1_1_9_2_1 → 1.1.1.9.2.1
    m = _REFERENCE_COL_RE.match(col_name)
    if m:
        code = m.group(1).replace("_", ".")
        # UAT Hive export normalized the "Not Sensitive" root code
        # ``0.0`` → ``0``.  Column names still carry the Gopala-vintage
        # ``0_0`` suffix; strip a trailing ``.0`` at the root so GT
        # matches the UAT vocabulary.  Safe: no sub-tier ends in ``.0``.
        if "." in code and code.count(".") == 1 and code.endswith(".0"):
            code = code[:-2]
        return code

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

    # Build the annotation records once for both the name→code index
    # and the fallback-code heuristic.  Uses the same normalized shape
    # as the filesystem vocab loader in taxonomy.py.
    records = _read_annotation_rows(mount / "annotations.csv")
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

        # Reference-column exclusion invariant.  Reference columns
        # (``attr_1_2_3_4``, ``code_1_1``, etc.) encode the paired
        # natural-named column's GT directly in their name.  They are
        # answer keys, not classifiable inputs.  Two exclusion vectors:
        #
        #   1. Reference columns are not returned as samples.  Dropping
        #      them prevents trivial name-parse "predictions" from
        #      inflating accuracy metrics.
        #   2. A natural-named column's sibling context (rendered into
        #      its ``embedding_text``) would otherwise include its
        #      reference twin, leaking the answer code through siblings.
        #      Strip reference names from the siblings list.
        #
        # Production data doesn't use the reference-column convention,
        # so these filters are no-ops there.  On the UAT reference
        # corpus they isolate the honest evaluation signal — accuracy
        # on columns that must be classified from values + name, not
        # from an adjacent answer key.
        clean_sibling_names = [
            n for n in col_names if not _REFERENCE_COL_RE.match(n)
        ]

        columns: list[ColumnSample] = []
        for i, col_name in enumerate(col_names):
            if _REFERENCE_COL_RE.match(col_name):
                continue  # leak vector #1: don't classify the answer key
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
                siblings=clean_sibling_names,  # leak vector #2: sanitized
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
    # Reference-exclusion invariant — catch any regression that would
    # let an answer-key column back into the classifiable sample set.
    leaked_refs = [
        c.name for t in samples for c in t.columns
        if _REFERENCE_COL_RE.match(c.name)
    ]
    assert not leaked_refs, (
        f"reference columns leaked into samples (should be excluded as "
        f"answer keys): {leaked_refs[:5]}"
    )
    log.info(
        "meta-tagging source loaded: %d tables, %d columns "
        "(%d with ground truth; reference columns excluded) from %s",
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
