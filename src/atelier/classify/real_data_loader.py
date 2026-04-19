"""Load real annotated data from CSV files for validation.

Parses CSV files from ``<repo>/build/meta-tagging/`` (UAT snapshot, in-repo
but gitignored) or ``~/local/tmp/meta-tagging/`` (legacy default) — the
full search chain lives in
``atelier.classify.meta_tagging_source.resolve_meta_tagging_mount``.
Ground-truth annotation codes are interspersed as adjacent columns.
**The CSV contents must NEVER enter git** — they carry Cloudera-internal
Ontology/Annotation mappings.

Column naming pattern:
  tablename.target_column         — the data column
  tablename.prefix_1_2_3_4       — ground truth code 1.2.3.4

Prefixes: attr, code, ref, key, val, var, field, data, item, col.
"""

from __future__ import annotations

import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from atelier.classify.sampler import ColumnSample, TableSample

logger = logging.getLogger(__name__)

# Matches annotation columns like attr_1_1_1_1_1_1_1 or code_1_1_1_1_2_1
_GT_PREFIX = re.compile(
    r"^(?:attr|code|ref|key|val|var|field|data|item|col)_(\d+(?:_\d+)*)$"
)

# CSV header mapping: annotations.csv → hive record schema
_HEADER_MAP = {
    # Gopala-vintage Excel export headers.
    "'ID": "id",
    "ID": "id",
    "Ontology": "ontology",
    "Annotation": "annotation",
    "Definition": "definition",
    "Common Names": "common_names",
    "Specifics, Examples and/or Additional Context": "specifics",
    "NON_CORP": "non_corp",
    "EMP, CONTRACTOR": "emp_contractor",
    "INDIVIDUAL": "individual",
    "CORP": "corp",
    "Deprecated": "deprecated",
    # UAT Hive export headers — lowercase, underscore-separated.  The
    # dotted ``annotations.`` prefix is stripped before lookup (see
    # ``_normalize_annotations_row`` below).
    "id": "id",
    "ontology": "ontology",
    "annotation": "annotation",
    "definition": "definition",
    "common_names": "common_names",
    "non_corp": "non_corp",
    "emp_contractor": "emp_contractor",
    "individual": "individual",
    "corp": "corp",
    "deprecated": "deprecated",
}


def _strip_table_prefix(fieldnames: list[str]) -> tuple[list[str], str]:
    """Detect + strip a uniform ``<table>.`` prefix on column headers.

    The UAT Hive export of ``annotations.csv`` prefixes every column with
    ``annotations.`` (``annotations.id``, ``annotations.ontology``, …).
    Strip the prefix up-front so header-map lookups match.  Returns
    ``(stripped_fieldnames, prefix)`` — prefix is ``""`` when no uniform
    prefix is present.
    """
    dotted = [f for f in fieldnames if "." in f]
    if not dotted or len(dotted) != len(fieldnames):
        return fieldnames, ""
    prefixes = {f.split(".", 1)[0] for f in fieldnames}
    if len(prefixes) != 1:
        return fieldnames, ""
    prefix = next(iter(prefixes)) + "."
    return [f[len(prefix):] for f in fieldnames], prefix


def load_annotations_csv(path: Path) -> list[dict]:
    """Parse annotations.csv into records matching hive schema.

    Handles the leading single-quote BOM on the first header field
    and maps CSV column names to standard record keys (id, ontology,
    annotation, definition, common_names, etc.).

    Returns list[dict] compatible with taxonomy._build_category_set_from_records().
    """
    records: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return records

        # Build field mapping from actual headers.  Two tolerances:
        #   1. ``<table>.`` prefix stripping for the UAT Hive export
        #      (``annotations.id`` → ``id``).
        #   2. Leading apostrophe from the Excel export (``'ID`` → ``ID``).
        stripped_names, _prefix = _strip_table_prefix(list(reader.fieldnames))
        field_map: dict[str, str] = {}
        for raw_name, stripped in zip(reader.fieldnames, stripped_names):
            clean = stripped.strip().strip("'")
            if stripped in _HEADER_MAP:
                field_map[raw_name] = _HEADER_MAP[stripped]
            elif clean in _HEADER_MAP:
                field_map[raw_name] = _HEADER_MAP[clean]

        for row in reader:
            rec: dict[str, Any] = {}
            for raw_name, mapped_name in field_map.items():
                val = row.get(raw_name, "")
                if val is None:
                    val = ""
                rec[mapped_name] = str(val).strip()
            # Skip rows without a valid code
            row_id = rec.get("id", "").strip()
            if row_id and row_id[0].isdigit():
                records.append(rec)

    logger.info("Loaded %d annotation records from %s", len(records), path)
    return records


def parse_real_csv(
    path: Path,
    *,
    sample_size: int = 50,
) -> list[tuple[str, str, list[str]]]:
    """Parse a single real data CSV, extracting target/ground-truth pairs.

    Returns list of (column_name, ground_truth_code, sample_values).
    Column names have the table prefix stripped (e.g., personal_data.email → email).
    """
    results: list[tuple[str, str, list[str]]] = []

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)

        # Read all rows into memory (files are small, ~100 rows)
        all_rows = list(reader)

    # Strip table prefix from headers
    clean_headers: list[str] = []
    for h in headers:
        parts = h.split(".", 1)
        clean_headers.append(parts[1] if len(parts) > 1 else parts[0])

    # Identify annotation columns and pair them with preceding target columns
    for i, name in enumerate(clean_headers):
        m = _GT_PREFIX.match(name)
        if not m:
            continue

        code = m.group(1).replace("_", ".")
        # UAT's Hive export normalized the "Not Sensitive" root code
        # from ``0.0`` to ``0``.  Column names in the data CSVs still
        # encode the Gopala-vintage ``0_0`` suffix; strip a trailing
        # ``.0`` from the root so the extracted ground truth matches
        # the UAT vocabulary.  Safe under the current tree: no
        # non-root code ends in ``.0`` (all sub-tiers start at ``.1``).
        if "." in code and code.count(".") == 1 and code.endswith(".0"):
            code = code[:-2]

        # The target column is the one immediately before this annotation column
        if i == 0:
            continue
        prev_name = clean_headers[i - 1]
        if _GT_PREFIX.match(prev_name):
            # Skip consecutive annotation columns
            continue
        if prev_name == "row_id":
            continue

        # Collect sample values from the target column
        target_idx = i - 1
        values: list[str] = []
        for row in all_rows[:sample_size]:
            if target_idx < len(row):
                val = row[target_idx].strip()
                if val:
                    values.append(val)

        results.append((prev_name, code, values))

    return results


def load_real_samples(
    data_dir: Path,
    *,
    sample_size: int = 50,
) -> list[TableSample]:
    """Load all real CSVs from data_dir as TableSample objects.

    Each CSV becomes one TableSample. Target columns become ColumnSample
    objects with ground_truth set. The annotations.csv file is skipped.
    """
    data_dir = Path(data_dir)
    samples: list[TableSample] = []

    csv_files = sorted(
        p for p in data_dir.glob("*.csv")
        if p.name != "annotations.csv"
    )

    for csv_path in csv_files:
        table_name = csv_path.stem
        parsed = parse_real_csv(csv_path, sample_size=sample_size)

        if not parsed:
            continue

        target_names = [name for name, _, _ in parsed]
        columns: list[ColumnSample] = []
        for col_name, gt_code, values in parsed:
            columns.append(ColumnSample(
                name=col_name,
                column_type=_infer_type(values),
                values=values[:5],  # Keep 5 for embedding text, like mock
                total_count=len(values),
                null_count=0,
                table_name=table_name,
                database="real_data",
                siblings=target_names,
                ground_truth=gt_code,
            ))

        samples.append(TableSample(
            name=table_name,
            database="real_data",
            columns=columns,
        ))

    total_cols = sum(len(t.columns) for t in samples)
    logger.info(
        "Loaded %d columns across %d tables from %s",
        total_cols, len(samples), data_dir,
    )
    return samples


def extract_value_templates(
    data_dir: Path,
    *,
    max_per_code: int = 200,
) -> dict[str, list[str]]:
    """Collect real data values grouped by ground truth code.

    Returns {code: [value1, value2, ...]} with at most max_per_code
    values per code. Used as templates for synthetic data generation.
    """
    data_dir = Path(data_dir)
    templates: dict[str, list[str]] = defaultdict(list)

    csv_files = sorted(
        p for p in data_dir.glob("*.csv")
        if p.name != "annotations.csv"
    )

    for csv_path in csv_files:
        # Read all data with full sample size for template extraction
        parsed = parse_real_csv(csv_path, sample_size=max_per_code)
        for _, gt_code, values in parsed:
            existing = templates[gt_code]
            remaining = max_per_code - len(existing)
            if remaining > 0:
                existing.extend(values[:remaining])

    logger.info(
        "Extracted value templates for %d codes from %s",
        len(templates), data_dir,
    )
    return dict(templates)


def load_real_vocabulary(
    data_dir: Path,
    *,
    hierarchical: bool = True,
):
    """Load annotations.csv from data_dir and build a CategorySet.

    Convenience wrapper: load_annotations_csv() → _build_category_set_from_records().
    """
    from atelier.classify.taxonomy import _build_category_set_from_records

    data_dir = Path(data_dir)
    ann_path = data_dir / "annotations.csv"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.csv not found in {data_dir}")

    records = load_annotations_csv(ann_path)
    return _build_category_set_from_records(records, hierarchical=hierarchical)


def _infer_type(values: list[str]) -> str:
    """Infer SQL-like column type from sample values."""
    if not values:
        return "VARCHAR"

    numeric = 0
    for v in values[:20]:
        try:
            float(v.replace(",", ""))
            numeric += 1
        except (ValueError, AttributeError):
            pass

    if numeric > len(values[:20]) * 0.8:
        # Check if integers
        all_int = all(
            "." not in v and "e" not in v.lower()
            for v in values[:20]
            if v.replace(",", "").replace("-", "").isdigit()
        )
        return "INTEGER" if all_int else "DECIMAL"

    return "VARCHAR"
