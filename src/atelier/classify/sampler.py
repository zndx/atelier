"""Metadata sampler for hive tables via CAI Data Platform.

Discovers tables and samples column metadata from production databases.
For dev/test without hive, use load_fixture_samples() and inject via samples=.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ColumnSample:
    """Sampled metadata for a single column."""

    name: str
    column_type: str | None = None
    values: list[str] = field(default_factory=list)
    total_count: int = 0
    null_count: int = 0
    table_name: str = ""
    database: str = ""
    siblings: list[str] = field(default_factory=list)
    ground_truth: str | None = None  # Known annotation code (for validation)
    distinct_count: int | None = None  # True COUNT(DISTINCT) bounded by column_sample_limit

    def to_dict(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "column_type": self.column_type,
            "values": self.values,
            "total_count": self.total_count,
            "null_count": self.null_count,
            "table_name": self.table_name,
            "database": self.database,
            "siblings": self.siblings,
            "ground_truth": self.ground_truth,
        }
        if self.distinct_count is not None:
            d["distinct_count"] = self.distinct_count
        return d


@dataclass
class TableSample:
    """Sampled metadata for a table."""

    name: str
    database: str = ""
    columns: list[ColumnSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "database": self.database,
            "columns": [c.to_dict() for c in self.columns],
        }


def discover_tables(
    cfg,
    connection_name: str | None = None,
    database: str = "default",
    limit: int = 100,
) -> list[str]:
    """List tables from a hive database via CAI Data Platform.

    Raises RuntimeError when cml.data_v1 is unavailable.
    For dev/test, inject samples= into the pipeline instead.
    """
    try:
        import cml.data_v1 as cmldata
    except ImportError:
        raise RuntimeError(
            "cml.data_v1 not available — inject samples= for dev/test"
        ) from None

    if connection_name is None:
        names = cfg.cml_data_connection_names
        if not names:
            raise ValueError("No data connections configured")
        connection_name = names[0]

    conn = cmldata.get_connection(connection_name)
    df = conn.get_pandas_dataframe(f"SHOW TABLES IN {database}")
    tables = df.iloc[:, 0].tolist()[:limit]
    return [str(t) for t in tables]


def sample_table_metadata(
    cfg,
    table_name: str,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int = 50,
    column_sample_limit: int = 1000,
) -> TableSample:
    """Sample column metadata from a hive table.

    Raises RuntimeError when cml.data_v1 is unavailable.
    For dev/test, inject samples= into the pipeline instead.
    """
    try:
        import cml.data_v1 as cmldata
    except ImportError:
        raise RuntimeError(
            "cml.data_v1 not available — inject samples= for dev/test"
        ) from None

    if connection_name is None:
        names = cfg.cml_data_connection_names
        if not names:
            raise ValueError("No data connections configured")
        connection_name = names[0]

    conn = cmldata.get_connection(connection_name)
    df = conn.get_pandas_dataframe(
        f"SELECT * FROM {database}.{table_name} LIMIT {sample_size}"
    )

    column_names = list(df.columns)

    # True cardinality: one query for all columns, bounded by limit
    distinct_counts: dict[str, int] = {}
    try:
        distinct_exprs = ", ".join(
            f"COUNT(DISTINCT `{col}`) AS `distinct_{col}`"
            for col in column_names
        )
        cardinality_df = conn.get_pandas_dataframe(
            f"SELECT {distinct_exprs} FROM "
            f"(SELECT * FROM {database}.{table_name} "
            f"LIMIT {column_sample_limit}) sub"
        )
        for col in column_names:
            distinct_counts[col] = int(cardinality_df[f"distinct_{col}"].iloc[0])
    except Exception:
        pass  # Fall back to sample-based cardinality in features

    columns = []
    for col_name in column_names:
        col_values = [str(v) for v in df[col_name].dropna().head(5).tolist()]
        null_count = int(df[col_name].isna().sum())
        total_count = len(df)
        col_type = str(df[col_name].dtype)

        columns.append(ColumnSample(
            name=col_name,
            column_type=col_type,
            values=col_values,
            total_count=total_count,
            null_count=null_count,
            table_name=table_name,
            database=database,
            siblings=column_names,
            distinct_count=distinct_counts.get(col_name),
        ))

    return TableSample(name=table_name, database=database, columns=columns)


def load_annotations_from_hive(
    cfg,
    connection_name: str | None = None,
) -> list[dict]:
    """Load annotation records from default.annotations via hive.

    Returns empty list when hive is unavailable (no domain extensions).
    """
    try:
        import cml.data_v1 as cmldata

        if connection_name is None:
            names = cfg.cml_data_connection_names
            if not names:
                raise ValueError("No data connections configured")
            connection_name = names[0]

        conn = cmldata.get_connection(connection_name)
        df = conn.get_pandas_dataframe("SELECT * FROM default.annotations")
        records = df.to_dict("records")
        log.info(
            "Loaded %d annotation records from hive (columns: %s)",
            len(records), list(df.columns),
        )
        return records
    except ImportError:
        log.info("cml.data_v1 not available — no domain annotations loaded")
        return []
    except Exception as exc:
        log.warning("Failed to load annotations from hive: %s — no domain annotations loaded", exc)
        return []


# ── Fixture data (for dev/test) ───────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _fixture_table_names() -> list[str]:
    """Return fixture table names from static test data."""
    tables = _load_fixture_tables()
    return [t["table_name"] for t in tables]


def _fixture_table_sample(table_name: str) -> TableSample:
    """Return a fixture TableSample for the named table."""
    tables = _load_fixture_tables()
    for t in tables:
        if t["table_name"] == table_name:
            col_names = [c["name"] for c in t["columns"]]
            columns = [
                ColumnSample(
                    name=c["name"],
                    column_type=c.get("type"),
                    values=c.get("values", []),
                    total_count=len(c.get("values", [])),
                    null_count=0,
                    table_name=table_name,
                    database=t.get("database", "default"),
                    siblings=col_names,
                    ground_truth=c.get("ground_truth"),
                    distinct_count=c.get("distinct_count"),
                )
                for c in t["columns"]
            ]
            return TableSample(
                name=table_name,
                database=t.get("database", "default"),
                columns=columns,
            )
    return TableSample(name=table_name)


def _load_fixture_tables() -> list[dict]:
    """Load fixture tables from static test data."""
    path = _FIXTURES_DIR / "fixture_tables.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_fixture_samples() -> list[TableSample]:
    """Load all fixture TableSamples for dev/test."""
    tables = _load_fixture_tables()
    return [_fixture_table_sample(t["table_name"]) for t in tables]


# Backward-compatible alias
load_all_mock_samples = load_fixture_samples


# ── Sample source data (OOTB onboarding) ─────────────────────────

_SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "sample"


def load_sample_source(
    sample_dir: str | Path | None = None,
) -> list[TableSample]:
    """Load OOTB sample tables from data/sample/tables/*.csv.

    Reads all CSVs in the sample tables directory and the ground truth
    mapping. Returns TableSample objects with ground_truth labels attached
    to each column — ready to inject into the classification pipeline.
    """
    import csv as csv_mod

    base = Path(sample_dir) if sample_dir else _SAMPLE_DIR
    tables_dir = base / "tables"
    gt_path = base / "ground_truth.json"

    if not tables_dir.is_dir():
        log.warning("Sample tables directory not found: %s", tables_dir)
        return []

    # Load ground truth if available
    ground_truth: dict[str, str] = {}
    if gt_path.exists():
        with open(gt_path) as f:
            ground_truth = json.load(f)

    samples: list[TableSample] = []
    for csv_path in sorted(tables_dir.glob("*.csv")):
        table_name = csv_path.stem
        with open(csv_path, newline="") as f:
            reader = csv_mod.reader(f)
            header = next(reader, None)
            if not header:
                continue
            # Read all rows for sampling
            rows = list(reader)

        col_names = header
        columns: list[ColumnSample] = []
        for i, col_name in enumerate(col_names):
            values = [row[i] for row in rows[:5] if i < len(row) and row[i]]
            total_count = len(rows)
            null_count = sum(1 for row in rows if i >= len(row) or not row[i])
            gt_key = f"{table_name}.{col_name}"

            columns.append(ColumnSample(
                name=col_name,
                column_type="object",
                values=values,
                total_count=total_count,
                null_count=null_count,
                table_name=table_name,
                database="sample",
                siblings=col_names,
                ground_truth=ground_truth.get(gt_key),
                distinct_count=len(set(row[i] for row in rows if i < len(row))),
            ))

        samples.append(TableSample(
            name=table_name,
            database="sample",
            columns=columns,
        ))

    log.info(
        "Loaded %d sample tables (%d columns) from %s",
        len(samples),
        sum(len(t.columns) for t in samples),
        tables_dir,
    )
    return samples


def sample_source_stats(sample_dir: str | Path | None = None) -> dict:
    """Return summary stats for the sample source without loading all data."""
    base = Path(sample_dir) if sample_dir else _SAMPLE_DIR
    tables_dir = base / "tables"
    gt_path = base / "ground_truth.json"

    table_count = len(list(tables_dir.glob("*.csv"))) if tables_dir.is_dir() else 0
    column_count = 0
    if gt_path.exists():
        with open(gt_path) as f:
            column_count = len(json.load(f))

    return {
        "table_count": table_count,
        "column_count": column_count,
        "has_data": table_count > 0,
    }
