"""Metadata sampler for hive tables via CAI Data Platform.

Discovers tables and samples column metadata from production databases.
Falls back to mock fixtures when cml.data_v1 is unavailable (devenv/CI).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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

    Falls back to mock fixture table names when cml.data_v1 is unavailable.
    """
    try:
        import cml.data_v1 as cmldata

        if connection_name is None:
            names = cfg.cml_data_connection_names
            if not names:
                raise ValueError("No data connections configured")
            connection_name = names[0]

        conn = cmldata.get_connection(connection_name)
        df = conn.get_pandas_dataframe(f"SHOW TABLES IN {database}")
        tables = df.iloc[:, 0].tolist()[:limit]
        return [str(t) for t in tables]
    except (ImportError, Exception):
        return _mock_table_names()


def sample_table_metadata(
    cfg,
    table_name: str,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int = 50,
    column_sample_limit: int = 1000,
) -> TableSample:
    """Sample column metadata from a hive table.

    Falls back to mock fixtures when cml.data_v1 is unavailable.
    """
    try:
        import cml.data_v1 as cmldata

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
    except (ImportError, Exception):
        return _mock_table_sample(table_name)


def load_annotations_from_hive(
    cfg,
    connection_name: str | None = None,
) -> list[dict]:
    """Load annotation records from default.annotations via hive.

    Falls back to mock annotations when unavailable.
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
        return df.to_dict("records")
    except (ImportError, Exception):
        return _load_mock_annotation_records()


# ── Mock data ────────────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _mock_table_names() -> list[str]:
    """Return mock table names from fixtures."""
    tables = _load_mock_tables()
    return [t["table_name"] for t in tables]


def _mock_table_sample(table_name: str) -> TableSample:
    """Return mock table sample from fixtures."""
    tables = _load_mock_tables()
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


def _load_mock_tables() -> list[dict]:
    """Load mock tables from fixtures."""
    path = _FIXTURES_DIR / "mock_tables.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _load_mock_annotation_records() -> list[dict]:
    """Load mock annotation records from fixtures."""
    path = _FIXTURES_DIR / "mock_annotations.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_all_mock_samples() -> list[TableSample]:
    """Load all mock table samples for testing."""
    tables = _load_mock_tables()
    return [_mock_table_sample(t["table_name"]) for t in tables]
