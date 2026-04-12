"""Synthetic data generation stub.

M0: Provides the interface and mock fixture loading.
M1: Will use Claude Agent SDK to generate deterministic procedural
    generators for synthetic training tables.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_synth_tables(
    category_set,
    output_dir: str | Path,
    *,
    tables_per_category: int = 2,
    columns_per_table: int = 50,
    rows_per_table: int = 100,
) -> list[dict[str, Any]]:
    """Generate synthetic training tables with known ground truth.

    M0 stub — returns mock fixture data rather than generating new tables.
    M1 will implement deterministic procedural generators driven by
    Claude Agent SDK keystone agents.

    Args:
        category_set: Reference categories for ground truth labels.
        output_dir: Directory to write generated CSV/parquet files.
        tables_per_category: Number of synthetic tables per category.
        columns_per_table: Target columns per table.
        rows_per_table: Target rows per table.

    Returns:
        List of generated table metadata dicts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # M0: Return the mock tables as "generated" data
    from atelier.classify.sampler import load_all_mock_samples

    results = []
    for table_sample in load_all_mock_samples():
        results.append({
            "table_name": table_sample.name,
            "database": "synth",
            "column_count": len(table_sample.columns),
            "row_count": max(c.total_count for c in table_sample.columns) if table_sample.columns else 0,
            "ground_truth": {
                c.name: c.ground_truth
                for c in table_sample.columns
                if c.ground_truth
            },
        })

    return results
