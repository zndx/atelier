"""Generic filesystem data source — a directory of CSV tables.

Registered data sources with a ``file://`` ``source_uri`` (the SDG
sample, exported snapshots, ad-hoc CSV drops) load through this
module instead of the Hive/``cml.data_v1`` path.  Unlike
``meta_tagging_source`` this loader is corpus-agnostic: no paired
reference-column convention, no name→code heuristics — just tables,
columns, and sampled values in the canonical bare-name form
(``ColumnSample`` invariant: names and siblings are table-relative).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from atelier.classify.sampler import ColumnSample, TableSample

logger = logging.getLogger(__name__)


def load_filesystem_source(
    mount: Path | str,
    *,
    sample_size: int = 50,
    database: str = "",
) -> list[TableSample]:
    """Load every ``<mount>/*.csv`` as a :class:`TableSample`.

    ``annotations.csv`` is skipped when present (vocabulary, not
    data).  Raises ``RuntimeError`` when the mount is missing or
    holds no data tables — a registered filesystem source with no
    tables is a broken deployment, not an empty result.
    """
    mount = Path(mount)
    if not mount.is_dir():
        raise RuntimeError(
            f"Filesystem source mount {mount} is not a directory.  "
            f"Re-register the data source or rebuild the sample "
            f"(just sdg-sample)."
        )

    samples: list[TableSample] = []
    for csv_path in sorted(mount.glob("*.csv")):
        if csv_path.name == "annotations.csv":
            continue
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                logger.warning("Skipping empty CSV %s", csv_path.name)
                continue
            rows = [r for _, r in zip(range(sample_size * 4), reader)]

        table_name = csv_path.stem
        columns: list[ColumnSample] = []
        for idx, col_name in enumerate(header):
            raw = [r[idx] if idx < len(r) else "" for r in rows]
            non_null = [v for v in raw if v not in ("", None)]
            columns.append(ColumnSample(
                name=col_name,
                column_type=None,  # CSV carries no type metadata
                values=non_null[:10],
                all_values=non_null[:sample_size],
                total_count=len(raw),
                null_count=len(raw) - len(non_null),
                table_name=table_name,
                database=database,
                siblings=[h for h in header if h != col_name],
            ))
        samples.append(TableSample(
            name=table_name, database=database, columns=columns,
        ))

    if not samples:
        raise RuntimeError(
            f"Filesystem source mount {mount} contains no data CSVs."
        )
    logger.info(
        "Filesystem source: %d tables / %d columns from %s",
        len(samples), sum(len(t.columns) for t in samples), mount,
    )
    return samples
