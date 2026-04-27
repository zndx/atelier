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
    all_values: list[str] = field(default_factory=list)  # Full row reservoir (up to sample_size)
    total_count: int = 0
    null_count: int = 0
    table_name: str = ""
    database: str = ""
    siblings: list[str] = field(default_factory=list)
    reference_code: str | None = None  # Curated reference label (for validation)
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
            "reference_code": self.reference_code,
        }
        if self.distinct_count is not None:
            d["distinct_count"] = self.distinct_count
        if self.all_values and len(self.all_values) > len(self.values):
            d["all_values"] = self.all_values
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


def _run_with_timeout(callable_, *args, timeout_s: float, **kwargs):
    """Execute a blocking callable on a side thread with a timeout.

    ``cml.data_v1.Connection.get_pandas_dataframe`` doesn't accept a
    timeout parameter, so we wrap the call in
    ``concurrent.futures.ThreadPoolExecutor.submit().result(timeout=...)``.
    Used for the lightweight liveness and metadata probes (short fixed
    timeouts, fail-fast on broken connections); the heavy queries
    themselves are NOT timeout-wrapped — proof-of-progress paradigm
    means heavy queries run unbounded but observable via
    :func:`atelier.classify.phase_heartbeat.phase_heartbeat`.

    Raises ``TimeoutError`` (Python's builtin) on timeout — caller
    decides whether to log+continue or abort.
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="hive-probe",
    ) as pool:
        future = pool.submit(callable_, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(
                f"Hive probe exceeded {timeout_s:.1f}s timeout"
            ) from exc


def probe_connection(
    cfg,
    connection_name: str | None = None,
    *,
    timeout_s: float | None = None,
) -> bool:
    """Lightweight liveness probe — confirms the Hive connection responds.

    Runs ``SELECT 1`` with a short timeout (default 5s) before any
    expensive query.  Catches dead connections, expired credentials,
    or completely-blackholed endpoints fast — well before they manifest
    as a long pre-LLM stall.

    Returns True when the connection is live, False otherwise.  Logs
    the failure reason but does not raise — callers decide whether
    to abort the run or proceed (some pipelines may want to attempt
    the heavier query anyway, since liveness probes can be flaky).

    The 5s default is a true fail-fast budget; legitimate Hive
    connections respond in milliseconds.
    """
    if timeout_s is None:
        timeout_s = float(getattr(cfg, "classify_liveness_probe_timeout_s", 5.0))

    try:
        import cml.data_v1 as cmldata
    except ImportError:
        log.warning("cml.data_v1 not available; skipping liveness probe")
        return False

    if connection_name is None:
        names = cfg.cml_data_connection_names
        if not names:
            log.warning("No data connections configured; skipping liveness probe")
            return False
        connection_name = names[0]

    try:
        conn = cmldata.get_connection(connection_name)
        _run_with_timeout(
            conn.get_pandas_dataframe, "SELECT 1",
            timeout_s=timeout_s,
        )
        return True
    except TimeoutError as exc:
        log.warning(
            "Hive liveness probe TIMED OUT (%s) for %s — connection appears "
            "dead.  The pipeline will still attempt the heavy query, but "
            "operators should expect a longer-than-usual stall.",
            exc, connection_name,
        )
        return False
    except Exception as exc:
        log.warning(
            "Hive liveness probe FAILED for %s: %s — connection may be "
            "misconfigured or credentials may have expired.",
            connection_name, exc,
        )
        return False


def probe_table_shape(
    cfg,
    table_name: str,
    connection_name: str | None = None,
    database: str = "default",
    *,
    timeout_s: float | None = None,
) -> dict[str, Any] | None:
    """Cheap metadata probe — column count + types via DESCRIBE.

    ``DESCRIBE {db}.{table}`` returns column metadata only; the Hive
    metastore answers from cached schema, not from a data scan.  Use
    this BEFORE ``SELECT * FROM {table} LIMIT N`` to gauge cardinality
    and catch missing-table errors faster than the heavy query would.

    Returns ``{"columns": [...], "n_columns": int}`` on success, or
    ``None`` on timeout/failure (logged, non-fatal — the heavy query
    can still proceed; the probe is informational).

    Default timeout 10s — generous for a metadata call but strict
    enough to fail fast on a broken metastore.
    """
    if timeout_s is None:
        timeout_s = float(getattr(cfg, "classify_metadata_probe_timeout_s", 10.0))

    try:
        import cml.data_v1 as cmldata
    except ImportError:
        return None

    if connection_name is None:
        names = cfg.cml_data_connection_names
        if not names:
            return None
        connection_name = names[0]

    try:
        conn = cmldata.get_connection(connection_name)
        df = _run_with_timeout(
            conn.get_pandas_dataframe,
            f"DESCRIBE {database}.{table_name}",
            timeout_s=timeout_s,
        )
        # DESCRIBE schema is roughly (col_name, data_type, comment).
        cols = df.iloc[:, 0].tolist() if not df.empty else []
        return {"columns": [str(c) for c in cols], "n_columns": len(cols)}
    except TimeoutError as exc:
        log.warning(
            "Hive metadata probe (DESCRIBE %s.%s) timed out: %s",
            database, table_name, exc,
        )
        return None
    except Exception as exc:
        log.debug(
            "Hive metadata probe (DESCRIBE %s.%s) failed (non-fatal): %s",
            database, table_name, exc,
        )
        return None


def discover_tables(
    cfg,
    connection_name: str | None = None,
    database: str = "default",
    limit: int | None = None,
) -> list[str]:
    """List tables from a hive database via CAI Data Platform.

    When ``limit`` is None, falls back to ``cfg.classify_tables_limit``
    — the HOCON / env-configured value — rather than a hard-coded
    function default.  Raises RuntimeError when cml.data_v1 is
    unavailable.  For dev/test, inject samples= into the pipeline
    instead.
    """
    if limit is None:
        limit = cfg.classify_tables_limit

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
    if len(df) > limit:
        log.warning(
            "discover_tables: database %s has %d tables, truncated to %d "
            "(classify.tables_limit). Increase ATELIER_CLASSIFY_TABLES_LIMIT "
            "to classify all tables.",
            database, len(df), limit,
        )
    return [str(t) for t in tables]


def sample_table_metadata(
    cfg,
    table_name: str,
    connection_name: str | None = None,
    database: str = "default",
    sample_size: int | None = None,
    column_sample_limit: int | None = None,
) -> TableSample:
    """Sample column metadata from a hive table.

    Raises RuntimeError when cml.data_v1 is unavailable.
    For dev/test, inject samples= into the pipeline instead.

    When ``sample_size`` or ``column_sample_limit`` is None, the values
    fall through to ``cfg.classify_sample_size`` and
    ``cfg.classify_column_sample_limit`` respectively — the operator-
    configured HOCON / env values — rather than silently clamping to a
    hard-coded function default.
    """
    if sample_size is None:
        sample_size = cfg.classify_sample_size
    if column_sample_limit is None:
        column_sample_limit = cfg.classify_column_sample_limit

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
        all_vals = [str(v) for v in df[col_name].dropna().tolist()]
        col_values = all_vals[:5]
        null_count = int(df[col_name].isna().sum())
        total_count = len(df)
        col_type = str(df[col_name].dtype)

        columns.append(ColumnSample(
            name=col_name,
            column_type=col_type,
            values=col_values,
            all_values=all_vals,
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
                    all_values=c.get("values", []),
                    total_count=len(c.get("values", [])),
                    null_count=0,
                    table_name=table_name,
                    database=t.get("database", "default"),
                    siblings=col_names,
                    reference_code=c.get("reference_code"),
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

    Reads all CSVs in the sample tables directory and the curated-reference
    mapping. Returns TableSample objects with reference_code labels
    attached to each column — ready to inject into the classification
    pipeline.
    """
    import csv as csv_mod

    base = Path(sample_dir) if sample_dir else _SAMPLE_DIR
    tables_dir = base / "tables"
    ref_path = base / "reference_labels.json"

    if not tables_dir.is_dir():
        log.warning("Sample tables directory not found: %s", tables_dir)
        return []

    reference_labels: dict[str, str] = {}
    if ref_path.exists():
        with open(ref_path) as f:
            reference_labels = json.load(f)

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
            all_vals = [row[i] for row in rows if i < len(row) and row[i]]
            values = all_vals[:5]
            total_count = len(rows)
            null_count = sum(1 for row in rows if i >= len(row) or not row[i])
            gt_key = f"{table_name}.{col_name}"

            columns.append(ColumnSample(
                name=col_name,
                column_type="object",
                values=values,
                all_values=all_vals,
                total_count=total_count,
                null_count=null_count,
                table_name=table_name,
                database="sample",
                siblings=col_names,
                reference_code=reference_labels.get(gt_key),
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
    ref_path = base / "reference_labels.json"

    table_count = len(list(tables_dir.glob("*.csv"))) if tables_dir.is_dir() else 0
    column_count = 0
    if ref_path.exists():
        with open(ref_path) as f:
            column_count = len(json.load(f))

    return {
        "table_count": table_count,
        "column_count": column_count,
        "has_data": table_count > 0,
    }


# ── Synth source data (large-scale synthetic) ──────────────────────
#
# The synth corpus can be mounted either uncompressed at build/data/synth/
# or directly from the zip at build/atelier-synth-db.zip.  Both layouts
# share the same internal structure (data/synth/tables/*.csv +
# reference_labels.json); the zip uses that same prefix as its top-level
# entries.
#
# resolve_synth_mount() picks the faster option (directory) when present,
# falling back to the zip so operators can drop the artifact in and go.

_BUILD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "build"
_SYNTH_DIR = _BUILD_DIR / "data" / "synth"
_SYNTH_ZIP = _BUILD_DIR / "atelier-synth-db.zip"


def resolve_synth_mount() -> Path | None:
    """Return the path Atelier should mount for the OOTB synth source.

    Preference: uncompressed directory > zip archive > None.  Returning
    None means neither artifact is present and the synth source should
    be hidden from the UI.
    """
    if _SYNTH_DIR.is_dir() and (_SYNTH_DIR / "tables").is_dir():
        return _SYNTH_DIR
    if _SYNTH_ZIP.is_file():
        return _SYNTH_ZIP
    return None


def _iter_synth_csvs(mount: Path):
    """Yield ``(table_name, header, rows)`` from either a dir or zip mount.

    For a directory mount, ``mount/tables/*.csv`` is walked.  For a zip
    mount, entries under ``data/synth/tables/`` are streamed without
    extracting to disk.  Row ordering follows sorted filename order so
    runs are reproducible across mount types.
    """
    import csv as csv_mod
    import io
    import zipfile

    if mount.is_dir():
        tables_dir = mount / "tables"
        for csv_path in sorted(tables_dir.glob("*.csv")):
            with open(csv_path, newline="") as f:
                reader = csv_mod.reader(f)
                header = next(reader, None)
                if not header:
                    continue
                rows = list(reader)
            yield csv_path.stem, header, rows
        return

    if mount.suffix == ".zip":
        with zipfile.ZipFile(mount) as zf:
            names = sorted(
                n for n in zf.namelist()
                if n.startswith("data/synth/tables/") and n.endswith(".csv")
            )
            for name in names:
                with zf.open(name) as raw:
                    text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
                    reader = csv_mod.reader(text)
                    header = next(reader, None)
                    if not header:
                        continue
                    rows = list(reader)
                table_name = Path(name).stem
                yield table_name, header, rows
        return

    raise ValueError(f"Unsupported synth mount: {mount}")


def _read_synth_reference_labels(mount: Path) -> dict[str, str]:
    """Read the reference_labels.json sidecar from either a dir or zip mount."""
    import zipfile

    if mount.is_dir():
        ref_path = mount / "reference_labels.json"
        if ref_path.exists():
            with open(ref_path) as f:
                return json.load(f)
        return {}

    if mount.suffix == ".zip":
        with zipfile.ZipFile(mount) as zf:
            for name in (
                "data/synth/reference_labels.json",
                "reference_labels.json",
            ):
                if name in zf.namelist():
                    with zf.open(name) as f:
                        return json.loads(f.read().decode("utf-8"))
        return {}

    raise ValueError(f"Unsupported synth mount: {mount}")


def load_synth_source(
    synth_mount: str | Path | None = None,
) -> list[TableSample]:
    """Load synth tables from build/data/synth/ or build/atelier-synth-db.zip.

    Same contract as :func:`load_sample_source` but for the larger
    synthetic database (~100 tables, ~100 columns each).  ``synth_mount``
    may point at either the uncompressed directory or a ``.zip`` archive;
    when omitted, :func:`resolve_synth_mount` picks whichever is present.
    """
    mount = Path(synth_mount) if synth_mount else resolve_synth_mount()
    if mount is None:
        log.warning(
            "Synth mount not found; looked at %s and %s", _SYNTH_DIR, _SYNTH_ZIP,
        )
        return []

    reference_labels = _read_synth_reference_labels(mount)

    samples: list[TableSample] = []
    for table_name, header, rows in _iter_synth_csvs(mount):
        col_names = header
        columns: list[ColumnSample] = []
        for i, col_name in enumerate(col_names):
            all_vals = [row[i] for row in rows if i < len(row) and row[i]]
            values = all_vals[:5]
            total_count = len(rows)
            null_count = sum(1 for row in rows if i >= len(row) or not row[i])
            ref_key = f"{table_name}.{col_name}"

            columns.append(ColumnSample(
                name=col_name,
                column_type="object",
                values=values,
                all_values=all_vals,
                total_count=total_count,
                null_count=null_count,
                table_name=table_name,
                database="synth",
                siblings=col_names,
                reference_code=reference_labels.get(ref_key),
                distinct_count=len(set(row[i] for row in rows if i < len(row))),
            ))

        samples.append(TableSample(
            name=table_name,
            database="synth",
            columns=columns,
        ))

    log.info(
        "Loaded %d synth tables (%d columns) from %s",
        len(samples),
        sum(len(t.columns) for t in samples),
        mount,
    )
    return samples


def synth_source_stats(synth_mount: str | Path | None = None) -> dict:
    """Return summary stats for the synth source without loading all data."""
    import zipfile

    mount = Path(synth_mount) if synth_mount else resolve_synth_mount()
    if mount is None:
        return {"table_count": 0, "column_count": 0, "has_data": False, "mount": None}

    if mount.is_dir():
        tables_dir = mount / "tables"
        table_count = len(list(tables_dir.glob("*.csv")))
        mount_kind = "dir"
    elif mount.suffix == ".zip":
        with zipfile.ZipFile(mount) as zf:
            table_count = sum(
                1 for n in zf.namelist()
                if n.startswith("data/synth/tables/") and n.endswith(".csv")
            )
        mount_kind = "zip"
    else:
        return {"table_count": 0, "column_count": 0, "has_data": False, "mount": None}

    reference_labels = _read_synth_reference_labels(mount)
    return {
        "table_count": table_count,
        "column_count": len(reference_labels),
        "has_data": table_count > 0,
        "mount": str(mount),
        "mount_kind": mount_kind,
    }
