"""Extract classification inputs for a hive-poc/synth snapshot.

Writes one JSON file per table under
``build/snapshots/hive-poc__synth/input/{table}.json`` containing the
column schema plus a small sample of rows.  Also stamps a copy of the
Atelier ICE ontology as ``vocabulary.json`` and a ``manifest.json``
seed with every table marked ``pending``.

No LLM calls; no mutation of live pipeline state.  Idempotent — tables
already extracted are skipped unless ``--force`` is passed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP_ROOT = REPO_ROOT / "build" / "snapshots" / "hive-poc__synth"
ONTOLOGY = REPO_ROOT / "data" / "sample" / "ontology.json"

DEFAULT_CONNECTION = "hive-poc"
DEFAULT_DATABASE = "synth"
DEFAULT_SAMPLE_ROWS = 50


def _sanitize(cell):
    if cell is None:
        return None
    if isinstance(cell, (bytes, bytearray)):
        try:
            return cell.decode("utf-8", errors="replace")
        except Exception:
            return repr(cell)
    if isinstance(cell, (int, float, bool, str)):
        return cell
    return str(cell)


def _describe(cursor, db: str, table: str) -> list[dict]:
    cursor.execute(f"DESCRIBE {db}.{table}")
    cols = []
    for row in cursor.fetchall():
        name = row[0]
        if not name or name.startswith("#") or name == "":
            break  # DESCRIBE returns partition-info headers after the schema
        col_type = row[1] if len(row) > 1 else ""
        cols.append({"name": name, "type": col_type})
    return cols


def _sample(cursor, db: str, table: str, n: int) -> list[list]:
    cursor.execute(f"SELECT * FROM {db}.{table} LIMIT {n}")
    return [[_sanitize(c) for c in row] for row in cursor.fetchall()]


def extract_table(cursor, db: str, table: str, n_rows: int) -> dict:
    cols = _describe(cursor, db, table)
    rows = _sample(cursor, db, table, n_rows)
    # Transpose to per-column sample lists for readability in the prompt.
    per_col_samples: dict[str, list] = {c["name"]: [] for c in cols}
    for row in rows:
        for i, c in enumerate(cols):
            if i < len(row):
                per_col_samples[c["name"]].append(row[i])
    for c in cols:
        c["samples"] = per_col_samples[c["name"]]
    return {
        "database": db,
        "table": table,
        "qualified_name": f"{db}.{table}",
        "row_sample_size": n_rows,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": cols,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default=DEFAULT_CONNECTION)
    ap.add_argument("--database", default=DEFAULT_DATABASE)
    ap.add_argument("--rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    ap.add_argument("--limit", type=int, default=0, help="Cap tables extracted (0=all)")
    ap.add_argument("--force", action="store_true", help="Re-extract even if input exists")
    ap.add_argument("--only", default="", help="Comma list of tables to extract")
    args = ap.parse_args()

    import cml.data_v1 as cmldata  # noqa: WPS433

    SNAP_ROOT.mkdir(parents=True, exist_ok=True)
    (SNAP_ROOT / "input").mkdir(exist_ok=True)
    (SNAP_ROOT / "output").mkdir(exist_ok=True)

    # Vocabulary stamp
    vocab_dst = SNAP_ROOT / "vocabulary.json"
    if args.force or not vocab_dst.exists():
        shutil.copy(ONTOLOGY, vocab_dst)
        print(f"[vocab] wrote {vocab_dst}")

    conn = cmldata.get_connection(args.connection)
    cur = conn.get_cursor()
    cur.execute(f"SHOW TABLES IN {args.database}")
    all_tables = [r[0] for r in cur.fetchall()]
    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        tables = [t for t in all_tables if t in wanted]
    else:
        tables = all_tables[: args.limit] if args.limit else all_tables

    print(f"[tables] {len(tables)} to extract (of {len(all_tables)} in {args.database})")

    manifest_path = SNAP_ROOT / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {
            "database": args.database,
            "connection": args.connection,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tables": {},
        }

    extracted = skipped = failed = 0
    t0 = time.time()
    for t in tables:
        out = SNAP_ROOT / "input" / f"{t}.json"
        if out.exists() and not args.force:
            skipped += 1
            manifest["tables"].setdefault(t, {"status": "pending", "attempts": 0})
            continue
        try:
            data = extract_table(cur, args.database, t, args.rows)
            out.write_text(json.dumps(data, indent=2, default=str) + "\n")
            manifest["tables"][t] = {
                "status": "pending",
                "attempts": 0,
                "columns": len(data["columns"]),
                "input_path": str(out.relative_to(REPO_ROOT)),
            }
            extracted += 1
            print(f"  [{extracted:3d}] {t}: {len(data['columns'])} cols, {len(data['columns'][0]['samples']) if data['columns'] else 0} rows")
        except Exception as e:
            failed += 1
            manifest["tables"][t] = {"status": "extract_failed", "error": str(e)[:300]}
            print(f"  [FAIL] {t}: {e}", file=sys.stderr)

    manifest["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    dt = time.time() - t0
    print(f"[done] extracted={extracted} skipped={skipped} failed={failed} in {dt:.1f}s")
    print(f"[snap] {SNAP_ROOT}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
