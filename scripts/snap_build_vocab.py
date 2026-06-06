"""Build vocabulary.json from the source's `annotations` table.

When the target database has an in-band taxonomy table (e.g.
``synth.annotations``), that table *is* the vocabulary — we ignore the
shipped ICE ontology.  This script pulls the full table via
``cml.data_v1`` and writes a leaf-filtered JSON array matching the
shape Atelier's ``load_annotations_from_hive`` produces downstream.

Also strips the source annotations table (and any other known
non-data tables) out of the manifest so workers don't try to
classify them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP_ROOT = REPO_ROOT / "build" / "snapshots" / "hive-poc__synth"

# Copied from pipeline.py:_NON_DATA_TABLE_NAMES so we stay aligned with
# how Atelier itself filters taxonomy / test-leftover tables.
NON_DATA_TABLES = {"annotations", "ice_t1"}


def _clean(s):
    if s is None:
        return ""
    return str(s).strip()


def _is_leaf(code: str, all_codes: set[str]) -> bool:
    """Legacy dot-notation leaf detection: a code is a leaf when no other
    code starts with ``code + "."``.  Mirrors the fallback branch in
    ``taxonomy._build_category_set_from_records``.
    """
    prefix = code + "."
    return not any(other != code and other.startswith(prefix) for other in all_codes)


def _fetch_annotations(connection: str, database: str) -> list[dict]:
    import cml.data_v1 as cmldata  # noqa: WPS433

    conn = cmldata.get_connection(connection)
    df = conn.get_pandas_dataframe(f"SELECT * FROM {database}.annotations")
    # Hive/Impala prefixes columns with ``{table}.`` on SELECT * — strip
    # that so downstream code can read ``r["id"]`` etc. uniformly.
    df.columns = [c.split(".", 1)[-1] if "." in c else c for c in df.columns]
    records = df.to_dict("records")
    return records


def build_vocabulary(records: list[dict]) -> list[dict]:
    """Return the leaf-only, cleaned vocabulary entries."""
    # Clean + collect codes
    cleaned = []
    for r in records:
        entry = {
            "id": _clean(r.get("id")),
            "annotation": _clean(r.get("annotation")),
            "ontology": _clean(r.get("ontology")),
            "definition": _clean(r.get("definition")),
            "common_names": _clean(r.get("common_names")),
            "non_corp": _clean(r.get("non_corp")),
            "emp_contractor": _clean(r.get("emp_contractor")),
            "individual": _clean(r.get("individual")),
            "corp": _clean(r.get("corp")),
            "deprecated": _clean(r.get("deprecated")).lower() or "no",
        }
        if not entry["id"]:
            continue
        cleaned.append(entry)

    all_ids = {e["id"] for e in cleaned}
    leaves = [e for e in cleaned if _is_leaf(e["id"], all_ids)]
    return leaves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connection", default="hive-poc")
    ap.add_argument("--database", default="synth")
    ap.add_argument("--include-deprecated", action="store_true",
                    help="Keep deprecated codes in the vocabulary (default: drop)")
    args = ap.parse_args()

    print(f"[fetch] {args.database}.annotations via {args.connection} …")
    records = _fetch_annotations(args.connection, args.database)
    print(f"[fetch] got {len(records)} rows")

    leaves = build_vocabulary(records)
    print(f"[leaves] {len(leaves)} of {len(records)} rows are leaf codes")

    if not args.include_deprecated:
        kept = [e for e in leaves if e["deprecated"] != "yes"]
        print(f"[filter] {len(leaves) - len(kept)} deprecated entries dropped → {len(kept)} final")
        leaves = kept

    out_path = SNAP_ROOT / "vocabulary.json"
    payload = {
        "source": f"{args.database}.annotations",
        "connection": args.connection,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_rows": len(records),
        "leaf_count": len(leaves),
        "entries": leaves,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"[write] {out_path} ({out_path.stat().st_size} bytes)")

    # Also purge non-data tables from manifest + input dir so the
    # orchestrator never tries to classify them.
    mpath = SNAP_ROOT / "manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        removed = []
        for name in list(m["tables"]):
            if name.lower() in NON_DATA_TABLES:
                m["tables"].pop(name)
                removed.append(name)
                ip = SNAP_ROOT / "input" / f"{name}.json"
                if ip.exists():
                    ip.unlink()
        if removed:
            m["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            m["vocab_source"] = f"{args.database}.annotations"
            mpath.write_text(json.dumps(m, indent=2, default=str) + "\n")
            print(f"[manifest] removed non-data tables: {removed}")
        else:
            print("[manifest] no non-data tables to remove")

    # Sanity preview
    sample_codes = [e["annotation"] for e in leaves[:15] if e["annotation"]]
    print(f"[preview] first 15 codes: {sample_codes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
