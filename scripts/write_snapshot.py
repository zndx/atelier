#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Write a classification snapshot dataset from a pre-tagged CSV.

Usage:
    python scripts/write_snapshot.py <tagged_csv> [--source hive-poc/synth]

Input CSV schema (any of these forms):
    table_name,column_name,code                  (code or abbrev)
    table_name,column_name,code,confidence
    table_name,column_name,code,confidence,column_type

The script resolves codes against the annotations vocabulary at
``build/data/annotations/annotations.json``, producing a partitioned
Arrow/Parquet dataset in ``build/snapshots/{source_slug}/``.

Output schema mirrors ``classifications.json`` records closely enough
that a future Atelier version can ingest them as a "warm start" for
the pipeline — pre-filled LLM labels with synthetic DST evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_annotations(path: Path | None = None) -> dict:
    """Load annotations vocab → {code: record, abbrev: record}."""
    path = path or PROJECT_ROOT / "build" / "data" / "annotations" / "annotations.json"
    with open(path) as f:
        records = json.load(f)
    lookup = {}
    for r in records:
        lookup[r["code"]] = r
        if r.get("abbrev"):
            lookup[r["abbrev"]] = r
    return lookup


def read_tagged_csv(csv_path: Path) -> list[dict]:
    """Read the input CSV, tolerating various column layouts."""
    import csv

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in reader.fieldnames]
        for raw in reader:
            # Normalise keys
            row = {h.strip().lower(): v.strip() for h, v in raw.items()}
            rows.append(row)
    return rows


def build_snapshot_records(
    rows: list[dict],
    vocab: dict,
    source_id: str,
) -> list[dict]:
    """Map tagged CSV rows → classification-schema records."""
    records = []
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    unresolved = set()

    for row in rows:
        table = row.get("table_name", row.get("table", ""))
        column = row.get("column_name", row.get("column", ""))
        code_raw = row.get("code", row.get("predicted_code", row.get("annotation", "")))
        confidence = float(row.get("confidence", 0.95))
        column_type = row.get("column_type", row.get("type", ""))

        if not table or not column or not code_raw:
            continue

        # Resolve against vocab (try code first, then abbrev)
        ann = vocab.get(code_raw)
        if ann is None:
            # Try case-insensitive abbrev match
            for k, v in vocab.items():
                if k.upper() == code_raw.upper():
                    ann = v
                    break
        if ann is None:
            unresolved.add(code_raw)
            # Still emit the record with what we have
            ann = {"code": code_raw, "label": code_raw, "abbrev": code_raw}

        code = ann["code"]
        label = ann.get("label", "")
        abbrev = ann.get("abbrev", "")

        rec = {
            "table_name": table,
            "column_name": column,
            "column_type": column_type,
            "predicted_code": code,
            "predicted_label": label,
            "predicted_annotation": abbrev,
            "confidence": confidence,
            "belief": confidence,
            "plausibility": min(confidence + 0.02, 1.0),
            "uncertainty": round(1.0 - confidence, 4),
            "conflict": 0.0,  # External tag — no DST conflict
            "needs_clarification": False,
            "evidence": f"snapshot(source={source_id})",
            "evidence_sources": {"snapshot": {code: confidence}},
            "embedding_text": "",
            "pattern_signals": {},
            "belief_path": [
                {"code": code, "label": label, "bel": confidence,
                 "pl": min(confidence + 0.02, 1.0), "depth": code.count(".") + 1}
            ],
            "cautious_code": code,
            "reference_code": None,
            "reference_label": "",
            "matches_reference": None,
            "llm_code": code,
            "llm_confidence": confidence,
            "shap_top1_name": "",
            "shap_top1_value": 0.0,
            "shap_top2_name": "",
            "shap_top2_value": 0.0,
            "shap_top3_name": "",
            "shap_top3_value": 0.0,
            # Snapshot metadata
            "snapshot_source": source_id,
            "snapshot_timestamp": ts,
        }
        records.append(rec)

    if unresolved:
        print(f"⚠  {len(unresolved)} codes not in vocab (emitted as-is): "
              f"{sorted(unresolved)[:10]}", file=sys.stderr)

    return records


def write_parquet_dataset(
    records: list[dict],
    output_dir: Path,
) -> Path:
    """Write records as a partitioned Arrow dataset (one fragment per table)."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    if not records:
        raise ValueError("No records to write")

    # Build schema — keep it explicit for downstream consumers
    schema = pa.schema([
        ("table_name", pa.utf8()),
        ("column_name", pa.utf8()),
        ("column_type", pa.utf8()),
        ("predicted_code", pa.utf8()),
        ("predicted_label", pa.utf8()),
        ("predicted_annotation", pa.utf8()),
        ("confidence", pa.float64()),
        ("belief", pa.float64()),
        ("plausibility", pa.float64()),
        ("uncertainty", pa.float64()),
        ("conflict", pa.float64()),
        ("needs_clarification", pa.bool_()),
        ("evidence", pa.utf8()),
        ("evidence_sources", pa.utf8()),   # JSON-serialised
        ("embedding_text", pa.utf8()),
        ("pattern_signals", pa.utf8()),     # JSON-serialised
        ("belief_path", pa.utf8()),         # JSON-serialised
        ("cautious_code", pa.utf8()),
        ("reference_code", pa.utf8()),
        ("reference_label", pa.utf8()),
        ("matches_reference", pa.bool_()),
        ("llm_code", pa.utf8()),
        ("llm_confidence", pa.float64()),
        ("shap_top1_name", pa.utf8()),
        ("shap_top1_value", pa.float64()),
        ("shap_top2_name", pa.utf8()),
        ("shap_top2_value", pa.float64()),
        ("shap_top3_name", pa.utf8()),
        ("shap_top3_value", pa.float64()),
        ("snapshot_source", pa.utf8()),
        ("snapshot_timestamp", pa.utf8()),
    ])

    # Flatten nested dicts to JSON strings for parquet storage
    for r in records:
        r["evidence_sources"] = json.dumps(r["evidence_sources"])
        r["pattern_signals"] = json.dumps(r["pattern_signals"])
        r["belief_path"] = json.dumps(r["belief_path"])
        # Normalise None → ""
        if r["reference_code"] is None:
            r["reference_code"] = ""
        if r["matches_reference"] is None:
            r["matches_reference"] = False

    table = pa.Table.from_pylist(records, schema=schema)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Write partitioned by table_name for fast per-table reads
    ds.write_dataset(
        table,
        output_dir,
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema([("table_name", pa.utf8())]),
            flavor="hive",
        ),
        existing_data_behavior="overwrite_or_ignore",
    )

    # Also write a single combined file for convenience
    combined = output_dir / "_combined.parquet"
    pq.write_table(table, combined, compression="zstd")

    return output_dir


def write_manifest(output_dir: Path, source_id: str, record_count: int):
    """Write a JSON manifest alongside the dataset."""
    manifest = {
        "format": "atelier-snapshot-v1",
        "source_id": source_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "record_count": record_count,
        "partitioning": "hive(table_name)",
        "compression": "zstd",
        "schema_version": "classifications-v1",
        "notes": (
            "Pre-tagged snapshot for warm-start ingestion. "
            "DST evidence fields are synthetic (single-source snapshot mass). "
            "Suitable for label grafting into a live pipeline run."
        ),
    }
    path = output_dir / "_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Tagged CSV file")
    parser.add_argument("--source", default="hive-poc/synth",
                        help="Source identifier (default: hive-poc/synth)")
    parser.add_argument("--vocab", type=Path, default=None,
                        help="Annotations JSON (default: build/data/annotations/annotations.json)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: build/snapshots/{source_slug}/)")
    args = parser.parse_args()

    source_slug = args.source.replace("/", "__")
    output_dir = args.output or PROJECT_ROOT / "build" / "snapshots" / source_slug

    print(f"Loading vocabulary...")
    vocab = load_annotations(args.vocab)
    print(f"  {len(vocab)} entries (codes + abbrevs)")

    print(f"Reading tagged CSV: {args.csv}")
    rows = read_tagged_csv(args.csv)
    print(f"  {len(rows)} rows")

    print(f"Building snapshot records (source={args.source})...")
    records = build_snapshot_records(rows, vocab, args.source)
    print(f"  {len(records)} classification records")

    tables = sorted(set(r["table_name"] for r in records))
    print(f"  {len(tables)} tables")

    print(f"Writing Arrow dataset → {output_dir}/")
    write_parquet_dataset(records, output_dir)

    write_manifest(output_dir, args.source, len(records))
    print(f"  manifest written")

    # Summary
    print()
    print(f"Snapshot ready:")
    print(f"  {output_dir}/_combined.parquet  ({len(records)} records)")
    print(f"  {output_dir}/_manifest.json")
    print(f"  {output_dir}/table_name=*/  (partitioned fragments)")


if __name__ == "__main__":
    main()
