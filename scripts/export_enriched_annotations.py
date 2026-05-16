#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Export an enriched annotation collection to build/ for operator review.

Reads the Qdrant payload (vectors are skipped — the snapshot is for
human inspection, not similarity comparison) and writes parquet plus
a human-readable TSV alongside.

Files land in ``build/exports/`` named:

    <taxonomy_id>-enriched-<augmentation_version>-<utc>.parquet
    <taxonomy_id>-enriched-<augmentation_version>-<utc>.tsv

Snapshots are read-only — edits go back to Qdrant through a structured
CLI (deferred, P2 follow-on).  See
``docs/src/architecture/late-interaction-cosine.md`` § Operator
inspection and edit surface.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("export_enriched_annotations")


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _scroll_collection(client, collection: str, batch: int = 256):
    """Yield all payloads from a Qdrant collection.

    Vectors are deliberately not requested — the export is for
    inspection, and skipping vectors keeps the payload export light.
    """
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            yield p.payload or {}
        if next_offset is None:
            break


def _flatten_for_table(payload: dict) -> dict:
    """Flatten nested payload fields to scalar/CSV-friendly forms.

    Lists and dicts in non-key fields are JSON-encoded.  This makes the
    TSV grep-able while the parquet preserves typed structures.
    """
    flat = dict(payload)
    for k, v in list(flat.items()):
        if isinstance(v, (list, dict)):
            flat[k] = json.dumps(v, ensure_ascii=False)
    return flat


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection", required=True,
        help="Qdrant collection name to export.",
    )
    parser.add_argument(
        "--taxonomy-id", default=None,
        help='Logical taxonomy identifier; defaults to a slice of the collection name.',
    )
    parser.add_argument(
        "--augmentation-version", default=None,
        help="Augmentation version label; defaults to a slice of the collection name.",
    )
    parser.add_argument(
        "--qdrant-url", default="http://127.0.0.1:6333",
        help="Qdrant URL.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("build/exports"),
        help="Directory to write the export into (created if absent).",
    )
    parser.add_argument(
        "--no-tsv", action="store_true",
        help="Skip the TSV companion file (parquet only).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Infer taxonomy_id + augmentation_version from collection name when
    # the operator didn't supply them — the convention is
    # 'annotations_<taxonomy_id>_<augmentation_version>'.
    coll = args.collection
    taxonomy_id = args.taxonomy_id
    aug_version = args.augmentation_version
    if coll.startswith("annotations_"):
        rest = coll[len("annotations_"):]
        if "_" in rest and not taxonomy_id and not aug_version:
            taxonomy_id, _, aug_version = rest.rpartition("_")
    taxonomy_id = taxonomy_id or "unknown"
    aug_version = aug_version or "unknown"

    from qdrant_client import QdrantClient

    client = QdrantClient(url=args.qdrant_url)
    if not client.collection_exists(coll):
        logger.error("Qdrant collection %s does not exist at %s", coll, args.qdrant_url)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    utc = _utc_iso()
    parquet_path = args.output_dir / f"{taxonomy_id}-enriched-{aug_version}-{utc}.parquet"
    tsv_path = args.output_dir / f"{taxonomy_id}-enriched-{aug_version}-{utc}.tsv"

    rows: list[dict] = []
    for payload in _scroll_collection(client, coll):
        rows.append(_flatten_for_table(payload))
    logger.info("Read %d payload rows from %s", len(rows), coll)

    # Parquet first — preserves typing via pyarrow.
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    logger.info("Wrote parquet: %s", parquet_path)

    if not args.no_tsv:
        df.to_csv(tsv_path, sep="\t", index=False)
        logger.info("Wrote TSV: %s", tsv_path)

    print(json.dumps({
        "collection": coll,
        "rows_exported": len(rows),
        "parquet_path": str(parquet_path),
        "tsv_path": None if args.no_tsv else str(tsv_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
