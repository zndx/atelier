#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""CLI for the late-interaction annotation enrichment pipeline.

Reads a source taxonomy (a JSON file or, when ``--from-pglite`` is set,
the active ``default.annotations`` table in PGlite), runs the enrichment
loop, writes the result to a Qdrant collection, and updates the
``taxonomy_registry`` row to point at the new collection.

See ``docs/src/architecture/late-interaction-cosine.md`` for the design.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path


def _load_source_taxonomy(path: Path) -> list[dict]:
    """Load source taxonomy rows from a JSON file.

    Expected shape: a list of dicts with ``code``, ``label``, ``mnemonic``,
    ``description``, ``parent_code`` fields.  Other fields are passed
    through.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array of taxonomy rows in {path}")
    return data


def _load_from_connection(
    *,
    connection_name: str,
    database: str,
    table_name: str = "annotations",
    limit: int | None = None,
) -> list[dict]:
    """Load the active source-taxonomy from a CAI Data connection.

    Thin wrapper over
    :func:`atelier.enrichment.source_loader.load_from_connection` so
    the script-level CLI can stay as a simple imperative composition.
    """
    from atelier.enrichment.source_loader import load_from_connection
    return load_from_connection(
        connection_name=connection_name,
        database=database,
        table_name=table_name,
        limit=limit,
    )


def _build_embedder(model_name: str):
    """Build a sentence-transformers embedder callable.

    Returns a function that maps str | list[str] to embedding vectors.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(text):
        if isinstance(text, list):
            return [list(v) for v in model.encode(text, convert_to_numpy=True)]
        return list(model.encode(text, convert_to_numpy=True))

    return embed, model.get_sentence_embedding_dimension()


def _build_qdrant_client(qdrant_url: str | None):
    """Construct a Qdrant client.  ``None`` URL → in-process attempt."""
    from qdrant_client import QdrantClient

    if qdrant_url:
        return QdrantClient(url=qdrant_url)
    return QdrantClient(url="http://127.0.0.1:6333")


def _register_in_pglite(*, taxonomy_id: str, source_table: str,
                        qdrant_collection: str, qdrant_url: str | None,
                        augmentation_version: str, embedding_model: str,
                        embedding_dim: int, summary: str) -> str:
    """Register the built collection in PGlite ``taxonomy_registry``.

    Returns the registry row id.  Deferred: import the DAO only when
    needed so the script's dry-run path doesn't require a live DB.
    """
    from atelier.db.dao import AtelierDao

    dao = AtelierDao()
    row_id = str(uuid.uuid4())
    dao.register_taxonomy_collection(
        id=row_id,
        taxonomy_id=taxonomy_id,
        source_table=source_table,
        qdrant_collection=qdrant_collection,
        qdrant_url=qdrant_url,
        augmentation_version=augmentation_version,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        summary=summary,
        status="building",
    )
    return row_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--taxonomy-id", required=True,
        help='Logical taxonomy identifier (e.g., "default").',
    )
    parser.add_argument(
        "--augmentation-version", required=True,
        help="Prompt + verifier function version label (e.g., 'v1').",
    )
    parser.add_argument(
        "--source-json", type=Path,
        help="Path to a JSON file containing the source taxonomy rows.",
    )
    parser.add_argument(
        "--from-connection", default=None,
        help=(
            "Load source taxonomy from a CAI Data connection (Hive/Impala "
            "via cml.data_v1).  Pass the connection name and use "
            "--database to specify the schema."
        ),
    )
    parser.add_argument(
        "--database", default="default",
        help="Database/schema name when using --from-connection.",
    )
    parser.add_argument(
        "--annotations-table", default="annotations",
        help="Annotations table name when using --from-connection.",
    )
    parser.add_argument(
        "--row-limit", type=int, default=None,
        help=(
            "Cap source rows (useful for smoke tests).  Default is no "
            "limit — production runs enrich the full table."
        ),
    )
    parser.add_argument(
        "--source-table", default="default.annotations",
        help=(
            "Source table label recorded in the registry.  Auto-derived "
            "from --database/--annotations-table when --from-connection "
            "is used."
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers model identifier.",
    )
    parser.add_argument(
        "--qdrant-url", default=None,
        help="Qdrant URL (default: http://127.0.0.1:6333).",
    )
    parser.add_argument(
        "--generator", choices=["backend", "stub"], default="backend",
        help=(
            "Enrichment generator implementation.  'backend' (default) uses "
            "the same LLM backend as the classify pipeline (provider co-"
            "location, apex reasoning model per provider — see "
            "atelier.enrichment.model_resolver).  'stub' is the "
            "deterministic stub for test/CI runs that don't spend on real "
            "LLM calls."
        ),
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="Max LLM-generation attempts per row before giving up.",
    )
    parser.add_argument(
        "--recreate-collection", action="store_true",
        help="Drop and recreate the Qdrant collection before writing.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip Qdrant writes and PGlite registry updates; report what would happen.",
    )
    parser.add_argument(
        "--no-skip-cache", action="store_true",
        help="Re-enrich every row even when its content-addressed point already exists.",
    )
    parser.add_argument(
        "--checkpoint-path", default=None,
        help=(
            "Per-row JSONL checkpoint path.  Default auto-derives to "
            "build/enrichment/{collection}-{started_at}.jsonl so a long "
            "run is inspectable mid-flight via tail -f.  Pass an empty "
            "string to disable explicitly (not recommended for "
            "production runs)."
        ),
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
    logger = logging.getLogger("enrich_annotations")

    # 1. Load source taxonomy.  Two paths:
    #    --source-json:    pre-exported JSON snapshot (offline/dev/CI)
    #    --from-connection: live read via cml.data_v1 on a CAI runtime
    if args.source_json:
        source_rows = _load_source_taxonomy(args.source_json)
        logger.info(
            "Loaded %d source rows from %s",
            len(source_rows), args.source_json,
        )
    elif args.from_connection:
        source_rows = _load_from_connection(
            connection_name=args.from_connection,
            database=args.database,
            table_name=args.annotations_table,
            limit=args.row_limit,
        )
        if args.source_table == "default.annotations":
            # Auto-derive a meaningful registry label from the connection
            # path so a hive-poc default run produces
            # "hive-poc/default.annotations" rather than the placeholder.
            args.source_table = (
                f"{args.from_connection}/{args.database}.{args.annotations_table}"
            )
        logger.info(
            "Loaded %d source rows from %s",
            len(source_rows), args.source_table,
        )
    else:
        parser.error(
            "one of --source-json or --from-connection NAME is required"
        )

    valid_tag_codes = {
        r.get("code") or r.get("mnemonic") for r in source_rows if r.get("code") or r.get("mnemonic")
    }

    # 2. Build generator — default 'backend' uses the same LLM backend as
    #    the classify pipeline and the strongest reasoning model available
    #    on that provider.  See atelier.enrichment.model_resolver for the
    #    selection rule.  The stub remains for test/CI runs.
    from atelier.enrichment.llm_generator import (
        ClassifyBackedEnrichmentGenerator,
        DeterministicStubGenerator,
    )

    if args.generator == "stub":
        generator = DeterministicStubGenerator()
    else:
        from atelier.config import load_config
        cfg = load_config()
        generator = ClassifyBackedEnrichmentGenerator(cfg)
        logger.info(
            "Enrichment generator: %s (backend co-located with classify)",
            generator.name,
        )

    # 3. Build embedder
    embed_fn, embedding_dim = _build_embedder(args.embedding_model)
    logger.info("Embedding model: %s (dim=%d)", args.embedding_model, embedding_dim)

    # 4. Connect Qdrant (skipped in dry-run)
    qdrant_client = None if args.dry_run else _build_qdrant_client(args.qdrant_url)

    # 5. Build expected parent-path lookup for verification
    by_code = {r.get("code"): r for r in source_rows if r.get("code")}

    def expected_parent_path(row: dict) -> list[str] | None:
        chain: list[str] = []
        cur = row
        # Walk up to root; cap at 32 to avoid loops on malformed input.
        for _ in range(32):
            chain.append(cur.get("label") or cur.get("mnemonic") or "")
            parent_code = cur.get("parent_code")
            if not parent_code or parent_code not in by_code:
                break
            cur = by_code[parent_code]
        return list(reversed(chain))

    # 6. Run the loop
    from atelier.enrichment.loop import EnrichmentLoopConfig, run_enrichment

    config = EnrichmentLoopConfig(
        taxonomy_id=args.taxonomy_id,
        augmentation_version=args.augmentation_version,
        embedding_model=args.embedding_model,
        embedding_dim=embedding_dim,
        qdrant_url=args.qdrant_url,
        max_attempts_per_row=args.max_attempts,
        skip_on_cache_hit=not args.no_skip_cache,
        recreate_collection=args.recreate_collection,
        dry_run=args.dry_run,
        checkpoint_path=args.checkpoint_path,
    )

    result = run_enrichment(
        source_rows=source_rows,
        generator=generator,
        embed=embed_fn,
        config=config,
        qdrant_client=qdrant_client,
        valid_tag_codes=valid_tag_codes,
        expected_parent_path_for=expected_parent_path,
    )

    # 7. Register in PGlite (skipped in dry-run)
    if not args.dry_run:
        summary = (
            f"enriched={result.counts.get('enriched', 0)} "
            f"cache_hit={result.counts.get('cache_hit', 0)} "
            f"verifier_failed={result.counts.get('verifier_failed', 0)} "
            f"generator_failed={result.counts.get('generator_failed', 0)}"
        )
        try:
            row_id = _register_in_pglite(
                taxonomy_id=args.taxonomy_id,
                source_table=args.source_table,
                qdrant_collection=result.collection_name,
                qdrant_url=args.qdrant_url,
                augmentation_version=args.augmentation_version,
                embedding_model=args.embedding_model,
                embedding_dim=embedding_dim,
                summary=summary,
            )
            logger.info("Registered taxonomy collection in PGlite (id=%s)", row_id)
            from atelier.db.dao import AtelierDao
            if AtelierDao().set_current_taxonomy_collection(row_id):
                logger.info("Promoted taxonomy collection %s to 'current'", row_id)
            else:
                logger.warning("Failed to promote taxonomy collection %s", row_id)
        except Exception as exc:  # noqa: BLE001 — registration failure is non-fatal
            logger.warning("PGlite registration failed (non-fatal): %s", exc)

    # 8. Final summary
    print(json.dumps({
        "collection": result.collection_name,
        "taxonomy_version_hash": result.taxonomy_version_hash,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "rows_total": len(result.rows),
        "counts": result.counts,
    }, indent=2))

    # Non-zero exit when any row failed unrecoverably
    if result.counts.get("generator_failed", 0) > 0:
        return 2
    if result.counts.get("verifier_failed", 0) > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
