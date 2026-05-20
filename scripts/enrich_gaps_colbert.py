#!/usr/bin/env python
"""Fill enrichment gaps in the annotations_default_v2_colbert collection.

Idempotent: scrolls the existing collection, identifies codes NOT yet
present, runs the LLM enrichment loop for those only, embeds via the
ColBERT encoder, and upserts into the existing collection.

Usage:
    python scripts/enrich_gaps_colbert.py

Safe to re-run — already-enriched codes are skipped at the pre-filter
stage (no wasted LLM calls).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("enrich_gaps_colbert")

COLLECTION = "annotations_default_v2_colbert"
TAXONOMY_ID = "default"
AUGMENTATION_VERSION = "v2_colbert"
EMBEDDING_MODEL = "colbert-ir/colbertv2.0"
EMBEDDING_DIM = 128
SOURCE_JSON = Path("build/enrichment/full_295_source.json")


def _get_enriched_codes(client) -> set[str]:
    """Scroll existing collection and return set of enriched mnemonics/codes."""
    enriched = set()
    offset = None
    while True:
        points, next_offset = client.scroll(
            COLLECTION, limit=100, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for p in points:
            code = (p.payload or {}).get("mnemonic") or (p.payload or {}).get("code")
            if code:
                enriched.add(code)
        if next_offset is None:
            break
        offset = next_offset
    return enriched


def main() -> int:
    from qdrant_client import QdrantClient

    from atelier.classify.colbert_encoder import get_encoder
    from atelier.config import load_config
    from atelier.enrichment.llm_generator import ClassifyBackedEnrichmentGenerator
    from atelier.enrichment.loop import EnrichmentLoopConfig, run_enrichment

    # 1. Load full source taxonomy (295 rows — needed for tree context)
    if not SOURCE_JSON.exists():
        logger.error("Source JSON not found: %s", SOURCE_JSON)
        return 1
    with open(SOURCE_JSON) as f:
        all_source_rows = json.load(f)
    logger.info("Loaded %d source rows from %s", len(all_source_rows), SOURCE_JSON)

    # 2. Connect to Qdrant and identify already-enriched codes
    client = QdrantClient(url="http://127.0.0.1:6333", check_compatibility=False)
    enriched_codes = _get_enriched_codes(client)
    logger.info("Already enriched: %d codes", len(enriched_codes))

    # 3. Filter to only missing rows
    missing_rows = [
        r for r in all_source_rows
        if (r.get("mnemonic") or r.get("code")) not in enriched_codes
    ]
    if not missing_rows:
        logger.info("100%% coverage — nothing to enrich. Done.")
        return 0
    logger.info("Missing: %d codes — will enrich these", len(missing_rows))
    for r in missing_rows[:10]:
        logger.info("  %s: %s", r.get("mnemonic") or r.get("code"), r.get("label"))
    if len(missing_rows) > 10:
        logger.info("  ... and %d more", len(missing_rows) - 10)

    # 4. Build generator (LLM backend, same as classify pipeline)
    cfg = load_config()
    generator = ClassifyBackedEnrichmentGenerator(cfg)
    logger.info("Generator: %s", generator.name)

    # 5. Build ColBERT embedder
    encoder = get_encoder()
    logger.info("ColBERT encoder ready: dim=%d", encoder.dim)

    def embed_fn(text):
        """ColBERT encode_single compatible with EmbedFn signature."""
        return encoder.encode_single(text)

    # 6. Build valid_tag_codes from full source (for anti-example verification)
    valid_tag_codes = {
        r.get("code") or r.get("mnemonic")
        for r in all_source_rows
        if r.get("code") or r.get("mnemonic")
    }

    # 7. Build expected_parent_path lookup from full source
    by_code = {r.get("code"): r for r in all_source_rows if r.get("code")}

    def expected_parent_path(row: dict) -> list[str] | None:
        chain: list[str] = []
        cur = row
        for _ in range(32):
            chain.append(cur.get("label") or cur.get("mnemonic") or "")
            parent_code = cur.get("parent_code")
            if not parent_code or parent_code not in by_code:
                break
            cur = by_code[parent_code]
        return list(reversed(chain))

    # 8. Run enrichment loop on MISSING rows only, but passing all_source_rows
    #    would give better tree context. However, the loop iterates source_rows
    #    directly — so we pass missing_rows but set skip_on_cache_hit=False
    #    (we already pre-filtered; no need for Qdrant cache checks).
    config = EnrichmentLoopConfig(
        taxonomy_id=TAXONOMY_ID,
        augmentation_version=AUGMENTATION_VERSION,
        embedding_model=EMBEDDING_MODEL,
        embedding_dim=EMBEDDING_DIM,
        qdrant_url="http://127.0.0.1:6333",
        max_attempts_per_row=3,
        skip_on_cache_hit=False,  # We pre-filtered — skip the per-row check
        recreate_collection=False,  # Append to existing collection
        dry_run=False,
    )

    # NOTE: We pass missing_rows as source_rows. The tree context (siblings,
    # children) will only reflect the 104 missing codes amongst themselves.
    # For better context, we monkey-patch the source_rows used for tree
    # building but iterate only missing codes. Unfortunately the loop
    # doesn't separate "iterate set" from "context set". Acceptable trade-off:
    # the LLM generator has the label/code to work with regardless.
    #
    # WORKAROUND: We'll inject ALL source rows into the loop but mark
    # already-enriched ones so the loop can skip them efficiently.
    # Since skip_on_cache_hit=False AND the loop doesn't have a "skip list",
    # we need to pass only missing rows. The context quality is slightly
    # degraded but the labels alone provide sufficient signal.

    result = run_enrichment(
        source_rows=missing_rows,
        generator=generator,
        embed=embed_fn,
        config=config,
        qdrant_client=client,
        valid_tag_codes=valid_tag_codes,
        expected_parent_path_for=expected_parent_path,
    )

    # 9. Report
    print(json.dumps({
        "collection": result.collection_name,
        "taxonomy_version_hash": result.taxonomy_version_hash,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "rows_total": len(result.rows),
        "counts": result.counts,
    }, indent=2))

    # 10. Check final coverage
    final_enriched = _get_enriched_codes(client)
    all_codes = {
        r.get("mnemonic") or r.get("code")
        for r in all_source_rows
        if r.get("mnemonic") or r.get("code")
    }
    still_missing = all_codes - final_enriched
    logger.info(
        "Final coverage: %d/%d (%.1f%%) — %d still missing",
        len(final_enriched), len(all_codes),
        len(final_enriched) / len(all_codes) * 100 if all_codes else 0,
        len(still_missing),
    )
    if still_missing:
        for m in sorted(still_missing)[:20]:
            logger.warning("  Still missing: %s", m)

    # Non-zero exit on failures
    if result.counts.get("generator_failed", 0) > 0:
        return 2
    if result.counts.get("verifier_failed", 0) > 0:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
