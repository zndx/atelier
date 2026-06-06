"""Enrichment payload loader for SVM corpus generation.

Abstracts the source of enrichment payloads: Qdrant (live, source of
truth) or a JSON export (under ``build/``, for offline/CI).  The SVM
training path always requires enrichment data — without it, the SVM
cannot emit user-provided terminology.

JSON format matches the dict shape already produced by ``/train-svm``
Phase 2 at ``build/data/svm_training/enrichment_payloads.json``::

    {code: {code, label, description, prototype_values,
            value_patterns, name_hints, anti_examples}}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_PAYLOAD_FIELDS = (
    "code", "label", "description",
    "prototype_values", "value_patterns", "name_hints", "anti_examples",
)


def load_enrichment_payloads(
    *,
    json_path: Path | None = None,
    cfg=None,
    taxonomy_id: str | None = None,
) -> dict[str, dict]:
    """Load enrichment payloads from JSON export or live Qdrant.

    Source priority:
      1. ``json_path`` — if given and the file exists, load from disk.
      2. Qdrant scroll via ``cfg`` — resolve taxonomy collection and
         scroll payloads from the live Qdrant instance.
      3. ``ValueError`` — neither source is available.

    Returns a dict keyed by annotation code/mnemonic, where each value
    is a payload dict with at least ``code`` and ``prototype_values``.
    """
    if json_path is not None:
        path = Path(json_path)
        if path.is_file():
            payloads = json.loads(path.read_text())
            logger.info(
                "enrichment_loader: loaded %d payloads from %s",
                len(payloads), path,
            )
            return payloads
        raise ValueError(
            f"enrichment_loader: json_path specified but file does not "
            f"exist: {path}"
        )

    payloads = _load_from_qdrant(cfg, taxonomy_id=taxonomy_id)
    if payloads is not None:
        return payloads

    raise ValueError(
        "enrichment_loader: no enrichment source available. Provide "
        "--payloads <path> to load from a JSON export, or ensure Qdrant "
        "is running with an enriched annotations collection. "
        "See: python scripts/export_enriched_annotations.py --format json"
    )


def _resolve_taxonomy_id(cfg) -> str:
    if cfg is None:
        return "default"
    explicit = getattr(cfg, "classify_taxonomy_id", None)
    return explicit or "default"


def _load_from_qdrant(
    cfg,
    taxonomy_id: str | None = None,
) -> dict[str, dict] | None:
    """Scroll Qdrant collection and extract full enrichment payloads."""
    try:
        from atelier.db.dao import AtelierDao
    except Exception as exc:
        logger.debug("enrichment_loader: AtelierDao unavailable: %s", exc)
        return None

    tax_id = taxonomy_id or _resolve_taxonomy_id(cfg)
    try:
        dao = AtelierDao()
        row = dao.get_current_taxonomy_collection(tax_id)
    except Exception as exc:
        logger.debug(
            "enrichment_loader: taxonomy lookup failed for %s: %s",
            tax_id, exc,
        )
        return None

    if row is None:
        logger.debug(
            "enrichment_loader: no current collection for taxonomy_id=%s",
            tax_id,
        )
        return None

    qdrant_url = row.get("qdrant_url") or ""
    collection = row.get("qdrant_collection") or ""
    if not collection:
        return None

    try:
        from qdrant_client import QdrantClient
    except ImportError:
        logger.warning(
            "enrichment_loader: qdrant-client not installed"
        )
        return None

    try:
        client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")
    except Exception as exc:
        logger.warning("enrichment_loader: Qdrant connect failed: %s", exc)
        return None

    payloads: dict[str, dict] = {}
    next_offset = None

    while True:
        try:
            points, next_offset = client.scroll(
                collection_name=collection,
                limit=256,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:
            logger.warning(
                "enrichment_loader: Qdrant scroll failed: %s", exc,
            )
            return payloads if payloads else None

        for p in points:
            raw = p.payload or {}
            code = raw.get("mnemonic") or raw.get("code")
            if not code:
                continue
            payloads[code] = {
                "code": code,
                "label": raw.get("label", ""),
                "description": raw.get("description", ""),
                "prototype_values": raw.get("prototype_values", []),
                "value_patterns": raw.get("value_patterns", []),
                "name_hints": raw.get("name_hints", []),
                "anti_examples": raw.get("anti_examples", []),
            }

        if next_offset is None:
            break

    if payloads:
        logger.info(
            "enrichment_loader: loaded %d payloads from Qdrant collection %s",
            len(payloads), collection,
        )
    return payloads if payloads else None
