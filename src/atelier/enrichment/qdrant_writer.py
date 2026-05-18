# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Qdrant writer for enriched annotation profiles.

Writes one multi-vector point per annotation with structured JSON payload.
The Qdrant collection is the source of truth for enriched annotations;
PGlite carries only the administrative pointer.  See
``docs/src/architecture/late-interaction-cosine.md`` § Qdrant payload schema.

Content-addressed caching makes rebuilds idempotent: the per-point cache
key is the sha256 of (taxonomy_id, taxonomy_version_hash, augmentation_version,
embedding_model, source_row_hash).  When a rebuild encounters a cache hit
on a row, the point is left untouched.

The module deliberately stays small.  Embedding computation lives in the
caller (the enrichment loop); this module's job is to translate a
(named_vectors, payload) bundle into Qdrant operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


# ── Named-vector schema ───────────────────────────────────────────


# Named vectors that hold *one* embedding (single-vector slots).
# Stored in Qdrant as separate named vectors on the same point.
SINGLE_VECTOR_NAMES: tuple[str, ...] = (
    "label_view",
    "description_view",
    "parent_path_view",
)

# Named vectors that hold *many* embeddings each (multi-vector slots).
# Stored in Qdrant using its multi-vector named-vectors API (one named
# vector per slot, holding a list of embeddings).
MULTI_VECTOR_NAMES: tuple[str, ...] = (
    "prototype_values",
    "value_patterns",
    "name_hints",
    "anti_examples",
)

ALL_VECTOR_NAMES: tuple[str, ...] = SINGLE_VECTOR_NAMES + MULTI_VECTOR_NAMES


# ── Cache keys ────────────────────────────────────────────────────


def source_row_hash(source_row: dict) -> str:
    """Hash of the immutable fields of one source taxonomy row.

    The hash covers the fields that, if changed, would invalidate the
    enrichment: label, mnemonic, description, parent_code.  Other
    fields (e.g., display formatting) may change without forcing a
    re-enrichment.
    """
    canonical = {
        "label": source_row.get("label", ""),
        "mnemonic": source_row.get("mnemonic", ""),
        "description": source_row.get("description", ""),
        "parent_code": source_row.get("parent_code", ""),
    }
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def taxonomy_version_hash(source_rows: list[dict]) -> str:
    """Hash of the full source taxonomy snapshot.

    Combines the row hashes (sorted by code/mnemonic) so reordering the
    rows doesn't change the hash but adding/removing/editing rows does.
    """
    per_row = sorted(
        f"{r.get('code', r.get('mnemonic', ''))}:{source_row_hash(r)}"
        for r in source_rows
    )
    blob = "\n".join(per_row)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def point_cache_key(
    *,
    taxonomy_id: str,
    taxonomy_version_hash_value: str,
    augmentation_version: str,
    embedding_model: str,
    source_row_hash_value: str,
) -> str:
    """Content-addressed cache key for one enriched annotation point.

    Two builds with identical inputs produce identical keys; a change
    to any input invalidates the key and forces a fresh enrichment.
    """
    blob = "::".join([
        taxonomy_id,
        taxonomy_version_hash_value,
        augmentation_version,
        embedding_model,
        source_row_hash_value,
    ])
    # Qdrant requires point IDs to be unsigned integers or UUIDs.
    # Derive a deterministic UUID (version 5 style) from the SHA256
    # by taking the first 16 bytes and formatting as a UUID.
    digest = hashlib.sha256(blob.encode("utf-8")).digest()
    import uuid as _uuid
    return str(_uuid.UUID(bytes=digest[:16]))


# ── Point construction ────────────────────────────────────────────


@dataclass
class AnnotationVectors:
    """Embeddings for one annotation, keyed by named-vector slot.

    Single-vector slots hold a single list[float]; multi-vector slots
    hold a list[list[float]].  Slot names must be drawn from
    :data:`SINGLE_VECTOR_NAMES` and :data:`MULTI_VECTOR_NAMES`.
    """

    label_view: list[float]
    description_view: list[float]
    parent_path_view: list[float]
    prototype_values: list[list[float]] = field(default_factory=list)
    value_patterns: list[list[float]] = field(default_factory=list)
    name_hints: list[list[float]] = field(default_factory=list)
    anti_examples: list[list[float]] = field(default_factory=list)

    def to_qdrant_vectors(self) -> dict[str, Any]:
        """Shape Qdrant's per-point vectors mapping expects.

        Qdrant's multi-vector API accepts a list-of-vectors under a single
        named-vector slot.  Single-vector slots take a flat list[float].
        """
        return {
            "label_view": self.label_view,
            "description_view": self.description_view,
            "parent_path_view": self.parent_path_view,
            "prototype_values": self.prototype_values,
            "value_patterns": self.value_patterns,
            "name_hints": self.name_hints,
            "anti_examples": self.anti_examples,
        }


@dataclass
class EnrichedAnnotationPoint:
    """One annotation, ready to be written to Qdrant.

    The ``point_id`` is the content-addressed cache key — using the cache
    key as the point ID means a re-run that produces identical content
    overwrites the point in place rather than creating duplicates, and
    "is this annotation already up-to-date?" reduces to a point-exists
    check.
    """

    point_id: str
    vectors: AnnotationVectors
    payload: dict


def build_point(
    *,
    source_row: dict,
    enrichment: dict,
    vectors: AnnotationVectors,
    verifier_results: dict,
    taxonomy_id: str,
    taxonomy_version_hash_value: str,
    taxonomy_version_label: str,
    augmentation_version: str,
    embedding_model: str,
    embedding_dim: int,
    generated_at: str,
    generated_by: str,
) -> EnrichedAnnotationPoint:
    """Assemble one enriched annotation point.

    Combines the passthrough source-row fields, the LLM-generated
    enrichment fields, the verifier results, and provenance metadata
    into the payload shape documented in the architecture note.
    """
    sr_hash = source_row_hash(source_row)
    pid = point_cache_key(
        taxonomy_id=taxonomy_id,
        taxonomy_version_hash_value=taxonomy_version_hash_value,
        augmentation_version=augmentation_version,
        embedding_model=embedding_model,
        source_row_hash_value=sr_hash,
    )

    payload = {
        # Source taxonomy fields (immutable passthrough)
        "code": source_row.get("code"),
        "label": source_row.get("label"),
        "mnemonic": source_row.get("mnemonic"),
        "description": source_row.get("description"),
        "parent_code": source_row.get("parent_code"),
        "parent_path": enrichment.get("parent_path", []),

        # Enrichment fields (LLM-generated + verified)
        "prototype_values": enrichment.get("prototype_values", []),
        "value_patterns": enrichment.get("value_patterns", []),
        "name_hints": enrichment.get("name_hints", []),
        "anti_examples": enrichment.get("anti_examples", []),

        # Provenance + audit
        "augmentation_version": augmentation_version,
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
        "generated_at": generated_at,
        "generated_by": generated_by,
        "verifier_results": verifier_results,
        "source_row_hash": sr_hash,

        # Append-only operator edits log (begins empty)
        "operator_edits": [],

        # Cross-reference
        "taxonomy_id": taxonomy_id,
        "taxonomy_version": taxonomy_version_label,
        "taxonomy_version_hash": taxonomy_version_hash_value,
    }

    return EnrichedAnnotationPoint(point_id=pid, vectors=vectors, payload=payload)


# ── Qdrant operations ─────────────────────────────────────────────


def collection_name_for(taxonomy_id: str, augmentation_version: str) -> str:
    """Compute the Qdrant collection name for a (taxonomy, version) pair.

    Convention: ``annotations_<taxonomy_id>_<augmentation_version>``.
    Non-alphanumeric characters in either component are replaced with
    underscores to satisfy Qdrant's naming rules.
    """
    safe_tax = "".join(c if c.isalnum() else "_" for c in taxonomy_id).strip("_")
    safe_ver = "".join(c if c.isalnum() else "_" for c in augmentation_version).strip("_")
    return f"annotations_{safe_tax}_{safe_ver}"


def ensure_collection(
    client: "QdrantClient",
    *,
    collection: str,
    embedding_dim: int,
    distance: str = "Cosine",
    recreate: bool = False,
) -> None:
    """Create the Qdrant collection with the required named-vector schema.

    If ``recreate`` is True and the collection exists, it is dropped and
    re-created.  Otherwise an existing collection is left in place
    (idempotent rebuilds rely on this).

    The named-vector schema declares one slot per name in
    :data:`ALL_VECTOR_NAMES`, all with the same embedding dimensionality
    and distance metric.  Multi-vector slots and single-vector slots are
    declared identically; Qdrant decides whether a slot stores one or
    many vectors at write time based on the input shape.
    """
    from qdrant_client.http import models as qm

    dist = qm.Distance[distance.upper()]
    vectors_config = {}
    for name in SINGLE_VECTOR_NAMES:
        vectors_config[name] = qm.VectorParams(size=embedding_dim, distance=dist)
    for name in MULTI_VECTOR_NAMES:
        vectors_config[name] = qm.VectorParams(
            size=embedding_dim,
            distance=dist,
            multivector_config=qm.MultiVectorConfig(
                comparator=qm.MultiVectorComparator.MAX_SIM,
            ),
        )

    exists = client.collection_exists(collection)
    if exists and recreate:
        logger.info("Recreating Qdrant collection %s", collection)
        client.delete_collection(collection)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=collection,
            vectors_config=vectors_config,
        )
        logger.info("Created Qdrant collection %s", collection)


def upsert_point(
    client: "QdrantClient",
    *,
    collection: str,
    point: EnrichedAnnotationPoint,
) -> None:
    """Upsert one enriched annotation point.

    Content-addressed point IDs mean idempotent re-runs are safe: a
    rebuild that produces identical content for a row overwrites the
    same point in place.
    """
    from qdrant_client.http import models as qm

    client.upsert(
        collection_name=collection,
        points=[
            qm.PointStruct(
                id=point.point_id,
                vector=point.vectors.to_qdrant_vectors(),
                payload=point.payload,
            )
        ],
    )


def point_exists(client: "QdrantClient", *, collection: str, point_id: str) -> bool:
    """Cheap "is this annotation already up-to-date?" check.

    Used by the enrichment loop to skip rows whose content-addressed
    point already exists in Qdrant.  Combined with the content-addressed
    point IDs, this gives idempotent rebuild semantics: only rows that
    actually changed get re-enriched.
    """
    try:
        result = client.retrieve(collection_name=collection, ids=[point_id])
        return bool(result)
    except Exception as exc:  # noqa: BLE001 — Qdrant client raises a broad set
        logger.debug("point_exists check failed for %s: %s", point_id, exc)
        return False
