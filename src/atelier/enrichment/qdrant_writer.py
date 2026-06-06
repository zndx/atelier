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

COLBERT_VECTOR_NAME: str = "colbert"

# Legacy constants — kept only for migration tooling that reads old
# collections.  New collections use a single ``colbert`` multi-vector
# field.
SINGLE_VECTOR_NAMES: tuple[str, ...] = (
    "label_view",
    "description_view",
    "parent_path_view",
)
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
    """ColBERT token vectors for one annotation.

    Holds the token-level embeddings produced by encoding the composed
    annotation text through a ColBERT model.  Stored as a single
    multi-vector field in Qdrant with MaxSim comparator.
    """

    colbert: list[list[float]]

    def to_qdrant_vectors(self) -> dict[str, Any]:
        """Shape Qdrant's per-point vectors mapping expects."""
        return {COLBERT_VECTOR_NAME: self.colbert}


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


# ── Annotation text composition ──────────────────────────────────


def compose_annotation_text(payload: dict) -> str:
    """Compose a single text passage from an annotation's payload.

    Mirrors ``ColumnFeatures.to_embedding_text()`` on the entity side:
    the entity text carries column-level features (name, type, samples,
    patterns, siblings, etc.); this text carries annotation-level
    features (label, description, prototypes, name hints, taxonomy path).

    Both are fed through the same ColBERT encoder — Qdrant's native
    MaxSim handles the token-level cross-alignment.

    Anti-examples are deliberately excluded: they add noise in the
    embedding space without improving MaxSim discrimination.
    """
    parts: list[str] = []

    label = payload.get("label") or ""
    if label:
        parts.append(label)

    description = payload.get("description") or ""
    if description:
        parts.append(description)

    prototype_values = payload.get("prototype_values") or []
    if prototype_values:
        vals = prototype_values[:10] if len(prototype_values) > 10 else prototype_values
        # Flatten if values are dicts with a 'value' key (enrichment format)
        flat = []
        for v in vals:
            if isinstance(v, dict):
                flat.append(str(v.get("value", v)))
            else:
                flat.append(str(v))
        parts.append("values: " + ", ".join(flat))

    name_hints = payload.get("name_hints") or []
    if name_hints:
        hints = name_hints[:10] if len(name_hints) > 10 else name_hints
        flat = []
        for h in hints:
            if isinstance(h, dict):
                flat.append(str(h.get("hint", h)))
            else:
                flat.append(str(h))
        parts.append("names: " + ", ".join(flat))

    value_patterns = payload.get("value_patterns") or []
    if value_patterns:
        pats = value_patterns[:5] if len(value_patterns) > 5 else value_patterns
        descs = []
        for p in pats:
            if isinstance(p, dict):
                desc = p.get("description") or p.get("expr") or str(p)
                descs.append(str(desc))
            else:
                descs.append(str(p))
        parts.append("patterns: " + ", ".join(descs))

    parent_path = payload.get("parent_path") or []
    if parent_path:
        if isinstance(parent_path, list):
            parts.append("ontology: " + " > ".join(str(p) for p in parent_path))
        else:
            parts.append("ontology: " + str(parent_path))

    mnemonic = payload.get("mnemonic") or ""
    if mnemonic and mnemonic != label:
        parts.append(f"abbrev: {mnemonic}")

    return " | ".join(parts) if parts else ""


# ── Qdrant operations ─────────────────────────────────────────────


def collection_name_for(taxonomy_id: str, augmentation_version: str) -> str:
    """Compute the legacy Qdrant collection name for a (taxonomy, version) pair.

    Convention: ``annotations_<taxonomy_id>_<augmentation_version>``.
    Non-alphanumeric characters in either component are replaced with
    underscores to satisfy Qdrant's naming rules.

    Retained for legacy enrichment paths.  New collections produced by
    the evolve-classification roll-forward loop use ``collection_name_v2``
    (UUIDv7-suffixed, opaque-identifier-style names whose semantic
    versioning lives in the ``taxonomy_registry.augmentation_version``
    field rather than embedded in the name).
    """
    safe_tax = "".join(c if c.isalnum() else "_" for c in taxonomy_id).strip("_")
    safe_ver = "".join(c if c.isalnum() else "_" for c in augmentation_version).strip("_")
    return f"annotations_{safe_tax}_{safe_ver}"


# Qdrant collection name limit per docs.  We pre-check rather than
# letting a confusing 4xx surface mid-apply.
_QDRANT_NAME_MAX = 255


def _normalize_name_component(s: str, *, what: str) -> str:
    """Normalize one naming component to ``[A-Za-z0-9_-]+``.

    Two-tier separator mapping preserves the semantic boundary between
    sub-parts of a compound identifier:

    * ``/`` and space → ``_``  (structural boundary, e.g. the slash
      between connection and schema.table in ``hive-poc/default.annotations``)
    * ``.`` and ``_`` → ``-``  (intra-part punctuation, e.g. the dot
      between schema and table)

    Raises ``ValueError`` on un-normalizable characters (punctuation
    outside the allowed set) or empty result.  The fail-fast posture is
    intentional — we'd rather refuse a build than ship a collection
    name that Qdrant later rejects mid-iteration.
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"{what} must be a non-empty string, got {s!r}")
    out: list[str] = []
    for c in s:
        if c.isalnum() or c == "-":
            out.append(c)
        elif c in ("/", " "):
            out.append("_")
        elif c in (".", "_"):
            out.append("-")
        else:
            raise ValueError(
                f"{what}={s!r}: character {c!r} cannot be normalized "
                f"(allowed: alphanumerics, '-'; mapped: '/', ' ' → '_'; "
                f"'.', '_' → '-')"
            )
    result = "".join(out).strip("-_")
    # Collapse runs of identical separators introduced by the mapping
    while "--" in result:
        result = result.replace("--", "-")
    while "__" in result:
        result = result.replace("__", "_")
    if not result:
        raise ValueError(f"{what}={s!r} normalized to empty string")
    return result


def collection_name_v2(*, source_table: str,
                      uuid_v7: str | None = None) -> str:
    """Generate an opaque UUIDv7-suffixed collection name.

    Convention: ``<source_table>_<uuidv7>``.  ``source_table`` is
    expected to be the fully-qualified, connection-prefixed identifier
    (e.g. ``"hive-poc/default.annotations"``) — it is already compound
    by requirement, so the connection is not passed separately.
    Each component is normalized to ``[A-Za-z0-9_-]``: ``/`` and
    spaces become ``_`` (preserving the structural connection/table
    boundary), while ``.`` and ``_`` inside a part become ``-``.
    Example: ``hive-poc/default.annotations`` →
    ``hive-poc_default-annotations``.  The source_table and the uuid
    are joined with ``_``.

    The collection name is *opaque*: it carries no semantic versioning.
    Lineage and meaning live in the taxonomy_registry row's
    ``augmentation_version`` and ``built_at`` fields, plus the manifest
    artifacts under ``build/data/transforms/``.

    Args:
        source_table: fully-qualified, connection-prefixed source table
            identifier (e.g. ``"hive-poc/default.annotations"``).
        uuid_v7: pre-generated UUIDv7 string; when omitted, one is
            generated via ``uuid_utils.uuid7()``.

    Raises:
        ValueError: if any component fails normalization, or if the
            resulting name exceeds Qdrant's 255-character limit.
    """
    tbl_norm = _normalize_name_component(source_table, what="source_table")
    if uuid_v7 is None:
        import uuid_utils
        uuid_v7 = str(uuid_utils.uuid7())
    uuid_norm = _normalize_name_component(uuid_v7, what="uuid_v7")
    name = f"{tbl_norm}_{uuid_norm}"
    if len(name) > _QDRANT_NAME_MAX:
        raise ValueError(
            f"collection name length {len(name)} exceeds Qdrant's "
            f"{_QDRANT_NAME_MAX}-character limit: {name!r}"
        )
    return name


def ensure_collection(
    client: "QdrantClient",
    *,
    collection: str,
    embedding_dim: int,
    distance: str = "Cosine",
    recreate: bool = False,
) -> None:
    """Create the Qdrant collection with a single ColBERT multi-vector field.

    If ``recreate`` is True and the collection exists, it is dropped and
    re-created.  Otherwise an existing collection is left in place
    (idempotent rebuilds rely on this).

    The collection has one named vector field (``colbert``) configured
    with ``MaxSim`` comparator for native late-interaction scoring.
    """
    from qdrant_client.http import models as qm

    dist = qm.Distance[distance.upper()]
    vectors_config = {
        COLBERT_VECTOR_NAME: qm.VectorParams(
            size=embedding_dim,
            distance=dist,
            multivector_config=qm.MultiVectorConfig(
                comparator=qm.MultiVectorComparator.MAX_SIM,
            ),
        ),
    }

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
