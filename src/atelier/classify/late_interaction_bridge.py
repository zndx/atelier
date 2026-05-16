# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Pipeline-side bridge for the late-interaction cosine evidence source.

Encapsulates the integration surface so ``pipeline.py``'s
``_classify_column`` only needs a 5-line gated insertion.  The bridge:

- Honors a ``classify.cosine.late_interaction.enabled`` config flag (default off)
- Resolves the current Qdrant collection for the active taxonomy via PGlite
- Materializes and caches the annotation index across columns
- Builds the column-side multi-vector query and computes per-tag scores
- Converts to a mass function via :func:`mass_functions.late_interaction_to_mass`

When the flag is off or any required infrastructure is missing
(qdrant-client not installed, no collection registered, Qdrant
unreachable), the bridge returns ``None`` and the caller falls back to
the legacy single-vector cosine path.  Failure modes are explicit and
non-fatal — late-interaction is *additive*, never a hard dependency.

See ``docs/src/architecture/late-interaction-cosine.md``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.belief import BeliefAssignment, FrameOfDiscernment
    from atelier.classify.late_interaction import AnnotationIndex
    from atelier.classify.multi_vector_features import ColumnMultiVectorQuery

logger = logging.getLogger(__name__)


# ── Public config probe ───────────────────────────────────────────


def is_enabled(cfg) -> bool:
    """Cheap probe — is the late-interaction path turned on in cfg?

    The HOCON binding is ``classify.cosine.late_interaction.enabled``,
    which can be flipped via ``ATELIER_CLASSIFY_COSINE_LATE_INTERACTION``.
    Default off.

    The probe is defensive: a missing config section is treated as
    "off" rather than an error, so existing configs without the new
    section continue to work unchanged.
    """
    if cfg is None:
        return False
    try:
        # AtelierConfig surfaces this as a flat field via the
        # ``_HOCON_MAP`` ``classify.cosine.late_interaction.enabled`` entry.
        v = getattr(cfg, "classify_cosine_late_interaction_enabled", None)
        if v is not None:
            return bool(v)
        # Nested attribute access fallback for non-AtelierConfig shapes
        # (tests, ad-hoc mocks).
        clf = getattr(cfg, "classify", None)
        if clf is not None:
            cosine = getattr(clf, "cosine", None)
            if cosine is not None:
                li = getattr(cosine, "late_interaction", None)
                if li is not None:
                    return bool(getattr(li, "enabled", False))
    except Exception as exc:  # noqa: BLE001 — config probe must never raise
        logger.debug("late_interaction.is_enabled probe failed: %s", exc)
    return False


# ── Per-process state ─────────────────────────────────────────────


@dataclass
class _BridgeState:
    """Per-process caches for the late-interaction bridge.

    The annotation index is the expensive thing to materialize
    (network round-trips to Qdrant + vector deserialization).  Cache it
    keyed by (qdrant_url, collection_name) for the lifetime of the
    process; reset between runs via :func:`reset` when the registry
    rotates the ``current`` row.
    """

    qdrant_client: Any | None = None
    qdrant_url: str | None = None
    annotation_indices: dict[tuple[str, str], "AnnotationIndex"] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.annotation_indices is None:
            self.annotation_indices = {}


_STATE: _BridgeState = _BridgeState()


def reset() -> None:
    """Reset the per-process cache.

    Called when the registry rotates the ``current`` row for a
    taxonomy (so subsequent columns re-materialize against the new
    collection).
    """
    global _STATE
    _STATE = _BridgeState()


# ── Resolution helpers ────────────────────────────────────────────


def _resolve_taxonomy_id(cfg) -> str:
    """Derive the taxonomy_id to look up in the registry.

    Initially keyed off the active vocabulary identity.  For phase-3
    work this stub returns ``"default"`` unless cfg surfaces an
    explicit override — extend with real wiring when multi-taxonomy
    deployments need it.
    """
    if cfg is None:
        return "default"
    explicit = getattr(cfg, "classify_taxonomy_id", None)
    return explicit or "default"


def _resolve_qdrant_collection(cfg) -> tuple[str, str] | None:
    """Look up (qdrant_url, collection_name) for the active taxonomy.

    Returns ``None`` when no ``current`` row exists for the taxonomy
    or PGlite is unreachable — caller falls back to legacy cosine.
    """
    try:
        from atelier.db.dao import AtelierDao

        dao = AtelierDao()
    except Exception as exc:  # noqa: BLE001 — DAO not always available
        logger.debug("late_interaction: AtelierDao unavailable: %s", exc)
        return None

    taxonomy_id = _resolve_taxonomy_id(cfg)
    try:
        row = dao.get_current_taxonomy_collection(taxonomy_id)
    except Exception as exc:  # noqa: BLE001 — DB query may fail mid-flight
        logger.debug(
            "late_interaction: get_current_taxonomy_collection(%s) failed: %s",
            taxonomy_id, exc,
        )
        return None
    if row is None:
        logger.debug(
            "late_interaction: no current collection for taxonomy_id=%s", taxonomy_id,
        )
        return None
    return (row.get("qdrant_url") or "", row.get("qdrant_collection") or "")


def _get_qdrant_client(qdrant_url: str):
    """Lazy-construct + cache a QdrantClient.  Returns None on import / connect failure."""
    if _STATE.qdrant_client is not None and _STATE.qdrant_url == qdrant_url:
        return _STATE.qdrant_client
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        logger.debug("late_interaction: qdrant_client not installed: %s", exc)
        return None
    try:
        client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")
        _STATE.qdrant_client = client
        _STATE.qdrant_url = qdrant_url
        return client
    except Exception as exc:  # noqa: BLE001 — connect may fail
        logger.debug("late_interaction: Qdrant connect failed: %s", exc)
        return None


def _get_annotation_index(qdrant_url: str, collection: str):
    """Lazy-materialize + cache the AnnotationIndex for a collection."""
    key = (qdrant_url, collection)
    cached = _STATE.annotation_indices.get(key)
    if cached is not None:
        return cached
    client = _get_qdrant_client(qdrant_url)
    if client is None:
        return None
    try:
        from atelier.classify.late_interaction import load_annotation_index
        index = load_annotation_index(client, collection=collection)
    except Exception as exc:  # noqa: BLE001 — load may fail
        logger.warning(
            "late_interaction: load_annotation_index(%s) failed: %s",
            collection, exc,
        )
        return None
    if not index.views:
        logger.warning(
            "late_interaction: annotation index for %s is empty; falling back",
            collection,
        )
        return None
    _STATE.annotation_indices[key] = index
    return index


# ── Public entry point ────────────────────────────────────────────


def try_compute_cosine_mass(
    *,
    cfg,
    column_features,
    column_name: str,
    table_name: str | None,
    samples: list,
    neighbor_column_names: list[str] | None,
    pattern_summary: str | None,
    frame: "FrameOfDiscernment",
    embed,
) -> "BeliefAssignment | None":
    """Compute a late-interaction cosine mass function for one column.

    Returns the :class:`BeliefAssignment` on success or ``None`` if
    the late-interaction path is disabled, infrastructure is
    unavailable, or anything fails non-fatally.  The caller is
    expected to handle ``None`` by falling through to the legacy
    single-vector cosine path.

    The signature accepts ``column_features`` for future use (SHAP
    integration, pattern-summary derivation) without requiring it
    today.

    Parameters
    ----------
    cfg
        AtelierConfig; ``is_enabled(cfg)`` must return True for the
        bridge to do anything.
    column_features
        Unused today; reserved for the SHAP integration that consumes
        per-view score breakdowns.
    column_name, table_name, samples, neighbor_column_names, pattern_summary
        Column-side inputs forwarded to
        :func:`multi_vector_features.build_column_query`.
    frame
        Frame of discernment for the candidate codes.
    embed
        Embedding callable shared with the legacy cosine path.
    """
    _ = column_features  # reserved
    if not is_enabled(cfg):
        return None

    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        return None
    qdrant_url, collection = resolved

    index = _get_annotation_index(qdrant_url, collection)
    if index is None:
        return None

    try:
        from atelier.classify.late_interaction import (
            ScoringWeights,
            score_column_against_index,
        )
        from atelier.classify.mass_functions import late_interaction_to_mass
        from atelier.classify.multi_vector_features import build_column_query
    except ImportError as exc:
        logger.debug("late_interaction: module import failed: %s", exc)
        return None

    try:
        query = build_column_query(
            column_name=column_name,
            table_name=table_name,
            samples=samples,
            neighbor_column_names=neighbor_column_names,
            pattern_summary=pattern_summary,
            embed=embed,
        )
        weights = _scoring_weights_from_cfg(cfg)
        scores = score_column_against_index(query, index, weights)
        return late_interaction_to_mass(scores, frame)
    except Exception as exc:  # noqa: BLE001 — bridge must not break the pipeline
        logger.warning(
            "late_interaction: scoring failed for %s.%s: %s; falling back",
            table_name, column_name, exc,
        )
        return None


def _scoring_weights_from_cfg(cfg):
    """Build :class:`ScoringWeights` from cfg, with defaults on missing keys."""
    from atelier.classify.late_interaction import ScoringWeights

    if cfg is None:
        return ScoringWeights()

    li = None
    try:
        clf = getattr(cfg, "classify", None)
        if clf is not None:
            cosine = getattr(clf, "cosine", None)
            if cosine is not None:
                li = getattr(cosine, "late_interaction", None)
    except Exception:  # noqa: BLE001 — defensive
        li = None

    if li is None:
        return ScoringWeights()

    return ScoringWeights(
        weight_label=getattr(li, "weight_label", 0.20),
        weight_name_hints=getattr(li, "weight_name_hints", 0.15),
        weight_prototype_values=getattr(li, "weight_prototype_values", 0.30),
        weight_value_patterns=getattr(li, "weight_value_patterns", 0.15),
        weight_context=getattr(li, "weight_context", 0.10),
        weight_parent_path=getattr(li, "weight_parent_path", 0.10),
        weight_anti=getattr(li, "weight_anti_examples", 0.30),
    )
