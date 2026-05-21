# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Pipeline-side bridge for the late-interaction cosine evidence source.

Uses ColBERT token-level embeddings with Qdrant's native MaxSim to
score columns against enriched annotations.  The entity side feeds
``ColumnFeatures.to_embedding_text()`` — the same text SAGE/SHAP
ablate over.  Qdrant performs the late-interaction MaxSim; the bridge
converts top-K scores to a DST mass function.

When the flag is true and the late-interaction path cannot run, the
bridge returns ``None`` so the caller can record the run as
**degraded**.  That condition is a deployment issue, not a normal
operating mode.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.belief import BeliefAssignment, FrameOfDiscernment

logger = logging.getLogger(__name__)


# ── Public config probe ───────────────────────────────────────────


def is_enabled(cfg) -> bool:
    """Cheap probe — is the late-interaction path turned on in cfg?"""
    if cfg is None:
        return False
    try:
        v = getattr(cfg, "classify_cosine_late_interaction_enabled", None)
        if v is not None:
            return bool(v)
        clf = getattr(cfg, "classify", None)
        if clf is not None:
            cosine = getattr(clf, "cosine", None)
            if cosine is not None:
                li = getattr(cosine, "late_interaction", None)
                if li is not None:
                    return bool(getattr(li, "enabled", False))
    except Exception as exc:  # noqa: BLE001
        logger.debug("late_interaction.is_enabled probe failed: %s", exc)
    return False


# ── Per-process state ─────────────────────────────────────────────


@dataclass
class _BridgeState:
    """Per-process caches for the late-interaction bridge."""

    qdrant_client: Any | None = None
    qdrant_url: str | None = None


_STATE: _BridgeState = _BridgeState()


def reset() -> None:
    """Reset the per-process cache."""
    global _STATE
    _STATE = _BridgeState()


# ── Resolution helpers ────────────────────────────────────────────


def _resolve_taxonomy_id(cfg) -> str:
    if cfg is None:
        return "default"
    explicit = getattr(cfg, "classify_taxonomy_id", None)
    return explicit or "default"


def _resolve_qdrant_collection(cfg) -> tuple[str, str] | None:
    """Look up (qdrant_url, collection_name) for the active taxonomy."""
    try:
        from atelier.db.dao import AtelierDao
        dao = AtelierDao()
    except Exception as exc:  # noqa: BLE001
        logger.debug("late_interaction: AtelierDao unavailable: %s", exc)
        return None

    taxonomy_id = _resolve_taxonomy_id(cfg)
    try:
        row = dao.get_current_taxonomy_collection(taxonomy_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "late_interaction: get_current_taxonomy_collection(%s) failed: %s",
            taxonomy_id, exc,
        )
        return None
    if row is None:
        logger.debug(
            "late_interaction: no current collection for taxonomy_id=%s",
            taxonomy_id,
        )
        return None
    return (row.get("qdrant_url") or "", row.get("qdrant_collection") or "")


def _get_qdrant_client(qdrant_url: str):
    """Lazy-construct + cache a QdrantClient."""
    if _STATE.qdrant_client is not None and _STATE.qdrant_url == qdrant_url:
        return _STATE.qdrant_client
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        return None
    try:
        client = QdrantClient(url=qdrant_url or "http://127.0.0.1:6333")
        _STATE.qdrant_client = client
        _STATE.qdrant_url = qdrant_url
        return client
    except Exception:  # noqa: BLE001
        return None


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
    embed=None,
    attribution_top_k: int = 3,
) -> "tuple[BeliefAssignment | None, str, dict | None]":
    """Compute late-interaction cosine mass for one column via Qdrant MaxSim.

    Returns ``(belief_assignment, status, attribution)``.

    The entity-side text is ``column_features.to_embedding_text()`` —
    the same text SAGE/SHAP ablate over.  This is encoded through
    ColBERT into token vectors and sent to Qdrant as a multi-vector
    query against the ``colbert`` field.  Qdrant returns top-K
    annotations ranked by MaxSim; those scores are converted to a
    DST mass function.
    """
    _ = embed  # legacy param — ColBERT encoder is self-supplied
    if not is_enabled(cfg):
        return None, "explicit_disable", None

    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        return None, "degraded_no_collection", None
    qdrant_url, collection = resolved

    try:
        import qdrant_client as _qdrant  # noqa: F401
    except ImportError:
        return None, "degraded_no_qdrant_client", None

    client = _get_qdrant_client(qdrant_url)
    if client is None:
        return None, "degraded_qdrant_connect", None

    try:
        from qdrant_client.http import models as qm

        from atelier.classify.colbert_encoder import get_encoder, set_model_name
        from atelier.classify.mass_functions import late_interaction_to_mass
        from atelier.enrichment.qdrant_writer import COLBERT_VECTOR_NAME

        colbert_model = getattr(cfg, "classify_colbert_model", None)
        if colbert_model:
            set_model_name(colbert_model)
        encoder = get_encoder()

        # Build entity text from the same features SAGE/SHAP operate on.
        if column_features is not None and hasattr(column_features, "to_embedding_text"):
            entity_text = column_features.to_embedding_text()
        else:
            parts = [column_name]
            if table_name:
                parts.append(f"in {table_name}")
            if samples:
                parts.append(", ".join(str(s) for s in samples[:10] if s is not None))
            entity_text = " | ".join(parts)

        query_vectors = encoder.encode_single(entity_text)
        num_query_tokens = query_vectors.shape[0]

        # Qdrant multi-vector MaxSim query — returns points ranked by
        # token-level late-interaction score.
        top_k = min(len(frame.singletons) + len(frame.internal_nodes), 50)
        results = client.query_points(
            collection_name=collection,
            query=query_vectors.tolist(),
            using=COLBERT_VECTOR_NAME,
            limit=top_k,
            with_payload=True,
        )

        if not results.points:
            return None, "degraded_empty_results", None

        # Normalize MaxSim scores to [0, 1].  Qdrant's MaxSim sums
        # per-query-token max-cosines; dividing by the query token
        # count recovers the mean per-token similarity, which is the
        # scale _cosine_reliability's sigmoid was calibrated for.
        scored_tags: list[tuple[str, float]] = []
        for point in results.points:
            code = (point.payload or {}).get("code")
            if code is None:
                continue
            if code not in frame.singletons and code not in frame.internal_nodes:
                continue
            normalized_score = point.score / max(num_query_tokens, 1)
            scored_tags.append((code, normalized_score))

        if not scored_tags:
            sample_codes = [
                (p.payload or {}).get("code", "?") for p in results.points[:3]
            ]
            logger.warning(
                "late_interaction: CODE NAMESPACE MISMATCH — Qdrant returned "
                "%d results but none are in the frame (sample codes: %s; "
                "sample frame: %s).  Re-enrich against the current taxonomy.",
                len(results.points), sample_codes,
                list(frame.singletons.keys())[:3],
            )
            return None, "degraded_namespace_mismatch", None

        mass = late_interaction_to_mass(scored_tags, frame)

        attribution = _build_attribution(
            scored_tags=scored_tags,
            frame=frame,
            top_k=attribution_top_k,
        )
        return mass, "ok", attribution

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "late_interaction: scoring raised for %s.%s: %s; "
            "this is a deployment issue (investigate), not a normal flow",
            table_name, column_name, exc,
        )
        return None, "degraded_score_error", None


def _build_attribution(
    *,
    scored_tags: list[tuple[str, float]],
    frame: "FrameOfDiscernment",
    top_k: int,
) -> dict:
    """Top-K tags by MaxSim score with leaf/internal annotation."""
    rows = []
    for code, score in scored_tags[:top_k]:
        rows.append({
            "code": code,
            "is_leaf": code in frame.singletons,
            "maxsim_score": round(score, 6),
        })
    return {
        "top_k": rows,
        "ranking_basis": "qdrant_maxsim",
    }
