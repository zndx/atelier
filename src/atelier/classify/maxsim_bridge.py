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
bridge raises ``MaxSimUnavailable`` so the pipeline fails
fast in the FSM rather than silently falling back to legacy
single-vector cosine.  Silent fallback is the no-silent-DST-
degradation anti-pattern: it has historically destroyed accuracy by
mixing a weaker evidence source into DST fusion without the operator
knowing.  The only ``None`` return is when ``is_enabled(cfg)`` is
``False`` — operator-explicit disable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.belief import BeliefAssignment, FrameOfDiscernment

logger = logging.getLogger(__name__)


# ── Fail-fast contract ────────────────────────────────────────────


class MaxSimUnavailable(RuntimeError):
    """Raised when the late-interaction path is enabled but cannot run.

    Per the no-silent-DST-degradation rule, the pipeline does NOT fall
    back to legacy single-vector cosine when late-interaction fails.
    Silent fallback historically ate accuracy (−13.6pp in measured runs)
    by mixing a weaker evidence source into DST fusion without the
    operator's knowledge.  Instead, the bridge raises this exception so
    the run fails loudly in the FSM and the operator restores the
    intended pathway (re-enrich, restart Qdrant, etc.).

    Attributes:
        status: machine-readable failure label
                (e.g. ``"degraded_no_collection"``,
                ``"degraded_namespace_mismatch"``, ``"degraded_score_error"``).
    """

    def __init__(self, status: str, message: str | None = None) -> None:
        self.status = status
        super().__init__(message or status)


# ── Public config probe ───────────────────────────────────────────


def is_enabled(cfg) -> bool:
    """Cheap probe — is the late-interaction path turned on in cfg?"""
    if cfg is None:
        return False
    try:
        v = getattr(cfg, "classify_maxsim_enabled", None)
        if v is not None:
            return bool(v)
        clf = getattr(cfg, "classify", None)
        if clf is not None:
            maxsim = getattr(clf, "maxsim", None)
            if maxsim is not None:
                return bool(getattr(maxsim, "enabled", False))
    except Exception as exc:  # noqa: BLE001
        logger.debug("maxsim.is_enabled probe failed: %s", exc)
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


def _auto_promote_latest(dao, taxonomy_id: str) -> dict | None:
    """If no 'current' collection exists, promote the latest 'building' one.

    The enrichment script registers with status='building' and promotes
    to 'current' on success.  If the pod restarted between registration
    and promotion — or the enrichment ran an older script version that
    lacked the promotion step — the row stays stuck at 'building'.
    Auto-promote so late-interaction works without manual intervention.
    """
    try:
        rows = dao.list_taxonomy_collections(taxonomy_id=taxonomy_id)
        building = [r for r in rows if r.get("status") == "building"]
        if not building:
            return None
        latest = building[0]  # list_taxonomy_collections returns newest first
        cid = latest.get("id")
        if cid and dao.set_current_taxonomy_collection(cid):
            logger.info(
                "late_interaction: auto-promoted taxonomy collection %s "
                "(%s) to 'current'",
                cid, latest.get("qdrant_collection", "?"),
            )
            return latest
    except Exception as exc:  # noqa: BLE001
        logger.debug("late_interaction: auto-promote failed: %s", exc)
    return None


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
        row = _auto_promote_latest(dao, taxonomy_id)
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


def try_compute_maxsim_mass(
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
        # Operator-explicit disable — only path that returns None.  All
        # other failure modes raise MaxSimUnavailable so the
        # pipeline fails fast rather than silently degrading DST.
        return None, "explicit_disable", None

    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        raise MaxSimUnavailable(
            "degraded_no_collection",
            "No 'current' Qdrant collection registered for the active "
            "taxonomy.  Run scripts/enrich_annotations.py to populate "
            "taxonomy_registry, then restart the Application.",
        )
    qdrant_url, collection = resolved

    try:
        import qdrant_client as _qdrant  # noqa: F401
    except ImportError as exc:
        raise MaxSimUnavailable(
            "degraded_no_qdrant_client",
            "qdrant_client is not importable — deployment is missing "
            "a required dependency.",
        ) from exc

    client = _get_qdrant_client(qdrant_url)
    if client is None:
        raise MaxSimUnavailable(
            "degraded_qdrant_connect",
            f"QdrantClient could not connect to {qdrant_url!r}.",
        )

    try:
        from qdrant_client.http import models as qm

        from atelier.classify.colbert_encoder import get_encoder, set_model_name
        from atelier.classify.mass_functions import maxsim_to_mass
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
            raise MaxSimUnavailable(
                "degraded_empty_results",
                f"Qdrant returned zero points for collection "
                f"{collection!r}; the taxonomy collection is empty or "
                f"the query vector failed to match anything.",
            )

        # Normalize MaxSim scores to [0, 1].  Qdrant's MaxSim sums
        # per-query-token max-cosines; dividing by the query token
        # count recovers the mean per-token similarity, which is the
        # scale _maxsim_reliability's sigmoid was calibrated for.
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
            raise MaxSimUnavailable(
                "degraded_namespace_mismatch",
                f"CODE NAMESPACE MISMATCH — Qdrant returned "
                f"{len(results.points)} results but none are in the "
                f"frame (sample Qdrant codes: {sample_codes}; sample "
                f"frame codes: {list(frame.singletons.keys())[:3]}).  "
                f"Re-enrich against the current taxonomy.",
            )

        mass = maxsim_to_mass(
            scored_tags, frame,
            alpha=getattr(cfg, "classify_mass_calibration_maxsim_alpha", 1.0),
            union_focal_k=int(getattr(cfg, "classify_maxsim_union_focal_k", 0)),
            union_focal_alpha=float(
                getattr(cfg, "classify_maxsim_union_focal_alpha", 0.45)
            ),
        )

        attribution = _build_attribution(
            scored_tags=scored_tags,
            frame=frame,
            top_k=attribution_top_k,
        )
        return mass, "ok", attribution

    except MaxSimUnavailable:
        # Already a typed fail-fast exception — propagate as-is so the
        # pipeline sees the status string intact.
        raise
    except Exception as exc:
        logger.warning(
            "late_interaction: scoring raised for %s.%s — re-raising as "
            "MaxSimUnavailable so the pipeline fails fast "
            "(no silent fallback to legacy single-vector cosine)",
            table_name, column_name, exc_info=True,
        )
        raise MaxSimUnavailable(
            "degraded_score_error",
            f"Late-interaction scoring raised "
            f"{type(exc).__name__}: {exc!r} for "
            f"{table_name}.{column_name}",
        ) from exc


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
            "is_leaf": code not in frame.internal_nodes,
            "maxsim_score": round(score, 6),
        })
    return {
        "top_k": rows,
        "ranking_basis": "qdrant_maxsim",
    }
