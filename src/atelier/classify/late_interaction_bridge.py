# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Pipeline-side bridge for the late-interaction cosine evidence source.

This is the production cosine evidence path under the DST-independence
architecture — not an optional feature.  It restores discriminative
multi-vector signal on adversarial corpora and is load-bearing for the
"no collapsed-DST" property the pipeline depends on.  The bridge:

- Reads ``classify.cosine.late_interaction.enabled`` (default **true**);
  an explicit ``false`` is an emergency rollback path only.
- Resolves the current Qdrant collection for the active taxonomy via PGlite.
- Materializes and caches the annotation index across columns.
- Builds the column-side multi-vector query and computes per-tag scores.
- Converts to a mass function via :func:`mass_functions.late_interaction_to_mass`.

When the flag is true and the late-interaction path cannot run
(qdrant-client missing, no collection registered, Qdrant unreachable,
scoring error), the bridge returns ``None`` so the caller can record
the run as **degraded** and proceed under the legacy single-vector
cosine path *as a transitional emergency fallback only*.  That
condition is a deployment issue that must be investigated — it is not
a normal operating mode and must not be normalized in code or docs.
The caller (``pipeline.py``) emits a WARNING and marks the run's
source-mass metadata so operators see the degradation in the result
artifact.

When the flag is explicitly ``false``, the bridge returns ``None``
silently — that's the "operator knows what they're doing" path.

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


def _get_default_embedder():
    """Self-supply a sentence-transformer embedder when the caller passes None.

    The pipeline call-site passes ``embed=getattr(cfg, "_embedder", None)``
    but that attribute is never actually populated on AtelierConfig.  Rather
    than thread the embedder setup through the config, the bridge lazily
    loads the same MiniLM model the rest of the classify path uses via
    :func:`atelier.classify.embedding._get_model`.

    Returns a callable matching ``EmbedFn`` from multi_vector_features:
    ``str | list[str] → list[float] | list[list[float]]``.
    """
    from atelier.classify.embedding import _get_model

    model = _get_model()

    def _embed(text):
        """Thin adapter: model.encode → list[float] (or list[list[float]])."""
        import numpy as np

        result = model.encode(text)
        if isinstance(result, np.ndarray):
            return result.tolist()
        return result

    return _embed


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
    attribution_top_k: int = 3,
) -> "tuple[BeliefAssignment | None, str, dict | None]":
    """Compute the late-interaction cosine mass function for one column.

    Returns ``(belief_assignment, status, attribution)``.  ``status`` is one of:

    - ``"ok"`` — late-interaction succeeded; ``belief_assignment`` is the result.
    - ``"explicit_disable"`` — the operator set
      ``classify.cosine.late_interaction.enabled = false``; the caller
      should fall back to legacy cosine silently.  This path is for
      emergency rollback only.
    - ``"degraded_no_dao"`` — PGlite unavailable.
    - ``"degraded_no_collection"`` — no ``current`` collection for the
      active taxonomy.  Run the enrichment script.
    - ``"degraded_no_qdrant_client"`` — qdrant-client not installed.
    - ``"degraded_qdrant_connect"`` — Qdrant unreachable at the URL.
    - ``"degraded_load_failed"`` — annotation index materialization failed.
    - ``"degraded_score_error"`` — late-interaction scoring raised.

    Every ``degraded_*`` status indicates a deployment issue, not a
    normal operating mode.  The caller is expected to emit a WARNING
    and mark the run's source-mass metadata so the degradation is
    visible in the result artifact.  Operators should investigate and
    restore the late-interaction path rather than leaving the
    pipeline in degraded mode.

    The third return element, ``attribution``, is the **per-decision
    SHAP surface** over the late-interaction roles.  When status is
    ``"ok"``, it carries the per-role contribution breakdown for the
    top-K (default 3) tags by post-fusion mass — one entry per tag
    with ``positive_score``, ``negative_score``,
    ``per_role: {label, name_hints, prototype_values, value_patterns,
    context, parent_path, anti_examples}``, and
    ``verifier_pass_rate``.  This is direct attribution, not
    permutation-based: late-interaction's structure already exposes
    the per-role components, so the operator-legible "predicted EMAIL
    because prototype_values=0.42 and name_hints=0.15" view is
    available without re-scoring cost.  ``None`` for any non-``"ok"``
    status.

    Parameters
    ----------
    cfg
        AtelierConfig.  When ``is_enabled(cfg)`` is False the bridge
        short-circuits to ``"explicit_disable"``.
    column_features
        Reserved for SHAP feature-aggregation use (the per-decision
        attribution surface is returned in ``attribution``).
    column_name, table_name, samples, neighbor_column_names, pattern_summary
        Column-side inputs forwarded to
        :func:`multi_vector_features.build_column_query`.
    frame
        Frame of discernment for the candidate codes.
    embed
        Embedding callable shared with the legacy cosine path.
    attribution_top_k
        Number of top-mass tags to include in the attribution surface.
        Default 3 — enough for an operator to see the prediction plus
        its closest competitors.
    """
    _ = column_features  # reserved for SHAP feature-aggregation use
    if not is_enabled(cfg):
        return None, "explicit_disable", None

    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        return None, "degraded_no_collection", None
    qdrant_url, collection = resolved

    # Distinguish "no qdrant-client lib" from "Qdrant down" — both are
    # degraded but require different operator action.
    try:
        import qdrant_client as _qdrant  # noqa: F401
    except ImportError:
        return None, "degraded_no_qdrant_client", None

    client = _get_qdrant_client(qdrant_url)
    if client is None:
        return None, "degraded_qdrant_connect", None

    index = _get_annotation_index(qdrant_url, collection)
    if index is None:
        return None, "degraded_load_failed", None

    try:
        from atelier.classify.late_interaction import (
            score_column_against_index,
        )
        from atelier.classify.mass_functions import late_interaction_to_mass
        from atelier.classify.multi_vector_features import build_column_query

        # Self-supply embedder when caller passes None — the pipeline
        # historically set cfg._embedder but that attribute was never
        # populated.  Lazy-loading via _get_model() is the canonical
        # path for the sentence-transformer encoder.
        _embed = embed
        if _embed is None:
            _embed = _get_default_embedder()

        query = build_column_query(
            column_name=column_name,
            table_name=table_name,
            samples=samples,
            neighbor_column_names=neighbor_column_names,
            pattern_summary=pattern_summary,
            embed=_embed,
        )
        weights = _scoring_weights_from_cfg(cfg)
        scores = score_column_against_index(query, index, weights)
        mass = late_interaction_to_mass(scores, frame)
        attribution = _build_attribution(
            mass=mass, scores=scores, frame=frame, top_k=attribution_top_k,
        )
        return mass, "ok", attribution
    except Exception as exc:  # noqa: BLE001 — bridge must not break the pipeline
        logger.warning(
            "late_interaction: scoring raised for %s.%s: %s; "
            "this is a deployment issue (investigate), not a normal flow",
            table_name, column_name, exc,
        )
        return None, "degraded_score_error", None


def _build_attribution(
    *,
    mass: "BeliefAssignment",
    scores: list,
    frame: "FrameOfDiscernment",
    top_k: int,
) -> dict:
    """Per-decision attribution: top-K post-fusion tags + their per-role breakdowns.

    The attribution surface answers "why did the late-interaction
    cosine source vote the way it did, on this column?" — operator-
    legible explanation per prediction, with no permutation-based
    re-scoring cost because the per-role components are already
    computed during MaxSim execution.

    Top-K selection is by **post-fusion tag mass** — both **singleton
    focal elements** (leaves) and **internal-node focal elements**
    (parent categories) qualify, per the every-tag-is-first-class
    policy that ``HierarchicalClassification.from_combined_evidence``
    already embodies.  An internal-node parent that carries
    significant fused mass appears in the attribution surface
    alongside leaves, with ``is_leaf: false`` marking the
    distinction.  This honors the operator's view of predictions:
    if the fusion lands on a parent category as the correct
    granularity, the attribution surface shows it.

    Returns a dict shaped for direct surfacing in the per-column
    result:

        {
          "top_k": [
            {
              "code": "EMAIL",
              "is_leaf": true,
              "post_fusion_mass": 0.737,
              "positive_score": 0.83,
              "negative_score": 0.05,
              "verifier_pass_rate": 1.0,
              "per_role": {
                "label": 0.78,
                "name_hints": 0.65,
                "prototype_values": 0.89,
                "value_patterns": 0.71,
                "context": 0.42,
                "parent_path": 0.31,
                "anti_examples": 0.05
              }
            },
            ...
          ],
          "ranking_basis": "post_fusion_tag_mass"
        }
    """
    # Index the TagScores by code for O(1) lookup during top-K selection.
    by_code = {s.code: s for s in scores}

    # FE → code lookup over both leaves and internal nodes.  FocalElement
    # equality is by codes-frozenset only (label is ignored in __eq__),
    # so Dempster-produced FEs (which carry no label) match frame.singletons
    # / frame.internal_nodes entries (which do).
    fe_to_code: dict["FocalElement", str] = {}
    for code, fe in frame.singletons.items():
        fe_to_code[fe] = code
    for code, fe in frame.internal_nodes.items():
        fe_to_code[fe] = code

    # Top-K by post-fusion mass over BOTH singletons and internal nodes.
    # Skip Θ (the ignorance slot is not a tag prediction) and any focal
    # element that doesn't map to a known tag (e.g., complement focal
    # elements emitted by the negative channel, or Dempster-produced
    # set intersections that don't correspond to a frame tag).
    tag_masses: list[tuple[str, float]] = []
    for fe, m in mass.masses.items():
        if fe == frame.theta:
            continue
        code = fe_to_code.get(fe)
        if code is None:
            continue
        if code not in by_code:
            # Tag is in the frame but the scoring pass didn't see it
            # (e.g., absent from the enriched annotation collection).
            # No per-role breakdown to surface — skip.
            continue
        tag_masses.append((code, m))
    tag_masses.sort(key=lambda kv: (-kv[1], kv[0]))
    top = tag_masses[:top_k]

    rows: list[dict] = []
    for code, post_mass in top:
        s = by_code[code]
        rows.append({
            "code": code,
            "is_leaf": code in frame.singletons,
            "post_fusion_mass": round(post_mass, 6),
            "positive_score": round(s.positive_score, 6),
            "negative_score": round(s.negative_score, 6),
            "verifier_pass_rate": round(
                getattr(s, "verifier_pass_rate", 1.0), 6,
            ),
            "per_role": {k: round(v, 6) for k, v in (s.per_role or {}).items()},
        })

    return {
        "top_k": rows,
        "ranking_basis": "post_fusion_tag_mass",
    }


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
