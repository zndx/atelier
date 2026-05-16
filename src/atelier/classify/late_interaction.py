# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Late-interaction MaxSim execution for column→annotation scoring.

Given a Qdrant collection of enriched annotation profiles (one
multi-vector point per tag) and a per-column multi-vector query (built
by ``multi_vector_features.build_column_query``), compute the
ColBERT-style late-interaction score for every (column, tag) pair.

Design choices
--------------

The Qdrant collection holds at most a few hundred tags × ~20 vectors
per multi-slot ≈ tens of thousands of vectors at ~384 dimensions —
well under a gigabyte.  Rather than orchestrate Qdrant's native
multi-vector query API (which is tuned for ranked IR retrieval), we
**materialize** the annotation-side vectors into an in-memory
structure once per pipeline run and compute MaxSim in Python.  This:

- Preserves explicit per-role MaxSim breakdowns (positive channel,
  negative channel, per-named-vector subscore) that the mass
  function construction needs.
- Avoids round-tripping per-column queries through the Qdrant HTTP
  API at every classify call.
- Keeps the score arithmetic auditable — operators can reproduce a
  per-decision SHAP attribution by replaying the in-memory scoring.

The materialized form is also cheaper to keep warm across iterations
of the bootstrap loop than to refetch per iteration.

When the collection grows (Aegir-scale ontologies, multi-customer
deployments with overlay-extended taxonomies), this module gains a
streaming / Qdrant-native fast path; the in-memory path remains a
correctness reference and a fallback.

See ``docs/src/architecture/late-interaction-cosine.md`` § Late-
interaction execution.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

    from atelier.classify.multi_vector_features import ColumnMultiVectorQuery

logger = logging.getLogger(__name__)


# ── Materialized annotation view ──────────────────────────────────


@dataclass
class AnnotationView:
    """One annotation's named vectors + minimal payload, ready for scoring."""

    code: str
    label: str | None
    label_view: list[float] | None
    description_view: list[float] | None
    parent_path_view: list[float] | None
    prototype_values: list[list[float]] = field(default_factory=list)
    value_patterns: list[list[float]] = field(default_factory=list)
    name_hints: list[list[float]] = field(default_factory=list)
    anti_examples: list[list[float]] = field(default_factory=list)
    # Verifier pass rate from payload — used by the mass function to
    # attenuate evidence from poorly-verified annotations.
    verifier_pass_rate: float = 1.0


@dataclass
class AnnotationIndex:
    """In-memory materialization of an entire Qdrant annotation collection."""

    collection: str
    embedding_dim: int
    views: list[AnnotationView]

    @property
    def codes(self) -> list[str]:
        return [v.code for v in self.views]


def load_annotation_index(
    client: "QdrantClient",
    *,
    collection: str,
    batch: int = 256,
) -> AnnotationIndex:
    """Materialize the annotation collection from Qdrant.

    Reads vectors + payload for every point.  Designed to be called
    once per pipeline run (or once per process, with cache eviction
    when the registry rotates the ``current`` row).
    """
    next_offset = None
    views: list[AnnotationView] = []
    embedding_dim = 0

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch,
            offset=next_offset,
            with_payload=True,
            with_vectors=True,
        )
        for p in points:
            payload = p.payload or {}
            vec_map = p.vector or {}
            view = _payload_and_vectors_to_view(payload, vec_map)
            if view is None:
                continue
            views.append(view)
            # Capture dimensionality once from the first usable slot.
            if not embedding_dim and view.label_view:
                embedding_dim = len(view.label_view)
        if next_offset is None:
            break

    logger.info(
        "Materialized annotation index: collection=%s views=%d dim=%d",
        collection, len(views), embedding_dim,
    )
    return AnnotationIndex(
        collection=collection,
        embedding_dim=embedding_dim,
        views=views,
    )


def _payload_and_vectors_to_view(
    payload: dict,
    vec_map: dict,
) -> AnnotationView | None:
    """Translate one Qdrant point into an AnnotationView."""
    code = payload.get("code") or payload.get("mnemonic")
    if code is None:
        return None

    # Verifier pass rate: checks_passed / checks_total when both present.
    vr = payload.get("verifier_results") or {}
    passed = vr.get("checks_passed", 0)
    total = vr.get("checks_total", 0) or 1
    pass_rate = float(passed) / float(total)

    def _vec(name: str) -> list[float] | None:
        v = vec_map.get(name)
        if v is None:
            return None
        # Qdrant client may return either list[float] (single) or
        # list[list[float]] (multi-vector slot with one element).
        # Normalize to list[float] for single-vector slots.
        if v and isinstance(v[0], list):
            return list(v[0])
        return list(v)

    def _multi(name: str) -> list[list[float]]:
        v = vec_map.get(name)
        if v is None:
            return []
        if v and not isinstance(v[0], list):
            return [list(v)]
        return [list(x) for x in v]

    return AnnotationView(
        code=str(code),
        label=payload.get("label"),
        label_view=_vec("label_view"),
        description_view=_vec("description_view"),
        parent_path_view=_vec("parent_path_view"),
        prototype_values=_multi("prototype_values"),
        value_patterns=_multi("value_patterns"),
        name_hints=_multi("name_hints"),
        anti_examples=_multi("anti_examples"),
        verifier_pass_rate=pass_rate,
    )


# ── MaxSim arithmetic ─────────────────────────────────────────────


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) or 1.0


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity for two equal-dimension dense vectors."""
    return _dot(a, b) / (_norm(a) * _norm(b))


def _maxsim_one_query_vs_doc_list(
    query_vec: list[float],
    doc_vecs: list[list[float]],
) -> float:
    """MaxSim contribution of one query vector against a doc-side multi-vector slot.

    Returns 0.0 when the doc slot is empty — graceful degradation:
    absent doc-side evidence contributes nothing without polluting
    other roles.
    """
    if not query_vec or not doc_vecs:
        return 0.0
    return max(_cosine(query_vec, d) for d in doc_vecs)


def _maxsim_query_list_vs_doc_list(
    query_vecs: list[list[float]],
    doc_vecs: list[list[float]],
) -> tuple[float, int]:
    """Aggregate MaxSim across many query vectors against one doc slot.

    Returns ``(sum_of_maxsims, query_count)``.  Sum is the contribution;
    query_count is captured so the caller can normalize for fair
    cross-column comparison.
    """
    if not query_vecs or not doc_vecs:
        return 0.0, 0
    total = 0.0
    for q in query_vecs:
        total += _maxsim_one_query_vs_doc_list(q, doc_vecs)
    return total, len(query_vecs)


# ── Per-(column, tag) score breakdown ─────────────────────────────


@dataclass
class TagScore:
    """Late-interaction score breakdown for one (column, tag) pair.

    Carries the per-role components alongside the aggregated channels so
    SHAP-style per-decision attribution can inspect which views drove
    the score (and the mass function can compute reliability-shaping
    α_marg from the within-channel margin).
    """

    code: str
    positive_score: float
    negative_score: float
    per_role: dict[str, float] = field(default_factory=dict)
    verifier_pass_rate: float = 1.0


@dataclass
class ScoringWeights:
    """Per-role weights in the positive channel.

    Values follow the defaults in ``config/base.conf`` under
    ``classify.cosine.late_interaction.weight_*``.  Set any to 0.0 to
    disable that role.  Negative-channel weight (``weight_anti``) is
    applied to the anti-examples contribution.
    """

    weight_label: float = 0.20
    weight_name_hints: float = 0.15
    weight_prototype_values: float = 0.30
    weight_value_patterns: float = 0.15
    weight_context: float = 0.10
    weight_parent_path: float = 0.10
    weight_anti: float = 0.30


def score_column_against_index(
    query: "ColumnMultiVectorQuery",
    index: AnnotationIndex,
    weights: ScoringWeights | None = None,
) -> list[TagScore]:
    """Compute late-interaction scores for one column over every tag.

    Returns a list of :class:`TagScore`, one per annotation in the
    index.  Roles are normalized by query-vector count so columns with
    different sample populations compare fairly (standard ColBERT
    normalization).
    """
    weights = weights or ScoringWeights()
    out: list[TagScore] = []

    # Column-side query vectors
    name_query = query.name_view[0] if query.name_view else None
    sample_queries = query.sample_value_views
    context_query = query.context_view[0] if query.context_view else None
    pattern_query = query.pattern_view[0] if query.pattern_view else None

    for view in index.views:
        per_role: dict[str, float] = {}

        # ── name_view (single q × single d)
        if name_query and view.label_view and weights.weight_label > 0:
            s = _cosine(name_query, view.label_view)
            per_role["label"] = s
        else:
            per_role["label"] = 0.0

        # ── name_view × name_hints (single q × multi d)
        if name_query and view.name_hints and weights.weight_name_hints > 0:
            per_role["name_hints"] = _maxsim_one_query_vs_doc_list(
                name_query, view.name_hints
            )
        else:
            per_role["name_hints"] = 0.0

        # ── sample_value_views × prototype_values (multi q × multi d)
        if sample_queries and view.prototype_values and weights.weight_prototype_values > 0:
            total, n = _maxsim_query_list_vs_doc_list(
                sample_queries, view.prototype_values
            )
            per_role["prototype_values"] = total / max(n, 1)
        else:
            per_role["prototype_values"] = 0.0

        # ── sample_value_views × value_patterns (multi q × multi d)
        if sample_queries and view.value_patterns and weights.weight_value_patterns > 0:
            total, n = _maxsim_query_list_vs_doc_list(
                sample_queries, view.value_patterns
            )
            per_role["value_patterns"] = total / max(n, 1)
        else:
            per_role["value_patterns"] = 0.0

        # ── context_view × parent_path_view (single q × single d)
        if context_query and view.parent_path_view and weights.weight_context > 0:
            per_role["context"] = _cosine(context_query, view.parent_path_view)
        else:
            per_role["context"] = 0.0

        # ── pattern_view × value_patterns (single q × multi d)
        if pattern_query and view.value_patterns and weights.weight_parent_path > 0:
            per_role["parent_path"] = _maxsim_one_query_vs_doc_list(
                pattern_query, view.value_patterns
            )
        else:
            per_role["parent_path"] = 0.0

        positive = (
            weights.weight_label * per_role["label"]
            + weights.weight_name_hints * per_role["name_hints"]
            + weights.weight_prototype_values * per_role["prototype_values"]
            + weights.weight_value_patterns * per_role["value_patterns"]
            + weights.weight_context * per_role["context"]
            + weights.weight_parent_path * per_role["parent_path"]
        )

        # ── Negative channel: sample_value_views × anti_examples
        if sample_queries and view.anti_examples and weights.weight_anti > 0:
            total, n = _maxsim_query_list_vs_doc_list(
                sample_queries, view.anti_examples
            )
            neg_raw = total / max(n, 1)
        else:
            neg_raw = 0.0
        per_role["anti_examples"] = neg_raw
        negative = weights.weight_anti * neg_raw

        out.append(TagScore(
            code=view.code,
            positive_score=positive,
            negative_score=negative,
            per_role=per_role,
            verifier_pass_rate=view.verifier_pass_rate,
        ))

    return out
