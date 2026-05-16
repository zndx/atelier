# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Column-side multi-vector representation for late-interaction cosine.

Builds the per-column query vectors that pair with the enriched
annotation profiles in Qdrant (see
``docs/src/architecture/late-interaction-cosine.md`` § Late-interaction
execution).  Multi-vector representation means each *aspect* of the
column (name, each sample value, table context, extracted pattern view)
becomes its own query vector — late interaction's MaxSim then operates
on the bipartite (query, document) similarity matrix rather than a
single compressed comparison.

The representation is deliberately small and stateless: a
:class:`ColumnMultiVectorQuery` dataclass plus a builder function.  The
Qdrant late-interaction query path (see ``late_interaction.py``)
consumes this directly.

This module is independent of the existing single-vector cosine path —
the legacy ``features.ColumnFeatures.to_embedding_text`` remains in
force for the legacy cosine source.  Both paths coexist during the
A/B comparison window.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)


# ── Query representation ──────────────────────────────────────────


@dataclass
class ColumnMultiVectorQuery:
    """Per-column multi-vector query payload, ready for Qdrant late interaction.

    Each field is a list of embeddings (multi-vector); single-vector
    aspects (column name, table context, pattern view) carry exactly one
    embedding inside a list for shape uniformity in the MaxSim path.

    Attributes
    ----------
    name_view : one embedding of ``column_name + " in " + table_name``.
    sample_value_views : per-deduped-sample embeddings, capped by
        ``column_samples_max`` (default 50).
    context_view : one embedding of the comma-joined neighbor column
        names in the same table.  Empty list when no table context.
    pattern_view : one embedding of the extracted format hints
        (regex matches, length stats) summarized as a short string.
        Empty list when the pattern source didn't fire for this column.
    """

    name_view: list[list[float]]
    sample_value_views: list[list[float]] = field(default_factory=list)
    context_view: list[list[float]] = field(default_factory=list)
    pattern_view: list[list[float]] = field(default_factory=list)


# ── Embedding callable signature ──────────────────────────────────


# Same signature as the enrichment loop's EmbedFn: str → list[float],
# list[str] → list[list[float]].  Implementations are
# sentence-transformers wrappers.
EmbedFn = Callable[[str | list[str]], list[float] | list[list[float]]]


# ── Builders ──────────────────────────────────────────────────────


def build_column_query(
    *,
    column_name: str,
    table_name: str | None,
    samples: list[str | float | int | None],
    neighbor_column_names: list[str] | None = None,
    pattern_summary: str | None = None,
    embed: EmbedFn,
    samples_max: int = 50,
    dedup_samples: bool = True,
) -> ColumnMultiVectorQuery:
    """Assemble the column-side multi-vector query.

    Parameters
    ----------
    column_name, table_name
        Used to build the ``name_view`` embedding.
    samples
        Raw sample values from the column.  None values are dropped;
        non-string types are stringified.
    neighbor_column_names
        Other column names in the same table (excluding the current
        one).  Used for the ``context_view``.  Empty / None → no
        context vector emitted.
    pattern_summary
        Short text summary of pattern matches (e.g.,
        ``"matched: monetary_pattern; 87% of samples"``).  Empty /
        None → no pattern vector emitted.
    embed
        Embedding callable.
    samples_max
        Cap on the number of per-sample query vectors.  Higher values
        improve recall on heterogeneous columns at the cost of MaxSim
        compute.  Default 50.
    dedup_samples
        When True, deduplicate samples before embedding and keep the
        most frequent N (frequency-weighted retention).
    """
    # ── name_view (single)
    name_text = column_name.strip()
    if table_name:
        name_text = f"{name_text} in {table_name.strip()}"
    name_vec = embed(name_text)
    name_view = _wrap_single_vector(name_vec)

    # ── sample_value_views (multi)
    cleaned = _clean_samples(samples)
    if dedup_samples:
        cleaned = _dedup_top_n(cleaned, samples_max)
    else:
        cleaned = cleaned[:samples_max]
    if cleaned:
        sample_vecs = embed(cleaned)
        sample_views = _ensure_list_of_vectors(sample_vecs)
    else:
        sample_views = []

    # ── context_view (single, optional)
    context_views: list[list[float]] = []
    if neighbor_column_names:
        neighbor_text = "table columns: " + ", ".join(
            n.strip() for n in neighbor_column_names if n and n.strip()
        )
        if neighbor_text != "table columns: ":
            ctx_vec = embed(neighbor_text)
            context_views = _wrap_single_vector(ctx_vec)

    # ── pattern_view (single, optional)
    pattern_views: list[list[float]] = []
    if pattern_summary and pattern_summary.strip():
        pat_vec = embed(pattern_summary.strip())
        pattern_views = _wrap_single_vector(pat_vec)

    return ColumnMultiVectorQuery(
        name_view=name_view,
        sample_value_views=sample_views,
        context_view=context_views,
        pattern_view=pattern_views,
    )


# ── Helpers ───────────────────────────────────────────────────────


# Token-ish characters retained when normalizing sample values for
# dedup.  Numeric / alphabetic / common punctuation; everything else
# collapses to underscores.  Conservative — preserves enough signal
# that EMAIL vs PHONE samples remain distinguishable.
_DEDUP_NORMALIZE_RE = re.compile(r"\s+")


def _clean_samples(samples: list) -> list[str]:
    """Drop None / empty, stringify, strip whitespace.  Preserve order."""
    out: list[str] = []
    for v in samples:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        out.append(s)
    return out


def _dedup_top_n(values: list[str], n: int) -> list[str]:
    """Frequency-weighted dedup: keep the top-N most common samples.

    Preserves original ordering among tied counts so the result is
    deterministic.  Returns at most ``n`` distinct values.
    """
    counts = Counter(values)
    # Sort by (count desc, first-occurrence-index asc) for determinism.
    first_idx = {v: i for i, v in enumerate(values) if v not in counts or counts[v] > 0
                 and v not in (first_idx_seen := set())  # noqa: F841 — used as side-effect set
                 }
    # Cleaner second pass for first-occurrence ordering
    first_idx = {}
    for i, v in enumerate(values):
        if v not in first_idx:
            first_idx[v] = i
    ordered = sorted(
        counts.keys(),
        key=lambda v: (-counts[v], first_idx[v]),
    )
    return ordered[:n]


def _wrap_single_vector(vec) -> list[list[float]]:
    """Wrap a flat list[float] in an outer list for shape uniformity.

    Defensively accepts list-of-vectors (single-element) too.
    """
    if not vec:
        return []
    if isinstance(vec, list) and vec and isinstance(vec[0], list):
        return list(vec)
    return [list(vec)]


def _ensure_list_of_vectors(vecs) -> list[list[float]]:
    """Coerce ``embed(list[str])`` output into ``list[list[float]]``.

    Defensive: when the embedder returns a single vector for a
    single-element input list, wrap it.
    """
    if not vecs:
        return []
    if isinstance(vecs, list) and vecs and isinstance(vecs[0], list):
        return [list(v) for v in vecs]
    return [list(vecs)]
