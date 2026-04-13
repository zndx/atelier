"""Cosine similarity classifier using sentence-transformers.

Lazy-loads all-MiniLM-L6-v2 (384-dim, normalized) and computes cosine
similarity between column feature embedding text and category reference
embeddings.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.classify.features import ColumnFeatures
    from atelier.classify.taxonomy import CategorySet

_model = None
_model_name: str = "all-MiniLM-L6-v2"
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the sentence-transformer model (thread-safe)."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for embedding classification. "
                    "Install with: pip install sentence-transformers"
                )
            _model = SentenceTransformer(_model_name)
    return _model


def set_model_name(name: str) -> None:
    """Override the default model name (before first use)."""
    global _model_name, _model
    with _model_lock:
        _model_name = name
        _model = None


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized embeddings."""
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def classify_cosine(
    features: ColumnFeatures,
    category_set: CategorySet,
    *,
    mask: dict[str, bool] | None = None,
) -> dict[str, float]:
    """Classify a column against all categories via cosine similarity.

    Args:
        features: Extracted column features.
        category_set: Reference categories with embedding_text.
        mask: Optional SAGE ablation mask.

    Returns:
        Dict of {category_code: cosine_similarity} for all categories.
    """
    query_text = features.to_embedding_text(mask)
    if not query_text:
        return {}

    ref_texts = [cat.embedding_text for cat in category_set.categories]
    ref_codes = [cat.code for cat in category_set.categories]

    if not ref_texts:
        return {}

    model = _get_model()
    all_texts = [query_text] + ref_texts
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    query_emb = embeddings[0]
    ref_embs = embeddings[1:]

    similarities = {}
    for i, code in enumerate(ref_codes):
        sim = float(sum(a * b for a, b in zip(query_emb, ref_embs[i])))
        similarities[code] = sim

    return similarities


def classify_cosine_batch(
    features_list: list[ColumnFeatures],
    category_set: CategorySet,
    *,
    mask: dict[str, bool] | None = None,
) -> list[dict[str, float]]:
    """Batch-classify multiple columns for efficiency.

    Encodes all queries and references in a single model.encode() call.
    """
    query_texts = [f.to_embedding_text(mask) for f in features_list]
    ref_texts = [cat.embedding_text for cat in category_set.categories]
    ref_codes = [cat.code for cat in category_set.categories]

    if not ref_texts:
        return [{} for _ in features_list]

    model = _get_model()
    all_texts = query_texts + ref_texts
    embeddings = model.encode(all_texts, normalize_embeddings=True)

    n_queries = len(query_texts)
    query_embs = embeddings[:n_queries]
    ref_embs = embeddings[n_queries:]

    results = []
    for qi in range(n_queries):
        if not query_texts[qi]:
            results.append({})
            continue
        similarities = {}
        for ri, code in enumerate(ref_codes):
            sim = float(sum(a * b for a, b in zip(query_embs[qi], ref_embs[ri])))
            similarities[code] = sim
        results.append(similarities)

    return results


def precompute_reference_embeddings(
    category_set: CategorySet,
    cache_path: str | Path | None = None,
) -> dict[str, list[float]]:
    """Precompute and optionally cache reference category embeddings.

    Returns {code: embedding_vector} dict.
    """
    import json

    if cache_path:
        cache_path = Path(cache_path)
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)

    ref_texts = [cat.embedding_text for cat in category_set.categories]
    ref_codes = [cat.code for cat in category_set.categories]

    embeddings = embed_texts(ref_texts)
    result = dict(zip(ref_codes, embeddings))

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)

    return result
