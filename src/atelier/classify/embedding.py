"""Cosine similarity classifier using sentence-transformers.

Lazy-loads all-MiniLM-L6-v2 (384-dim, normalized) and computes cosine
similarity between column feature embedding text and category reference
embeddings.

GPU acceleration: call ``configure(device="auto")`` before first use.

On a single GPU, the model loads on the device and batch sizes auto-scale
(32→256).  On multiple GPUs, a ``MultiDeviceEncoder`` pool holds one
model instance per device; large encode calls fan out across devices
via a thread pool, and small calls (below the configured shard
threshold) take the single-device fast path.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.classify.features import ColumnFeatures
    from atelier.classify.taxonomy import CategorySet

logger = logging.getLogger(__name__)

_model: Any = None
_model_name: str = "all-MiniLM-L6-v2"
_model_lock = threading.Lock()

# Device and batch size — set via configure() before first model load
_device: str = "cpu"
_devices: list[str] = ["cpu"]
_batch_size: int = 32
_shard_threshold: int = 1500


class MultiDeviceEncoder:
    """Pool of SentenceTransformer instances, one per device.

    Exposes the subset of the SentenceTransformer API the pipeline uses
    (``encode``), so it's a drop-in replacement in _get_model() sites.
    When only one device is configured, behaves exactly like the
    underlying SentenceTransformer (no thread overhead).
    """

    def __init__(
        self,
        model_name: str,
        devices: list[str],
        batch_size: int,
        shard_threshold: int,
    ):
        self._model_name = model_name
        self._devices = list(devices)
        self._batch_size = batch_size
        self._shard_threshold = shard_threshold
        # Lazy model allocation: we only load cuda:0 (or cpu) eagerly, which
        # covers the common case where batches stay below the shard
        # threshold.  Additional replicas come online only if multi-GPU
        # sharding actually engages — saves tens of seconds of startup
        # when MiniLM-sized models never saturate a single device.
        self._models: list[Any] = [None] * len(self._devices)
        self._models[0] = self._load_one(0)
        self._pool: ThreadPoolExecutor | None = None

    def _load_one(self, idx: int):
        """Load one SentenceTransformer on ``self._devices[idx]``."""
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(self._model_name, device=self._devices[idx])

    def _ensure_pool_ready(self) -> None:
        """Lazily load the remaining replicas + thread pool on first shard."""
        if self._pool is not None:
            return
        for i, m in enumerate(self._models):
            if m is None:
                self._models[i] = self._load_one(i)
        self._pool = ThreadPoolExecutor(
            max_workers=len(self._models), thread_name_prefix="emb"
        )
        logger.info(
            "Multi-device sharding engaged: %d replicas warmed", len(self._models),
        )

    @property
    def device_count(self) -> int:
        return len(self._devices)

    def encode(
        self,
        texts,
        *,
        normalize_embeddings: bool = True,
        batch_size: int | None = None,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ):
        """Sharded encode; single-device fast path for small inputs.

        ``texts`` may be a list or a single string — we mirror the
        upstream API so swapping in this encoder is invisible.
        """
        import numpy as np

        bs = batch_size if batch_size is not None else self._batch_size

        # Single-device model or small batch → direct encode.
        # Only one device configured, or small batch → single-device path.
        if len(self._devices) == 1 or len(texts) < self._shard_threshold:
            return self._models[0].encode(
                texts, normalize_embeddings=normalize_embeddings,
                batch_size=bs, show_progress_bar=show_progress_bar,
                convert_to_numpy=convert_to_numpy,
            )

        # Shard across all configured devices — warm remaining replicas
        # on first use.
        self._ensure_pool_ready()
        n = len(texts)
        n_shards = len(self._models)
        shard_sizes = [n // n_shards] * n_shards
        for i in range(n % n_shards):
            shard_sizes[i] += 1
        offsets = [0]
        for s in shard_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        shards = [
            texts[off:off + sz] for off, sz in zip(offsets, shard_sizes) if sz > 0
        ]

        def _encode_one(args):
            model, shard = args
            return model.encode(
                shard, normalize_embeddings=normalize_embeddings,
                batch_size=bs, show_progress_bar=False,
                convert_to_numpy=convert_to_numpy,
            )

        results = list(self._pool.map(_encode_one, zip(self._models, shards)))
        return np.concatenate(results, axis=0) if results else np.zeros((0,))


def configure(
    device: str = "auto",
    batch_size: int = 32,
    *,
    devices: list[str] | None = None,
    shard_threshold: int = 1500,
) -> None:
    """Set device(s), batch size and shard threshold before first model load.

    Args:
        device: ``"auto"`` detects CUDA via preflight_gpu(), or explicit
                ``"cuda"``/``"cpu"``.  Used when ``devices`` is None.
        batch_size: Base batch size for model.encode().  Auto-scaled to
                    256 when GPU is detected and batch_size <= 64.
        devices: Explicit device list (``["cuda:0","cuda:1",…]``).  When
                 None and ``device="auto"`` detects CUDA, uses every
                 visible GPU.  Respects ``CUDA_VISIBLE_DEVICES``.
        shard_threshold: Minimum number of texts per encode() to trigger
                         multi-GPU fan-out.  Below this, the single-device
                         fast path avoids thread-coordination overhead.
    """
    global _device, _devices, _batch_size, _shard_threshold, _model

    if devices is not None and len(devices) > 0:
        # Operator-pinned device list wins.
        _devices = list(devices)
        _device = _devices[0]
    elif device == "auto":
        from atelier.classify.gpu import preflight_gpu
        gpu = preflight_gpu()
        if gpu.warnings:
            for w in gpu.warnings:
                logger.warning("GPU: %s", w)
        _devices = gpu.resolved_devices
        _device = _devices[0]
        logger.info(
            "Embedding device: %s across %d device(s) (%s)",
            _device, len(_devices), gpu.summary(),
        )
    else:
        _device = device
        _devices = [device]
        logger.info("Embedding device: %s (explicit)", _device)

    _batch_size = batch_size
    if _device.startswith("cuda") and _batch_size <= 64:
        _batch_size = 256
        logger.info("Embedding batch size auto-scaled to %d for GPU", _batch_size)

    _shard_threshold = shard_threshold

    with _model_lock:
        _model = None


def _get_model():
    """Lazy-load the sentence-transformer encoder (thread-safe).

    Returns a :class:`MultiDeviceEncoder` that wraps one or more
    ``SentenceTransformer`` instances.  Callers should only use the
    ``.encode(...)`` method; the encoder mirrors its signature.
    """
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            try:
                import sentence_transformers  # noqa: F401
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for embedding classification. "
                    "Install with: pip install sentence-transformers"
                )
            _model = MultiDeviceEncoder(
                model_name=_model_name,
                devices=_devices,
                batch_size=_batch_size,
                shard_threshold=_shard_threshold,
            )
            if _model.device_count > 1:
                logger.info(
                    "MultiDeviceEncoder loaded: %d devices (%s), shard_threshold=%d",
                    _model.device_count, _devices, _shard_threshold,
                )
    return _model


def set_model_name(name: str) -> None:
    """Override the default model name (before first use)."""
    global _model_name, _model
    with _model_lock:
        _model_name = name
        _model = None


def get_batch_size() -> int:
    """Return the resolved batch size (for use by external callers like SAGE)."""
    return _batch_size


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode texts into normalized embeddings."""
    model = _get_model()
    embeddings = model.encode(
        texts, normalize_embeddings=True, batch_size=_batch_size,
    )
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
    embeddings = model.encode(
        all_texts, normalize_embeddings=True, batch_size=_batch_size,
    )

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
    embeddings = model.encode(
        all_texts, normalize_embeddings=True, batch_size=_batch_size,
    )

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
