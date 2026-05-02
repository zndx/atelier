# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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
        # Lazy model allocation: we only load one device eagerly, which
        # covers the common case where batches stay below the shard
        # threshold.  Additional replicas come online only if multi-GPU
        # sharding actually engages — saves tens of seconds of startup
        # when MiniLM-sized models never saturate a single device.
        self._models: list[Any] = [None] * len(self._devices)
        # Pick the device with the most free memory above a threshold
        # (multi-tenant GPU hosts often have one device occupied by
        # an unrelated training run; blindly grabbing devices[0] would
        # crash with CUDA OOM on those hosts).  Falls back to
        # devices[0] when probing fails or no device clears the
        # threshold — preserves prior behaviour on single-GPU hosts.
        self._eager_idx = self._pick_eager_device()
        if self._eager_idx != 0:
            logger.info(
                "MultiDeviceEncoder eagerly loading on device %d (%s) — "
                "more free memory than device 0",
                self._eager_idx, self._devices[self._eager_idx],
            )
        self._models[self._eager_idx] = self._load_one(self._eager_idx)
        self._pool: ThreadPoolExecutor | None = None

    def _pick_eager_device(self, min_free_gib: float = 1.5) -> int:
        """Return the index of the device with most free memory.

        Probes each CUDA device via ``torch.cuda.mem_get_info`` and
        picks the one with the most free bytes above the threshold.
        Falls back to index 0 when:

        - Probing fails (torch unavailable, no CUDA, mem_get_info missing).
        - ``self._devices`` is single-element (probing wouldn't help).
        - Devices are CPU only.
        - No device has at least ``min_free_gib`` free.

        The 1.5 GiB threshold is chosen to leave headroom for the
        MiniLM-L6 model load (~400 MB) plus typical batch activations
        (~500 MB) plus some spare.  For larger encoder models this
        should be raised.
        """
        if len(self._devices) <= 1:
            return 0
        if not all(d.startswith("cuda") for d in self._devices):
            return 0
        try:
            import torch
        except Exception:
            return 0
        if not torch.cuda.is_available():
            return 0
        if not hasattr(torch.cuda, "mem_get_info"):
            return 0

        threshold_bytes = int(min_free_gib * (1024 ** 3))
        best_idx, best_free = 0, -1
        for i, dev in enumerate(self._devices):
            try:
                # mem_get_info accepts either a device index (int) or a
                # torch.device.  Devices in self._devices are strings
                # like "cuda:0" — pass the integer parsed out.
                idx = int(dev.split(":")[1]) if ":" in dev else i
                free, _total = torch.cuda.mem_get_info(idx)
            except Exception as exc:
                logger.debug(
                    "MultiDeviceEncoder: mem probe failed for %s: %s", dev, exc,
                )
                continue
            if free > best_free:
                best_free, best_idx = free, i
        if best_free < threshold_bytes:
            logger.warning(
                "MultiDeviceEncoder: no device has >= %.1f GiB free "
                "(best: %.2f GiB on %s); loading on devices[0] anyway",
                min_free_gib, best_free / (1024 ** 3), self._devices[best_idx],
            )
            return 0
        return best_idx

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
        # Use ``self._eager_idx`` rather than 0 because the eager device
        # may not be index 0 — the constructor probes for the device
        # with most free memory.
        if len(self._devices) == 1 or len(texts) < self._shard_threshold:
            return self._models[self._eager_idx].encode(
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


def warmup() -> None:
    """Eagerly load the embedding model and validate via a probe encode.

    Defense against silent first-run model downloads on restricted-egress
    deployments (e.g. CAI):

    - Triggers the SentenceTransformer download into the local HF cache
      if it isn't already there.
    - Performs a one-token probe encode to confirm the model is
      genuinely loadable, not just present-on-disk-but-corrupt.
    - Raises a clear ``RuntimeError`` on failure rather than letting
      a half-loaded model silently fail at first cosine-evidence call
      hours into a pipeline run.

    Idempotent — subsequent calls hit the same singleton via
    ``_get_model()`` without retrying the download.

    Operators should call this from install scripts (post-pip, to
    pre-populate the HF cache during build) and from start-app.sh
    (post-preflight, to fail-fast on a missing or corrupt cache before
    the FastAPI server starts accepting requests).
    """
    try:
        model = _get_model()
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {_model_name!r} unavailable — check HF cache "
            f"or network egress.  On CAI deployments, ensure the model "
            f"was pre-downloaded during install (scripts/install_deps.py) "
            f"and that HF_HUB_OFFLINE=1 is exported only AFTER the cache "
            f"is populated.  Original error: {exc}"
        ) from exc
    try:
        probe = model.encode(
            ["warmup"], normalize_embeddings=True, batch_size=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model {_model_name!r} loaded but probe encode "
            f"failed — model artifact may be corrupt or incompatible "
            f"with the installed sentence-transformers.  Original error: "
            f"{exc}"
        ) from exc
    # Defensive: confirm we got a non-trivial embedding back.
    try:
        n_dims = len(probe[0])
    except Exception:
        n_dims = -1
    if n_dims <= 0:
        raise RuntimeError(
            f"Embedding probe returned an unexpected shape (n_dims={n_dims}) "
            f"for model {_model_name!r}; treating as load failure."
        )
    logger.info(
        "Embedding model warmup complete: %s (%d dims, devices=%s)",
        _model_name, n_dims, _devices,
    )


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


# Order MUST match what ``CatBoostColumnClassifier`` expects in its
# concatenated input — and what
# ``shap_explanations._build_feature_groups`` slices by.  The three
# constants below are the SINGLE source of truth; refactoring the
# input shape means updating exactly these.

# Text features get their own SentenceTransformer encoding (one
# 384-dim slice each).  Order is fixed to keep slice offsets stable.
EMBED_TEXT_FEATURES: list[str] = [
    "column_name",
    "column_type",
    "sample_values",
    "pattern_signals",
    "sibling_context",
    "value_description",
    "source_table",
]

# Scalar features pass through as native CatBoost numerical inputs.
# Missing values become NaN — CatBoost handles them natively.
EMBED_SCALAR_FEATURES: list[str] = [
    "cardinality",
    "null_ratio",
    "value_entropy",
    "avg_value_length",
    "numeric_ratio",
]


def feature_input_groups(emb_dim: int = 384) -> list[tuple[str, int, int]]:
    """Return ``(feature_name, start, end)`` slices into the concatenated
    CatBoost input matrix produced by :func:`embed_features`.

    Used by:
    - ``CatBoostColumnClassifier.fit`` to set ``feature_names`` so trees
      and SHAP both see human-readable feature identifiers
    - ``shap_explanations._build_feature_groups`` to sum per-slice SHAP
      values back into per-feature attribution

    The slice ordering is fixed: 7 text features × ``emb_dim`` dims
    each, then 5 scalar features × 1 dim each.  Total dims:
    ``7 * emb_dim + 5`` (= 2693 at the default 384-dim encoder).
    """
    groups: list[tuple[str, int, int]] = []
    offset = 0
    for name in EMBED_TEXT_FEATURES:
        groups.append((name, offset, offset + emb_dim))
        offset += emb_dim
    for name in EMBED_SCALAR_FEATURES:
        groups.append((name, offset, offset + 1))
        offset += 1
    return groups


def embed_features(features_list) -> "Any":
    """Encode a list of ``ColumnFeatures`` into the structured input matrix.

    Each row is the concatenation of:

    - 7 SentenceTransformer slices (column_name, column_type,
      sample_values, pattern_signals, sibling_context,
      value_description, source_table), each of size ``emb_dim``
      (384 for the default MiniLM-L6 encoder).
    - 5 scalar slots (cardinality, null_ratio, value_entropy,
      avg_value_length, numeric_ratio).  ``None`` becomes ``NaN``;
      CatBoost handles missing values natively.

    Total per-row dimension: ``7 * emb_dim + 5``.  The ordering
    matches :func:`feature_input_groups` so TreeSHAP can be sliced
    back to per-feature attribution.

    Empty text segments (e.g. column has no sibling context, no
    pattern matches) still occupy their slice but encode to a
    consistent "empty-text" embedding from MiniLM.  Tree splits
    learn to recognize that as "feature unavailable" rather than
    requiring a sentinel value.

    Encoding cost: one batched ``encode`` call on
    ``len(features_list) * len(EMBED_TEXT_FEATURES)`` strings.  The
    ``MultiDeviceEncoder`` already batches optimally; the 7×
    increase over the legacy single-text path is amortized by the
    batched call.
    """
    import numpy as np

    n = len(features_list)
    if n == 0:
        return np.zeros((0, 7 * 384 + 5), dtype=np.float32)

    # Collect per-feature texts in feature-major order so we can flatten
    # for a single batched encode and reshape back cleanly.
    segments_per_row = [f.to_embedding_segments() for f in features_list]

    # Flatten: [row0_feat0, row0_feat1, ..., row0_featK, row1_feat0, ...]
    # Then reshape encoded result to (n, K, emb_dim).  Row-major flatten
    # keeps per-row slices contiguous for fast np.concatenate at the end.
    flat_texts: list[str] = []
    for row in segments_per_row:
        for fname in EMBED_TEXT_FEATURES:
            flat_texts.append(str(row.get(fname, "") or ""))

    model = _get_model()
    flat_emb = model.encode(
        flat_texts, normalize_embeddings=True, batch_size=_batch_size,
    )
    flat_emb = np.asarray(flat_emb, dtype=np.float32)
    emb_dim = flat_emb.shape[1] if flat_emb.size else 384

    # (n, K, emb_dim) → (n, K * emb_dim)
    text_block = flat_emb.reshape(n, len(EMBED_TEXT_FEATURES), emb_dim)
    text_block = text_block.reshape(n, len(EMBED_TEXT_FEATURES) * emb_dim)

    # Scalar block — same row order, NaN for missing.
    scalar_block = np.full(
        (n, len(EMBED_SCALAR_FEATURES)), fill_value=np.nan, dtype=np.float32,
    )
    for i, row in enumerate(segments_per_row):
        for j, fname in enumerate(EMBED_SCALAR_FEATURES):
            v = row.get(fname)
            if v is not None:
                scalar_block[i, j] = float(v)

    return np.concatenate([text_block, scalar_block], axis=1)


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
