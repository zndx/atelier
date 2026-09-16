"""ColBERT-Zero late-interaction encoder for Qdrant MaxSim.

Federation encode/embeddings are ``lightonai/ColBERT-Zero`` (128-d token
vectors + mean ``agg``). ``colbert-ir/colbertv2.0`` and the ColPali/ColNomic
family are retired for this channel.

ModernBERT (``atelier.optimize.svm.encoder``) is a different model with a
different job — NHSVM classification — and is not used here.

Prompt alignment is mandatory: queries use ``prompt_name="query"``,
collection documents use ``"document"``.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

GURU_NOPYLATE = "#EM.00000002.NOPYLATE"
GURU_RETIRED = "#EM.00000003.RETIRED"

DEFAULT_MODEL = "lightonai/ColBERT-Zero"
EMBEDDING_DIM = 128
_RETIRED_NEEDLES = (
    "colbert-ir/colbertv2",
    "colbertv2.0",
    "nomic",
    "colnomic",
    "colpali",
    "colqwen",
    "vidore",
)

_model_name: str = DEFAULT_MODEL
_encoder: _ColBERTZeroEncoder | None = None
_lock = threading.Lock()
_infer_lock = threading.Lock()


def refuse_retired_embedding_model(model_name: str | None) -> None:
    """v2.0 / ColPali / ColNomic names are retired. Empty = ColBERT-Zero."""
    n = (model_name or "").strip().lower()
    if not n:
        return
    if n in {DEFAULT_MODEL.lower(), "colbert-zero", "colbert"}:
        return
    if any(needle in n for needle in _RETIRED_NEEDLES):
        raise RuntimeError(
            f"{GURU_RETIRED} {model_name} is retired for encode/embeddings. "
            f"Use {DEFAULT_MODEL} (ColBERT-Zero via pylate). "
            "ModernBERT remains the NHSVM encoder, not this channel."
        )


def _ensure_pylate() -> None:
    try:
        import pylate  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"{GURU_NOPYLATE} pylate is required for ColBERT-Zero.\n"
            "  Try: uv sync  (pylate>=1.3.4,<3)"
        ) from e


def _as_token_matrix(raw: object) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise RuntimeError(
            f"ColBERT-Zero encode returned shape {arr.shape}; expected (tokens, dim)"
        )
    return arr


class _ColBERTZeroEncoder:
    """pylate ColBERT-Zero: token vectors for Qdrant MaxSim."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        refuse_retired_embedding_model(model_name)
        _ensure_pylate()
        self.model_name = model_name
        self._device = device
        self._dim = EMBEDDING_DIM
        self._model: Any = None

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> Any:
        if self._model is None:
            from pylate import models

            logger.info("Loading ColBERT-Zero %s on %s", self.model_name, self._device)
            self._model = models.ColBERT(
                model_name_or_path=self.model_name,
                device=self._device,
            )
        return self._model

    def encode(
        self,
        texts: str | list[str],
        *,
        batch_size: int = 32,
        is_query: bool = False,
    ) -> list[np.ndarray]:
        """Encode text(s) into per-token ColBERT-Zero vectors.

        Each array has shape ``(num_tokens, 128)``. ``is_query`` selects
        the pylate query vs document prompt.
        """
        if isinstance(texts, str):
            texts = [texts]
        prompt_name = "query" if is_query else "document"
        out: list[np.ndarray] = []
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            with _infer_lock:
                raw = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    is_query=is_query,
                    prompt_name=prompt_name,
                    show_progress_bar=False,
                )
            for item in raw:
                out.append(_as_token_matrix(item))
        return out

    def encode_single(self, text: str, *, is_query: bool = False) -> np.ndarray:
        return self.encode(text, is_query=is_query)[0]


def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_encoder() -> _ColBERTZeroEncoder:
    """Lazy-load the ColBERT-Zero encoder (thread-safe singleton)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None:
            refuse_retired_embedding_model(_model_name)
            device = _get_device()
            logger.info("Loading ColBERT-Zero %s on %s", _model_name, device)
            _encoder = _ColBERTZeroEncoder(_model_name, device=device)
    return _encoder


def set_model_name(name: str) -> None:
    """Override the ColBERT model (before first use). Retired names fail."""
    global _model_name, _encoder
    refuse_retired_embedding_model(name)
    with _lock:
        if name == _model_name and _encoder is not None:
            return
        _model_name = name or DEFAULT_MODEL
        _encoder = None


def warmup() -> None:
    """Eagerly load ColBERT-Zero and validate with a probe encode."""
    encoder = get_encoder()
    result = encoder.encode_single("probe", is_query=False)
    if result.shape[1] != encoder.dim:
        raise RuntimeError(
            f"ColBERT-Zero probe produced dim={result.shape[1]}, "
            f"expected {encoder.dim}"
        )
    logger.info(
        "ColBERT-Zero warmup OK: probe produced %d tokens × %d dims",
        result.shape[0],
        result.shape[1],
    )
