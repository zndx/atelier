# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""ColBERT late-interaction encoder for Qdrant multi-vector scoring.

Loads a ColBERT model (BERT backbone + 768→128 linear projection) and
produces per-token 128-dim vectors suitable for Qdrant's native MaxSim.

The entity side feeds ``ColumnFeatures.to_embedding_text()`` — the same
text SAGE/SHAP ablate over.  The annotation side feeds a composed text
from the enrichment payload.  Both go through the same encoder; Qdrant
handles the late-interaction MaxSim at query time.

Parallel to ``embedding.py`` (MiniLM single-vector encoder) which
remains for CatBoost/SVM feature embedding.  The two models serve
different purposes and coexist.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "colbert-ir/colbertv2.0"
_PROJECTION_DIM = 128

_model_name: str = _DEFAULT_MODEL
_encoder: _ColBERTEncoder | None = None
_lock = threading.Lock()


class _ColBERTEncoder:
    """Thread-safe ColBERT encoder wrapping BERT + linear projection."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = device
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        self._bert = AutoModel.from_pretrained(model_name)
        self._bert.eval()
        self._bert.to(device)

        checkpoint_dir = Path(
            self._bert.config._name_or_path
            if hasattr(self._bert.config, "_name_or_path")
            else model_name
        )
        self._projection = self._load_projection(model_name, device)
        self._dim = self._projection.shape[0]

    def _load_projection(self, model_name: str, device: str):
        """Load the linear.weight projection from the model checkpoint."""
        import torch
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "model.safetensors")
        from safetensors import safe_open

        with safe_open(path, framework="pt", device=device) as f:
            weight = f.get_tensor("linear.weight")
        return weight

    @property
    def dim(self) -> int:
        return self._dim

    def encode(
        self,
        texts: str | list[str],
        *,
        batch_size: int = 32,
    ) -> list[np.ndarray]:
        """Encode text(s) into per-token ColBERT vectors.

        Returns a list of arrays, one per input text.  Each array has
        shape ``(num_tokens, dim)`` where dim is the projection
        dimensionality (128 for ColBERTv2).

        Special tokens ([CLS], [SEP], [PAD]) are stripped from the
        output — only content tokens contribute to MaxSim.
        """
        import torch

        if isinstance(texts, str):
            texts = [texts]

        all_results: list[np.ndarray] = []

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
                return_attention_mask=True,
                return_special_tokens_mask=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._bert(**{
                    k: v for k, v in encoded.items()
                    if k in ("input_ids", "attention_mask", "token_type_ids")
                })
                hidden = outputs.last_hidden_state  # (batch, seq, 768)
                projected = hidden @ self._projection.T  # (batch, seq, 128)
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

            attention_mask = encoded["attention_mask"]
            special_mask = encoded["special_tokens_mask"]
            content_mask = attention_mask & (~special_mask.bool()).long()

            for i in range(projected.shape[0]):
                mask = content_mask[i].bool()
                tokens = projected[i][mask].cpu().numpy()
                if tokens.shape[0] == 0:
                    tokens = projected[i][:1].cpu().numpy()
                all_results.append(tokens)

        return all_results

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text, returning shape ``(num_tokens, dim)``."""
        return self.encode(text)[0]


def _get_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def get_encoder() -> _ColBERTEncoder:
    """Lazy-load the ColBERT encoder (thread-safe singleton)."""
    global _encoder
    if _encoder is not None:
        return _encoder
    with _lock:
        if _encoder is None:
            device = _get_device()
            logger.info("Loading ColBERT encoder %s on %s", _model_name, device)
            _encoder = _ColBERTEncoder(_model_name, device=device)
            logger.info(
                "ColBERT encoder ready: dim=%d, device=%s",
                _encoder.dim,
                device,
            )
    return _encoder


def set_model_name(name: str) -> None:
    """Override the ColBERT model (before first use).

    No-op when the requested name matches the currently-loaded model —
    avoids invalidating the cached encoder on every call from per-row
    inference paths (the bridge calls this for each cosine query).
    """
    global _model_name, _encoder
    with _lock:
        if name == _model_name and _encoder is not None:
            return
        _model_name = name
        _encoder = None


def warmup() -> None:
    """Eagerly load the ColBERT model and validate with a probe encode."""
    encoder = get_encoder()
    result = encoder.encode_single("probe")
    if result.shape[1] != encoder.dim:
        raise RuntimeError(
            f"ColBERT probe produced dim={result.shape[1]}, "
            f"expected {encoder.dim}"
        )
    logger.info(
        "ColBERT warmup OK: probe produced %d tokens × %d dims",
        result.shape[0],
        result.shape[1],
    )
