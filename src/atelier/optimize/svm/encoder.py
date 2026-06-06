"""Canonical ModernBERT encoder for the NHSVM head.

Single source of truth for ``encode_modernbert`` — the dense-embedding
encoder behind the factorized NHSVM head (``answerdotai/ModernBERT-base``,
mean-pooled, 768-dim).  Both the runtime inference path
(``atelier.classify.factorized_nhsvm.NHSVMHeadAdapter``) and the offline
training/refinement tooling (``atelier.optimize.svm.reference``) import
``encode_modernbert`` from here.

History: this lived in ``scripts/reflect_nhsvm_modernbert.py`` and was
reached at runtime via a ``sys.path.insert(scripts/)`` hack — tech debt
held over from adapting to UAT challenges.  It now lives in the package;
the script re-exports it for backward compatibility with the research
scripts that import ``from reflect_nhsvm_modernbert import encode_modernbert``.

The model + tokenizer are loaded once per ``(model_id, device)`` and
cached process-wide.  This matters: the runtime adapter encodes one
column at a time (``batch_size=1``), so without the cache every column
would reload the ~600 MB encoder from ``from_pretrained`` — the reason
the SVM channel was previously too slow to default-enable.  The cache
makes the per-column path a single load followed by cheap forward passes.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache

log = logging.getLogger(__name__)

MODEL_ID = "answerdotai/ModernBERT-base"
EMB_DIM = 768  # ModernBERT-base hidden size


def _resolve_device(device: str | None) -> str:
    if device is not None:
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=2)
def _load_encoder(model_id: str, device: str):
    """Load + cache ``(tokenizer, model)`` for a given ``(model_id, device)``.

    Cached process-wide so repeated single-text encodes (the runtime
    per-column path) reuse one loaded model instead of reloading it.
    The model is float32 and in eval mode on ``device``.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    log.info("loading encoder %s on %s (cached)", model_id, device)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float32)
    model.eval().to(device)
    return tok, model


def encode_modernbert(
    texts: list[str],
    *,
    batch_size: int = 64,
    max_length: int = 256,
    pooling: str = "mean",
    device: str | None = None,
):
    """Encode texts with ModernBERT-base; return (N, 768) float32 ndarray.

    ``pooling=mean`` uses attention-mask-weighted mean over token states
    (standard retrieval pooling).  ``pooling=cls`` uses the [CLS] token
    state directly.

    The encoder + tokenizer are loaded once per ``(model_id, device)``
    and reused across calls (see :func:`_load_encoder`).
    """
    import numpy as np
    import torch

    device = _resolve_device(device)
    tok, model = _load_encoder(MODEL_ID, device)
    log.info(
        "  encoder=%s  pooling=%s  device=%s  batch=%d  max_len=%d  n=%d",
        MODEL_ID, pooling, device, batch_size, max_length, len(texts),
    )

    out = np.empty((len(texts), EMB_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state  # (B, T, H)
            if pooling == "cls":
                vec = hidden[:, 0, :]
            elif pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                vec = summed / counts
            else:
                raise ValueError(f"unknown pooling: {pooling}")
            out[start:start + len(chunk)] = vec.cpu().numpy()
            if (start // batch_size) % 10 == 0 and len(texts) > batch_size:
                done = start + len(chunk)
                rate = done / max(time.time() - t0, 1e-6)
                log.info("    %d/%d (%.0f rows/s)", done, len(texts), rate)
    log.info("  encoded %d texts in %.1fs", len(texts), time.time() - t0)
    return out
