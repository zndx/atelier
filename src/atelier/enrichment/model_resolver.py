# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Per-backend apex-reasoning-model resolution for annotation enrichment.

Enrichment provider co-locates with the classification path
(``cfg.classify_llm_backend``) — operators manage one set of
credentials, one cost regime, one billing surface.  Within that
backend, enrichment selects the strongest reasoning model available
because per-tag generation is single-shot and benefits from extended
reasoning on structural taxonomy judgments.

Selection priority (highest first):

1. ``cfg.enrichment_model_override`` (env: ``ATELIER_ENRICHMENT_MODEL``)
   — explicit operator choice, used verbatim.
2. Per-backend apex constant for backends where the platform owns the
   model identity (anthropic direct).
3. For backends where the model identity belongs to the operator's
   deployment (``openai_compatible``, ``cerebras``), fall through to
   ``cfg.classify_llm_model`` — the operator's choice of served model
   is the apex available to that endpoint.

Bedrock is deliberately rejected without an override.  Bedrock model
identities are AWS account + region + inference-profile-name specific;
there is no portable default constant that would be correct across
deployments, and silently falling back to a weaker model identifier
would contradict the strongest-reasoning-model discipline.  Failing
loudly with a remediation hint is the load-bearing contract — see
``feedback_no_silent_dst_degradation`` for the parallel rule on the
late-interaction cosine path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atelier.config import AtelierConfig

logger = logging.getLogger(__name__)


# Per-backend apex reasoning model.  Only populated for backends where
# the model identity is platform-owned (i.e., a stable, portable
# constant).  Bedrock is absent by design; openai_compatible/cerebras
# fall through to the operator's classify_llm_model selection.
_APEX_BY_BACKEND: dict[str, str] = {
    "anthropic": "claude-opus-4-7",
}


class EnrichmentModelError(RuntimeError):
    """Raised when the apex enrichment model cannot be resolved.

    Always carries an operator-facing remediation hint in the message
    — this exception is intended to surface as a deployment-readiness
    gate, not a runtime crash to be retried.
    """


def resolve_enrichment_model(cfg: "AtelierConfig") -> tuple[str, str]:
    """Resolve ``(backend, model_id)`` for enrichment.

    Returns the backend identifier (matching
    ``cfg.classify_llm_backend``) and the model id to pass to that
    backend's client.  Raises :class:`EnrichmentModelError` when the
    operator must make an explicit decision (currently: bedrock
    without ``model_override``).
    """
    backend = cfg.classify_llm_backend

    if cfg.enrichment_model_override:
        model = cfg.enrichment_model_override
        logger.info(
            "enrichment.model resolved via override: backend=%s model=%s",
            backend, _redact(model),
        )
        return backend, model

    if backend in _APEX_BY_BACKEND:
        model = _APEX_BY_BACKEND[backend]
        logger.info(
            "enrichment.model resolved via platform apex: backend=%s model=%s",
            backend, model,
        )
        return backend, model

    if backend == "bedrock":
        raise EnrichmentModelError(
            "Bedrock backend selected for enrichment but no model_override "
            "supplied.  Bedrock model identities are account + region + "
            "inference-profile specific and have no portable default.  "
            "Set ATELIER_ENRICHMENT_MODEL to the inference-profile ARN for "
            "the strongest reasoning Claude available in your Bedrock "
            "deployment (typically the latest Opus generation Bedrock "
            "hosts).  See config/base.conf [enrichment] for the env var "
            "binding."
        )

    # openai_compatible, cerebras, or any future operator-controlled
    # endpoint: the operator's classify model selection IS the apex
    # available to them on that endpoint.
    model = cfg.classify_llm_model
    if not model:
        raise EnrichmentModelError(
            f"Backend {backend!r} requires cfg.classify_llm_model to be "
            "set so enrichment can fall through to the operator's "
            "endpoint selection.  Got an empty classify_llm_model — "
            "configure classify.llm.model or set ATELIER_ENRICHMENT_MODEL."
        )
    logger.info(
        "enrichment.model resolved via classify_llm_model fallthrough: "
        "backend=%s model=%s",
        backend, _redact(model),
    )
    return backend, model


def _redact(model: str) -> str:
    """Truncate ARNs and long identifiers for log readability.

    Bedrock ARNs are ~100 chars; the model-name suffix is the useful
    part for human-readable logs.  Plain model names are returned as-is.
    """
    if len(model) <= 60:
        return model
    # Try to extract the trailing path component (works for ARNs and
    # OpenAI-style URLs alike).
    tail = model.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return f"...{tail}"
