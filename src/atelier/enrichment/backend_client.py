# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Single-shot LLM client adapter for annotation enrichment.

Enrichment is single-shot per taxonomy node — no batching, no
classify-style parsing, no halving-retry logic.  The
``classify.llm_backend`` abstraction is designed for many-columns-per-
call sweeps and would require bending its shape to fit enrichment, or
bending enrichment to fit its shape.  This module hosts a minimal
per-backend client adapter that reuses the same credential and
reasoning-mode conventions as the classify backends while exposing a
clean single-shot surface to the enrichment generator.

The function :func:`generate_text` is the single entry point:

    text = generate_text(
        backend="anthropic",
        model="claude-opus-4-7",
        cfg=cfg,
        system_prompt="...",
        user_prompt="...",
        max_tokens=8192,
        reasoning_budget=16384,
        temperature=0.0,
    )

Returns the model's textual response (post-thinking, post-reasoning).
Raises :class:`UnsupportedEnrichmentBackend` for backends not yet
wired; :class:`EnrichmentCallError` wraps provider-side failures.

Provider parity with the classify path is maintained by importing the
same reasoning helper (``_apply_thinking`` for Anthropic adaptive
thinking) so enrichment exercises the same reasoning surface as
classify.  The Bedrock path is deliberately left for a follow-up
since the model_resolver gates Bedrock-without-override at config
time — the operator must make an explicit Bedrock decision before
reaching this module.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from atelier.config import AtelierConfig

logger = logging.getLogger(__name__)


class UnsupportedEnrichmentBackend(NotImplementedError):
    """Backend identified by the resolver but not yet wired in this module."""


class EnrichmentCallError(RuntimeError):
    """Provider-side failure during enrichment generation.

    Carries the underlying exception for diagnostic logging; callers
    (the enrichment loop) decide whether to retry or mark the tag as
    degraded.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


def generate_text(
    *,
    backend: str,
    model: str,
    cfg: "AtelierConfig",
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    reasoning_budget: int,
    temperature: float,
) -> str:
    """Single-shot text generation routed to the active classify backend.

    Provider selection mirrors the classify path's credential
    conventions — no separate enrichment credentials.  Reasoning depth
    is enrichment-tuned (default 16384 tokens; see
    ``cfg.enrichment_reasoning_budget``) because per-tag generation
    benefits from extended deliberation on structural taxonomy
    judgments.
    """
    if backend == "anthropic":
        return _generate_anthropic(
            cfg=cfg, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, reasoning_budget=reasoning_budget,
            temperature=temperature,
        )
    if backend in ("openai_compatible", "cerebras"):
        return _generate_openai_compatible(
            cfg=cfg, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, reasoning_budget=reasoning_budget,
            temperature=temperature,
        )
    if backend == "bedrock":
        return _generate_bedrock(
            cfg=cfg, model=model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            max_tokens=max_tokens, reasoning_budget=reasoning_budget,
            temperature=temperature,
        )
    raise UnsupportedEnrichmentBackend(
        f"Backend {backend!r} is not supported for enrichment.  "
        "Supported: anthropic, openai_compatible, cerebras."
    )


# ── Anthropic direct ──────────────────────────────────────────────


def _generate_anthropic(
    *,
    cfg: "AtelierConfig",
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    reasoning_budget: int,
    temperature: float,
) -> str:
    try:
        import anthropic
        import httpx
    except ImportError as exc:
        raise EnrichmentCallError(
            "anthropic package required for Anthropic enrichment backend",
            cause=exc,
        )

    api_key = cfg.classify_llm_api_key
    if not api_key:
        raise EnrichmentCallError(
            "Anthropic API key required for enrichment.  Set "
            "ATELIER_LLM_API_KEY (shared with classify path)."
        )

    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=httpx.Timeout(connect=15.0, read=180.0, write=10.0, pool=5.0),
    )

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if reasoning_budget > 0:
        # Reuse the classify path's reasoning configurator — same shape
        # discipline (adaptive vs legacy thinking).
        from atelier.classify.llm_backend import _apply_thinking
        _apply_thinking(kwargs, model, reasoning_budget)

    try:
        response = client.messages.create(**kwargs)
    except Exception as exc:
        raise EnrichmentCallError(
            f"Anthropic enrichment call failed: {exc}",
            cause=exc,
        )

    text_block = next(
        (b for b in response.content if getattr(b, "type", None) == "text"),
        response.content[-1] if response.content else None,
    )
    text = (text_block.text if text_block else "").strip()
    logger.info(
        "enrichment.generate anthropic model=%s in_tokens=%s out_tokens=%s "
        "text_chars=%d",
        model,
        getattr(response.usage, "input_tokens", "?"),
        getattr(response.usage, "output_tokens", "?"),
        len(text),
    )
    return text


# ── OpenAI-compatible (incl. Cerebras) ────────────────────────────


def _generate_openai_compatible(
    *,
    cfg: "AtelierConfig",
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    reasoning_budget: int,
    temperature: float,
) -> str:
    try:
        import openai
    except ImportError as exc:
        raise EnrichmentCallError(
            "openai package required for OpenAI-compatible enrichment backend",
            cause=exc,
        )

    client_kwargs: dict[str, Any] = {}
    if cfg.classify_llm_api_key:
        client_kwargs["api_key"] = cfg.classify_llm_api_key
    else:
        client_kwargs["api_key"] = "EMPTY"
    if cfg.classify_llm_base_url:
        client_kwargs["base_url"] = cfg.classify_llm_base_url

    client = openai.OpenAI(**client_kwargs)

    api_params: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if reasoning_budget > 0:
        # OpenAI-compatible endpoints that support reasoning take the
        # budget via extra_body.  Stock vLLM 422s on unknown extra_body
        # keys; in that case the operator's endpoint simply doesn't
        # offer reasoning and the enrichment call falls back to
        # standard generation.  We don't retry here — enrichment is
        # single-shot and the loop's verifier-driven retry already
        # owns that surface.
        api_params["extra_body"] = {"reasoning_budget": reasoning_budget}

    try:
        response = client.chat.completions.create(**api_params)
    except Exception as exc:
        # Strip extra_body on reasoning rejection and try once more —
        # see classify path for the same one-shot peel.
        err_str = str(exc)
        unsupported = (
            "reasoning_budget" in err_str
            and ("422" in err_str or "unsupported" in err_str.lower())
        )
        if unsupported and "extra_body" in api_params:
            logger.warning(
                "OpenAI-compatible endpoint rejected reasoning_budget; "
                "retrying enrichment call without it.",
            )
            api_params.pop("extra_body", None)
            try:
                response = client.chat.completions.create(**api_params)
            except Exception as exc2:
                raise EnrichmentCallError(
                    f"OpenAI-compatible enrichment call failed (post-strip): {exc2}",
                    cause=exc2,
                )
        else:
            raise EnrichmentCallError(
                f"OpenAI-compatible enrichment call failed: {exc}",
                cause=exc,
            )

    msg = response.choices[0].message
    text = (msg.content or "").strip()
    in_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
    out_tokens = getattr(response.usage, "completion_tokens", 0) or 0
    logger.info(
        "enrichment.generate openai_compatible model=%s in_tokens=%s "
        "out_tokens=%s text_chars=%d",
        model, in_tokens, out_tokens, len(text),
    )
    return text


# ── AWS Bedrock ──────────────────────────────────────────────────


def _generate_bedrock(
    *,
    cfg: "AtelierConfig",
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    reasoning_budget: int,
    temperature: float,
) -> str:
    """Single-shot Bedrock invoke_model for enrichment.

    Reuses the same boto3 client conventions as
    ``atelier.classify.llm_backend.BedrockBackend`` — region inferred
    from the ARN where present, explicit connect/read timeouts,
    boto3's implicit retry disabled because the enrichment loop's
    verifier-driven retry already owns that surface.  Differences vs
    the classify path: no tool-use (we want raw JSON in the text
    block), no halving / per-call retry loop (single shot).

    The model is expected to be an inference-profile ARN
    (validated upstream by the resolver, which raises if
    ``model_override`` is unset on the bedrock backend).
    """
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError as exc:
        raise EnrichmentCallError(
            "boto3 required for Bedrock enrichment backend",
            cause=exc,
        )

    if not cfg.aws_access_key_id or not cfg.aws_secret_access_key:
        raise EnrichmentCallError(
            "AWS credentials required for Bedrock enrichment.  Set "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (shared with "
            "the classify path's Bedrock backend)."
        )

    # Region resolution mirrors BedrockBackend._get_client: prefer the
    # region embedded in the ARN, then cfg.aws_region, then us-east-1.
    from atelier.config import region_from_arn
    arn_region = region_from_arn(model)
    effective_region = arn_region or cfg.aws_region or "us-east-1"

    bcfg = BotoConfig(
        connect_timeout=15,
        read_timeout=180,
        retries={"max_attempts": 0},
    )
    session = boto3.Session(
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        aws_session_token=cfg.aws_session_token,
        region_name=effective_region,
    )
    client = session.client("bedrock-runtime", config=bcfg)

    # Clamp to the model's actual output ceiling.  Bedrock silently
    # truncates over-budget requests with no warning, so the same
    # ceiling table the classify path uses applies here.
    #
    # On Bedrock's invoke_model with anthropic_version, max_tokens is
    # the TOTAL output envelope (thinking + response).  When thinking
    # is enabled we need reasoning_budget + response_budget, clamped
    # to the model ceiling.
    from atelier.classify.llm_backend import bedrock_max_output_tokens
    model_ceiling = bedrock_max_output_tokens(model)

    use_thinking = reasoning_budget > 0
    if use_thinking:
        effective_max = min(reasoning_budget + max_tokens, model_ceiling)
    else:
        effective_max = min(max_tokens, model_ceiling)

    request_body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": effective_max,
        "temperature": 1.0 if use_thinking else temperature,
        "system": [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{"role": "user", "content": user_prompt}],
    }
    if use_thinking:
        # Bedrock invoke_model accepts the legacy thinking shape.
        # Adaptive thinking is a direct-API-only surface; Bedrock-
        # hosted Claude generations use the budget_tokens form.
        request_body["thinking"] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget,
        }

    import json as _json
    try:
        response = client.invoke_model(
            modelId=model,
            contentType="application/json",
            accept="application/json",
            body=_json.dumps(request_body),
        )
    except Exception as exc:
        raise EnrichmentCallError(
            f"Bedrock enrichment call failed: {exc}",
            cause=exc,
        )

    body = _json.loads(response["body"].read())
    # With thinking enabled the response contains a leading thinking
    # block, then a text block with the actual JSON output.  Pick the
    # last text block — robust to either thinking-on or thinking-off
    # response shapes.
    text_blocks = [b for b in body.get("content", []) if b.get("type") == "text"]
    text = (text_blocks[-1].get("text") if text_blocks else "").strip()

    usage = body.get("usage", {})
    logger.info(
        "enrichment.generate bedrock model=%s region=%s in_tokens=%s "
        "out_tokens=%s text_chars=%d thinking=%s",
        _short_model(model), effective_region,
        usage.get("input_tokens", "?"),
        usage.get("output_tokens", "?"),
        len(text),
        "on" if use_thinking else "off",
    )
    return text


def _short_model(model: str) -> str:
    """Trim ARNs for log readability."""
    if len(model) <= 60:
        return model
    tail = model.rsplit(":", 1)[-1].rsplit("/", 1)[-1]
    return f"...{tail}"
