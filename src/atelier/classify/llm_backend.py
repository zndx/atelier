"""LLM backend abstraction for bootstrap classification.

Provides a unified interface for LLM-based column classification:
- Anthropic Structured (Claude) — default when ANTHROPIC_API_KEY is
  available.  Uses the Messages API ``output_config`` with JSON Schema
  for guaranteed valid structured output, plus prompt caching for the
  taxonomy system prompt.
- Anthropic (Claude) — free-text fallback via the anthropic SDK
- OpenAI-compatible (vLLM/GLM-4.7) via the openai SDK
- Cerebras (GLM-4.7 via OpenAI-compatible API)
- AWS Bedrock (Claude via the Converse API)

Each backend converts batch column metadata into structured
classification responses with token tracking for cost estimation.

Ported from signals/src/sigint/llm_backend.py, adapted for atelier's
HOCON config and ColumnSample type.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Anthropic structured output constants ────────────────────────

# JSON Schema for structured classification output.  The SDK enforces
# this via constrained decoding — no regex parsing needed.
_CLASSIFICATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "column_name": {"type": "string"},
        "category_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["code", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["column_name", "category_code", "confidence", "evidence", "alternatives"],
    "additionalProperties": False,
}

CLASSIFICATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": _CLASSIFICATION_ITEM_SCHEMA,
        },
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


# ── Cerebras constants ──────────────────────────────────────────

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_DEFAULT_MODEL = "zai-glm-4.7"


# ── Response types ───────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnClassification:
    """Single column classification from an LLM."""

    column_name: str
    category_code: str | None
    confidence: float
    evidence: str
    alternatives: list[dict] = field(default_factory=list)  # [{code, confidence}, ...]


@dataclass(frozen=True)
class LLMResponse:
    """Batch response from an LLM backend."""

    classifications: list[ColumnClassification]
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = "stop"

    @property
    def truncated(self) -> bool:
        """Whether response was truncated due to max_tokens limit."""
        return self.finish_reason == "length"


# ── Configuration ────────────────────────────────────────────────


@dataclass
class LLMBackendConfig:
    """Configuration for LLM backend."""

    backend: str = "openai_compatible"  # "anthropic" | "openai_compatible" | "cerebras" | "bedrock"
    api_key: str | None = None
    model: str = "glm-4.7"
    base_url: str | None = None  # e.g. "http://localhost:8000/v1"
    max_tokens: int = 65536
    temperature: float = 0.0
    batch_size: int = 10  # columns per LLM call
    max_retries: int = 3
    retry_delay: float = 2.0
    disable_reasoning: bool = False
    # AWS Bedrock credentials (used only by "bedrock" backend)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    aws_session_token: str | None = None


def config_from_atelier(cfg) -> LLMBackendConfig:
    """Build LLMBackendConfig from an AtelierConfig."""
    return LLMBackendConfig(
        backend=cfg.classify_llm_backend,
        api_key=cfg.classify_llm_api_key,
        model=cfg.classify_llm_model,
        base_url=cfg.classify_llm_base_url,
        max_tokens=cfg.classify_llm_max_tokens,
        temperature=cfg.classify_llm_temperature,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
        disable_reasoning=cfg.classify_llm_disable_reasoning,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        aws_region=cfg.aws_region,
        aws_session_token=cfg.aws_session_token,
    )


# ── Prompt building ──────────────────────────────────────────────


def build_category_table(category_set) -> str:
    """Build a markdown table of leaf categories for the system prompt."""
    lines = ["| Code | Label | Description |", "|------|-------|-------------|"]
    for cat in category_set.categories:
        desc = (cat.description or "")[:80]
        lines.append(f"| {cat.code} | {cat.label} | {desc} |")
    return "\n".join(lines)


def build_system_prompt(category_table: str) -> str:
    """Build the bootstrap classification system prompt."""
    return (
        "You are a data governance classification engine. Your task is to "
        "classify database columns into taxonomy categories based on column "
        "name, data type, sample values, and sibling context.\n"
        "\n"
        "## Categories\n"
        "\n"
        f"{category_table}\n"
        "\n"
        "## Instructions\n"
        "\n"
        "- Classify each column into exactly ONE leaf category.\n"
        "- Consider column name, data type, sample values, and sibling columns.\n"
        "- If no category fits, set category_code to null.\n"
        "- Provide confidence 0.0–1.0 and brief evidence.\n"
        "- For each column, list up to 3 alternative categories with confidence.\n"
        "- Respond with ONLY a JSON array, no markdown fencing.\n"
        "\n"
        "## Response Format\n"
        "\n"
        '[{"column_name": "ssn", "category_code": "ICE.SENSITIVE.PID.IDENTITY.GOVID", "confidence": 0.95, '
        '"evidence": "SSN pattern", "alternatives": [{"code": "ICE.SENSITIVE.PID.IDENTITY.FULLNAME", "confidence": 0.03}]}]'
    )


def build_batch_user_prompt(
    samples: list,
    revisit_context: dict[str, dict] | None = None,
    table_name: str | None = None,
) -> str:
    """Build a user prompt for a batch of columns.

    Args:
        samples: ColumnSample objects to classify.
        revisit_context: Optional per-column enrichment for revisit passes.
        table_name: Optional table name for context header.
    """
    parts: list[str] = []

    if table_name:
        parts.append(f"## Table: {table_name}\n")

    for i, sample in enumerate(samples, 1):
        revisit = revisit_context.get(sample.name) if revisit_context else None
        tag = " (REVISIT)" if revisit else ""
        lines = [f"### Column {i}: {sample.name}{tag}"]
        lines.append(f"Type: {sample.column_type or 'UNKNOWN'}")

        if sample.values:
            preview = sample.values[:10]
            lines.append(f"Values: {preview}")

        if sample.siblings:
            lines.append(f"Siblings: {sample.siblings}")

        if revisit:
            ml_pred = revisit.get("ml_prediction", "")
            bel = revisit.get("belief", 0.0)
            pl = revisit.get("plausibility", 0.0)
            k = revisit.get("conflict", 0.0)
            lines.append(f"ML prediction: {ml_pred} [Bel={bel:.2f}, Pl={pl:.2f}, K={k:.2f}]")
            if revisit.get("confusable"):
                lines.append(f"Confusable: {revisit['confusable']}")
            if revisit.get("previous"):
                prev = revisit["previous"]
                lines.append(
                    f"Your previous: {prev.get('code', '?')} (conf={prev.get('confidence', 0):.2f})"
                )

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _parse_classifications(text: str, expected_names: list[str]) -> list[ColumnClassification]:
    """Parse LLM JSON response into ColumnClassification list.

    Handles markdown fencing, partial JSON, and single-object responses.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return _dicts_to_classifications(data, expected_names)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Direct JSON parse failed, trying regex array extraction")

    # Regex fallback: extract JSON array
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                return _dicts_to_classifications(data, expected_names)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Regex array extraction failed, trying individual object extraction")

    # Last resort: extract individual JSON objects
    results = []
    for obj_match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            d = json.loads(obj_match.group())
            results.append(d)
        except (json.JSONDecodeError, ValueError):
            continue

    if results:
        return _dicts_to_classifications(results, expected_names)

    logger.warning("Failed to parse LLM response: %s", text[:200])
    return []


def _dicts_to_classifications(
    data: list[dict], expected_names: list[str],
) -> list[ColumnClassification]:
    """Convert parsed dicts to ColumnClassification objects."""
    results = []
    for i, item in enumerate(data):
        name = item.get("column_name", "")
        if not name and i < len(expected_names):
            name = expected_names[i]

        alternatives = []
        for alt in item.get("alternatives", []):
            if isinstance(alt, dict) and "code" in alt:
                alternatives.append({
                    "code": str(alt["code"]),
                    "confidence": float(alt.get("confidence", 0.0)),
                })

        results.append(ColumnClassification(
            column_name=str(name),
            category_code=item.get("category_code"),
            confidence=float(item.get("confidence", 0.0)),
            evidence=str(item.get("evidence", "")),
            alternatives=alternatives,
        ))
    return results


def _parse_structured_response(text: str, expected_names: list[str]) -> list[ColumnClassification]:
    """Parse structured JSON output guaranteed valid by schema.

    Shared by AnthropicStructuredBackend and BedrockStructuredBackend.
    """
    data = json.loads(text)
    items = data.get("classifications", [])
    return _dicts_to_classifications(items, expected_names)


# ── Abstract backend ─────────────────────────────────────────────


class LLMBackend(ABC):
    """Abstract LLM backend for batch column classification."""

    def __init__(self, config: LLMBackendConfig) -> None:
        self._config = config

    @abstractmethod
    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        """Classify a batch of columns.

        Args:
            samples: ColumnSample objects (up to batch_size).
            system_prompt: Pre-built system prompt with category table.
            revisit_context: Optional enrichment for revisit passes.
            table_name: Optional table name for prompt context.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the backend is reachable and functional."""


# ── Anthropic backend ────────────────────────────────────────────


class AnthropicBackend(LLMBackend):
    """Backend using the Anthropic Messages API (Claude)."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )

        if not self._config.api_key:
            raise ValueError("Anthropic API key required. Set ATELIER_LLM_API_KEY.")

        self._client = anthropic.Anthropic(api_key=self._config.api_key)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        response = client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text.strip()
        classifications = _parse_classifications(text, expected_names)

        return LLMResponse(
            classifications=classifications,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._config.model,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(resp.content) > 0
        except Exception:
            return False


# ── Anthropic structured output backend ──────────────────────────


class AnthropicStructuredBackend(LLMBackend):
    """Backend using the Anthropic Messages API with structured output.

    Uses ``output_config`` with JSON Schema for guaranteed valid responses
    (constrained decoding — no regex/JSON parsing fallbacks).  The system
    prompt is sent with ``cache_control`` for prompt caching (90% input
    token discount on subsequent calls in the same session).

    This is the default backend when ``ANTHROPIC_API_KEY`` is present
    and no explicit classify LLM backend is configured.
    """

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )

        if not self._config.api_key:
            raise ValueError("Anthropic API key required.")

        self._client = anthropic.Anthropic(api_key=self._config.api_key)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        response = client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            system=[{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_prompt}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": CLASSIFICATION_OUTPUT_SCHEMA,
                },
            },
        )

        text = response.content[0].text
        classifications = _parse_structured_response(text, expected_names)

        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        if cache_read:
            logger.debug("Prompt cache hit: %d tokens read from cache", cache_read)
        elif cache_write:
            logger.debug("Prompt cache miss: %d tokens written to cache", cache_write)

        return LLMResponse(
            classifications=classifications,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._config.model,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._config.model,
                max_tokens=32,
                messages=[{"role": "user", "content": "Respond with {\"ok\":true}"}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            return len(resp.content) > 0
        except Exception:
            return False


# ── OpenAI-compatible backend ────────────────────────────────────


class OpenAICompatibleBackend(LLMBackend):
    """Backend using OpenAI-compatible API (vLLM, GLM-4.7, etc.)."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            )

        kwargs: dict[str, Any] = {}
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        else:
            kwargs["api_key"] = "EMPTY"

        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url

        self._client = openai.OpenAI(**kwargs)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        api_params: dict = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self._config.disable_reasoning:
            api_params["extra_body"] = {"disable_reasoning": True}

        # Retry with exponential backoff for transient errors
        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = client.chat.completions.create(**api_params)
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                retryable = any(code in err_str for code in ("429", "502", "503", "504"))
                if retryable and attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._config.max_retries, delay, e,
                    )
                    time.sleep(delay)
                    continue
                raise
        else:
            raise last_error  # type: ignore[misc]

        text = response.choices[0].message.content or ""
        text = text.strip()

        finish_reason = response.choices[0].finish_reason or "stop"
        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        reasoning_tokens = 0
        if response.usage and response.usage.completion_tokens_details:
            reasoning_tokens = getattr(
                response.usage.completion_tokens_details, "reasoning_tokens", 0
            ) or 0

        if finish_reason == "length":
            logger.warning(
                "LLM response TRUNCATED: %d chars, in=%d, out=%d "
                "(reasoning=%d, max_tokens=%d) — increase max_tokens or reduce batch size",
                len(text), input_tokens, output_tokens,
                reasoning_tokens, self._config.max_tokens,
            )
        else:
            logger.info(
                "LLM response: %d chars, finish=%s, in=%d, out=%d (reasoning=%d)",
                len(text), finish_reason, input_tokens, output_tokens, reasoning_tokens,
            )

        classifications = _parse_classifications(text, expected_names)

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(resp.choices) > 0
        except Exception:
            return False


# ── Bedrock backend ─────────────────────────────────────────────


class BedrockBackend(LLMBackend):
    """Backend using AWS Bedrock Converse API."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package required for Bedrock. Install with: uv add boto3"
            )

        if not self._config.aws_access_key_id or not self._config.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required for Bedrock backend. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        session = boto3.Session(
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            aws_session_token=self._config.aws_session_token,
            region_name=self._config.aws_region or "us-east-1",
        )
        self._client = session.client("bedrock-runtime")
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        response = client.converse(
            modelId=self._config.model,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": [{"text": user_prompt}],
            }],
            inferenceConfig={
                "maxTokens": self._config.max_tokens,
                "temperature": self._config.temperature,
            },
        )

        text = response["output"]["message"]["content"][0]["text"].strip()
        classifications = _parse_classifications(text, expected_names)
        usage = response.get("usage", {})

        finish_reason = response.get("stopReason", "end_turn")
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        if finish_reason == "max_tokens":
            logger.warning(
                "Bedrock response TRUNCATED: %d chars, in=%d, out=%d, max_tokens=%d",
                len(text), input_tokens, output_tokens, self._config.max_tokens,
            )
        else:
            logger.info(
                "Bedrock response: %d chars, finish=%s, in=%d, out=%d",
                len(text), finish_reason, input_tokens, output_tokens,
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = client.converse(
                modelId=self._config.model,
                messages=[{
                    "role": "user",
                    "content": [{"text": "ping"}],
                }],
                inferenceConfig={"maxTokens": 10},
            )
            return "output" in response
        except Exception:
            return False


# ── Bedrock structured output backend ────────────────────────────


class BedrockStructuredBackend(LLMBackend):
    """Backend using Bedrock invoke_model with Anthropic Messages API + structured output.

    Uses ``invoke_model`` with the raw Anthropic Messages format to access
    ``output_config`` (constrained JSON schema decoding) which is NOT available
    through the Converse API or the ``AnthropicBedrock`` SDK wrapper.

    The system prompt is sent with ``cache_control`` for prompt caching
    (90% input token discount on cache hits, 5-min default TTL on Bedrock).

    This is the auto-default backend when AWS Bedrock credentials are
    present and no explicit classify LLM or ANTHROPIC_API_KEY is configured.
    """

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package required for Bedrock. Install with: uv add boto3"
            )

        if not self._config.aws_access_key_id or not self._config.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required for Bedrock backend. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        session = boto3.Session(
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            aws_session_token=self._config.aws_session_token,
            region_name=self._config.aws_region or "us-east-1",
        )
        self._client = session.client("bedrock-runtime")
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": CLASSIFICATION_OUTPUT_SCHEMA,
                },
            },
        }

        last_error: Exception | None = None
        for attempt in range(self._config.max_retries):
            try:
                response = client.invoke_model(
                    modelId=self._config.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body),
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                retryable = any(code in err_str for code in (
                    "ThrottlingException", "TooManyRequestsException",
                    "ServiceUnavailableException", "ModelTimeoutException",
                    "429", "503", "529",
                ))
                if retryable and attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Bedrock transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._config.max_retries, delay, e,
                    )
                    time.sleep(delay)
                    continue
                raise
        else:
            raise last_error  # type: ignore[misc]

        body = json.loads(response["body"].read())
        text = body["content"][0]["text"]
        classifications = _parse_structured_response(text, expected_names)

        usage = body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        stop_reason = body.get("stop_reason", "end_turn")

        if cache_read:
            logger.debug("Bedrock prompt cache hit: %d tokens read", cache_read)
        elif cache_write:
            logger.debug("Bedrock prompt cache miss: %d tokens written", cache_write)

        if stop_reason == "max_tokens":
            logger.warning(
                "Bedrock structured response TRUNCATED: %d chars, in=%d, out=%d",
                len(text), input_tokens, output_tokens,
            )
        else:
            logger.info(
                "Bedrock structured: %d chars, stop=%s, in=%d, out=%d "
                "(cache_read=%d, cache_write=%d)",
                len(text), stop_reason, input_tokens, output_tokens,
                cache_read, cache_write,
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=stop_reason,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Respond with {\"ok\":true}"}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
            }
            response = client.invoke_model(
                modelId=self._config.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )
            body = json.loads(response["body"].read())
            return len(body.get("content", [])) > 0
        except Exception:
            return False


# ── Factory ──────────────────────────────────────────────────────


def create_backend(config: LLMBackendConfig) -> LLMBackend:
    """Create an LLM backend from configuration.

    Raises ValueError on unknown backend type.
    """
    if config.backend == "anthropic_structured":
        return AnthropicStructuredBackend(config)
    if config.backend == "anthropic":
        return AnthropicBackend(config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleBackend(config)
    if config.backend == "cerebras":
        cerebras_config = LLMBackendConfig(
            backend="cerebras",
            api_key=config.api_key,
            model=config.model if config.model != "glm-4.7" else CEREBRAS_DEFAULT_MODEL,
            base_url=config.base_url or CEREBRAS_BASE_URL,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            disable_reasoning=config.disable_reasoning,
        )
        return OpenAICompatibleBackend(cerebras_config)
    if config.backend == "bedrock":
        return BedrockBackend(config)
    if config.backend == "bedrock_structured":
        return BedrockStructuredBackend(config)
    raise ValueError(
        f"Unknown LLM backend: {config.backend!r}. "
        f"Use 'anthropic_structured', 'anthropic', 'openai_compatible', "
        f"'cerebras', 'bedrock', or 'bedrock_structured'."
    )


def create_backend_from_cfg(cfg) -> LLMBackend:
    """Create an LLM backend from an AtelierConfig.

    Resolution order:
    1. Explicit classify LLM config (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
    2. ANTHROPIC_SUBAGENT_MODEL — backend inferred from model format
    3. Neither → ValueError (fail fast)
    """
    from atelier.config import is_bedrock_model

    # 1. Explicit classify LLM backend configured
    if cfg.classify_llm_api_key or cfg.classify_llm_base_url:
        return create_backend(config_from_atelier(cfg))

    # 2. Subagent model — infer backend from model identifier format
    if not cfg.classify_subagent_model:
        raise ValueError(
            "No classification LLM configured. "
            "Set ANTHROPIC_SUBAGENT_MODEL or ATELIER_LLM_API_KEY."
        )

    model = cfg.classify_subagent_model
    if is_bedrock_model(model):
        return _build_bedrock_backend(cfg, model)
    return _build_anthropic_backend(cfg, model)


def _build_anthropic_backend(cfg, model: str) -> LLMBackend:
    """Build an AnthropicStructuredBackend from config + model."""
    if not cfg.anthropic_api_key:
        raise ValueError(
            f"Model {model!r} requires ANTHROPIC_API_KEY (direct API)."
        )
    logger.info("Classification LLM: AnthropicStructured %s", model)
    return create_backend(LLMBackendConfig(
        backend="anthropic_structured",
        api_key=cfg.anthropic_api_key,
        model=model,
        max_tokens=8192,
        temperature=0.0,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
    ))


def _build_bedrock_backend(cfg, model: str) -> LLMBackend:
    """Build a BedrockStructuredBackend from config + model."""
    if not cfg.has_bedrock:
        raise ValueError(
            f"Model {model!r} requires AWS Bedrock credentials."
        )
    logger.info("Classification LLM: BedrockStructured %s", model)
    return create_backend(LLMBackendConfig(
        backend="bedrock_structured",
        model=model,
        max_tokens=8192,
        temperature=0.0,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        aws_region=cfg.aws_region,
        aws_session_token=cfg.aws_session_token,
    ))
