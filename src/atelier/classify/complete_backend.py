"""Classification LLM via Engine/Complete (thinking / instruct).

No Anthropic, Bedrock, or Agent SDK. Structured JSON through Complete
json_schema. Revisit passes use thinking; first sweep uses instruct.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from atelier.classify.llm_backend import (
    CLASSIFICATION_OUTPUT_SCHEMA,
    LLMBackend,
    LLMBackendConfig,
    LLMResponse,
    build_batch_user_prompt,
    _parse_classifications,
    _parse_structured_response,
)
from atelier.flows.classify_phases import llm_capability_for_sweep
from atelier.flows.lattice import complete as lattice_complete

logger = logging.getLogger(__name__)


class CompleteLLMBackend(LLMBackend):
    """Batch classify through zndx.engine.v1.Engine/Complete."""

    def __init__(
        self,
        config: LLMBackendConfig | None = None,
        *,
        complete_fn: Callable[..., Any] | None = None,
        forwarder: Any | None = None,
    ) -> None:
        super().__init__(config or LLMBackendConfig(backend="engine_complete"))
        self._complete_fn = complete_fn
        self._forwarder = forwarder

    @classmethod
    def from_cfg(cls, cfg: Any) -> CompleteLLMBackend:
        return cls(LLMBackendConfig(
            backend="engine_complete",
            model="instruct",
            max_tokens=int(getattr(cfg, "classify_llm_max_tokens", 8192) or 8192),
            temperature=0.0,
            batch_size=int(getattr(cfg, "classify_llm_columns_per_call", 10) or 10),
            max_retries=int(getattr(cfg, "classify_llm_max_retries", 3) or 3),
        ))

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]
        cap = llm_capability_for_sweep(revisit=bool(revisit_context))
        fn = self._complete_fn or lattice_complete
        result = fn(
            user_prompt,
            capability=cap,
            system_prompt=system_prompt,
            json_schema=CLASSIFICATION_OUTPUT_SCHEMA,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
            forwarder=self._forwarder,
        )
        text = result.text if hasattr(result, "text") else str(result)
        try:
            classifications = _parse_structured_response(text, expected_names)
        except (json.JSONDecodeError, ValueError, TypeError):
            classifications = _parse_classifications(text, expected_names)
        partial = len(classifications) < len(expected_names)
        return LLMResponse(
            classifications=classifications,
            input_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(result, "completion_tokens", 0) or 0),
            model=str(getattr(result, "model", "") or cap),
            finish_reason=str(getattr(result, "finish_reason", "stop") or "stop"),
            reasoning_text=str(getattr(result, "reasoning_content", "") or ""),
            partial=partial,
        )

    def health_check(self) -> bool:
        try:
            from atelier.engine.forwarder import CapabilityForwarder

            fwd = self._forwarder or CapabilityForwarder()
            fwd.resolve("instruct")
            return True
        except Exception:
            return False
