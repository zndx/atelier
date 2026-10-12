"""Sweep backend: one entity, a genus-first Engine/Decide walk.

This is not Engine/Complete. A revisit context is refused so a thinking
pass cannot be served as a decision read.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from atelier.classify.decisions_walk import (
    DEFAULT_DESCEND_FLOOR,
    MAX_CALLS,
    WalkError,
    walk_entity,
)
from atelier.classify.llm_backend import (
    ColumnClassification,
    LLMBackend,
    LLMBackendConfig,
    LLMResponse,
)
from atelier.flows.lattice import GURU_NOCAP

logger = logging.getLogger(__name__)

DecideFn = Callable[..., Any]


def choice_distribution(answers: Any, *, where: str = "") -> dict[str, float]:
    """Option → probability for one Choice answer.

    A live Engine/Decide choice is ``{type, choice, probabilities,
    confidence}``. ``probabilities`` is the distribution the walk
    ranks. The ``confidence`` number on that object is a separate
    score and is not an option probability.

    A flat ``{option: probability}`` map is accepted as well.
    """
    suffix = f" {where}" if where else ""
    if not isinstance(answers, dict):
        raise RuntimeError(
            f"{GURU_NOCAP} Decide returned no choice distribution{suffix}"
        )
    choice = answers.get("choice")
    if not isinstance(choice, dict) or not choice:
        raise RuntimeError(
            f"{GURU_NOCAP} Decide returned no choice distribution{suffix}"
        )
    raw = choice.get("probabilities")
    if isinstance(raw, dict):
        source: Any = raw
    elif "type" in choice or isinstance(choice.get("choice"), str):
        # Structured choice without a probability map. Do not scrape
        # the confidence field into an option named "confidence".
        source = {}
    else:
        source = choice
    dist: dict[str, float] = {}
    for key, value in source.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        dist[str(key)] = float(value)
    if not dist:
        raise RuntimeError(
            f"{GURU_NOCAP} Decide returned no choice distribution{suffix}"
        )
    return dist


def entity_state(sample: Any, table_name: str | None = None) -> str:
    """Column text is the decision state. Other DST scores stay out of it."""
    name = getattr(sample, "qualified_name", None) or getattr(sample, "name", "")
    if table_name and name and "." not in str(name):
        name = f"{table_name}.{name}"
    col_type = getattr(sample, "column_type", None) or ""
    values = [str(v) for v in (getattr(sample, "values", None) or [])[:8]]
    shown = ", ".join(values)
    return f"{name} type={col_type} values=[{shown}]"


class DecisionsBackend(LLMBackend):
    """Classify entities by capability ``decisions``."""

    def __init__(
        self,
        config: LLMBackendConfig | None = None,
        *,
        decide_fn: DecideFn | None = None,
        forwarder: Any | None = None,
        category_set: Any | None = None,
        descend_floor: float = DEFAULT_DESCEND_FLOOR,
        max_calls: int = MAX_CALLS,
    ) -> None:
        super().__init__(config or LLMBackendConfig(backend="decisions", model="vllm-sr/d3-flash"))
        self._decide_fn = decide_fn
        self._forwarder = forwarder
        self._category_set = category_set
        self._floor = float(descend_floor)
        self._max_calls = int(max_calls)
        self.truncated_entities: list[str] = []

    @classmethod
    def from_cfg(cls, cfg: Any) -> DecisionsBackend:
        return cls(
            LLMBackendConfig(
                backend="decisions",
                model="vllm-sr/d3-flash",
                temperature=0.0,
                batch_size=1,
            ),
            descend_floor=float(getattr(cfg, "classify_decisions_descend_floor", DEFAULT_DESCEND_FLOOR) or DEFAULT_DESCEND_FLOOR),
            max_calls=int(getattr(cfg, "classify_decisions_max_calls", MAX_CALLS) or MAX_CALLS),
        )

    def bind_taxonomy(self, category_set: Any) -> None:
        self._category_set = category_set

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        if revisit_context:
            raise RuntimeError(
                f"{GURU_NOCAP} decisions sweep does not run the thinking revisit"
            )
        if self._category_set is None:
            raise RuntimeError(f"{GURU_NOCAP} decisions sweep has no taxonomy bound")
        classifications: list[ColumnClassification] = []
        in_tok = 0
        out_tok = 0
        model = "vllm-sr/d3-flash"
        for sample in samples:
            state = entity_state(sample, table_name)
            name = getattr(sample, "name", "") or state

            def _ask(criteria: dict, instructions: str, *, _state: str = state) -> dict[str, float]:
                nonlocal in_tok, out_tok, model
                result = self._call(_state, criteria, instructions)
                model = str(getattr(result, "model", "") or model)
                in_tok += int(getattr(result, "input_tokens", 0) or 0)
                out_tok += int(getattr(result, "output_tokens", 0) or 0)
                answers = getattr(result, "answers", None) or {}
                return choice_distribution(
                    answers,
                    where=f"for {name!r} model={model!r}",
                )

            try:
                walked = walk_entity(
                    self._category_set,
                    _ask,
                    floor=self._floor,
                    max_calls=self._max_calls,
                )
            except WalkError as exc:
                raise RuntimeError(f"{GURU_NOCAP} {exc}") from exc
            if walked.truncated:
                self.truncated_entities.append(str(name))
                logger.info(
                    "decisions walk truncated at %d calls for %s; "
                    "classified %s (discount 0.15 is uncalibrated against this softmax)",
                    self._max_calls, name, walked.category_code,
                )
            classifications.append(ColumnClassification(
                column_name=str(name),
                category_code=walked.category_code,
                confidence=walked.confidence,
                evidence="",
                alternatives=walked.alternatives,
            ))
        return LLMResponse(
            classifications=classifications,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model,
            finish_reason="stop",
            partial=len(classifications) < len(samples),
        )

    def health_check(self) -> bool:
        try:
            from atelier.engine.forwarder import CapabilityForwarder

            fwd = self._forwarder or CapabilityForwarder()
            fwd.resolve("decisions")
            return True
        except Exception:
            return False

    def _call(self, state: str, criteria: dict, instructions: str) -> Any:
        fn = self._decide_fn
        if fn is None:
            from atelier.flows.lattice import decide as lattice_decide

            fn = lattice_decide
        return fn(
            state,
            {"choice": {
                "type": "choice",
                "instructions": instructions,
                "criteria": criteria,
            }},
            forwarder=self._forwarder,
        )
