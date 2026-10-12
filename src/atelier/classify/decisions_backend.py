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

# One in-flight System One walk per d3-flash replica. Metaflow fans the
# same width out as foreach shards; the in-process pipeline uses it as
# a thread pool. Two shards must not each also run parallel=2.
REPLICA_COUNT = 2


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
        parallel: int = 1,
    ) -> None:
        super().__init__(config or LLMBackendConfig(backend="decisions", model="vllm-sr/d3-flash"))
        self._decide_fn = decide_fn
        self._forwarder = forwarder
        self._category_set = category_set
        self._floor = float(descend_floor)
        self._max_calls = int(max_calls)
        self._parallel = max(1, int(parallel))
        self.truncated_entities: list[str] = []

    @classmethod
    def from_cfg(cls, cfg: Any, *, parallel: int = 1) -> DecisionsBackend:
        return cls(
            LLMBackendConfig(
                backend="decisions",
                model="vllm-sr/d3-flash",
                temperature=0.0,
                batch_size=1,
            ),
            descend_floor=float(getattr(cfg, "classify_decisions_descend_floor", DEFAULT_DESCEND_FLOOR) or DEFAULT_DESCEND_FLOOR),
            max_calls=int(getattr(cfg, "classify_decisions_max_calls", MAX_CALLS) or MAX_CALLS),
            parallel=parallel,
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
        workers = min(self._parallel, len(samples)) if samples else 1
        if workers == 1:
            walked = [self._walk_sample(sample, table_name) for sample in samples]
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=workers) as pool:
                walked = list(pool.map(
                    lambda sample: self._walk_sample(sample, table_name),
                    samples,
                ))
        classifications = [item[0] for item in walked]
        in_tok = sum(item[1] for item in walked)
        out_tok = sum(item[2] for item in walked)
        model = next((item[3] for item in reversed(walked) if item[3]), "vllm-sr/d3-flash")
        for name in (item[4] for item in walked):
            if name:
                self.truncated_entities.append(name)
        return LLMResponse(
            classifications=classifications,
            input_tokens=in_tok,
            output_tokens=out_tok,
            model=model,
            finish_reason="stop",
            partial=len(classifications) < len(samples),
        )

    def _walk_sample(self, sample: Any, table_name: str | None) -> tuple:
        """One entity. The walk itself stays serial; callers overlap entities."""
        state = entity_state(sample, table_name)
        name = getattr(sample, "name", "") or state
        in_tok = 0
        out_tok = 0
        model = "vllm-sr/d3-flash"

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
            logger.info(
                "decisions walk truncated at %d calls for %s; "
                "classified %s (discount 0.15 is uncalibrated against this softmax)",
                self._max_calls, name, walked.category_code,
            )
        classification = ColumnClassification(
            column_name=str(name),
            category_code=walked.category_code,
            confidence=walked.confidence,
            evidence="",
            alternatives=walked.alternatives,
        )
        truncated = str(name) if walked.truncated else ""
        return classification, in_tok, out_tok, model, truncated

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


def select_shard_columns(samples, shard: int, width: int = REPLICA_COUNT) -> list[tuple[Any, Any]]:
    """Columns whose stable index belongs to this replica.

    Index 0 is the first column of the first table. Shard ``i`` keeps
    ``index % width == i``, so five entities and width 2 yield three
    and two. The split is the Metaflow foreach, not a hidden loop.
    """
    width = int(width)
    shard = int(shard)
    if width < 1:
        raise ValueError("decisions shard width must be at least 1")
    if shard < 0 or shard >= width:
        raise ValueError(f"decisions shard {shard} is outside width {width}")
    chosen: list[tuple[Any, Any]] = []
    index = 0
    for table in samples:
        for column in getattr(table, "columns", []) or []:
            if index % width == shard:
                chosen.append((table, column))
            index += 1
    return chosen


class ReplayDecisionsBackend(LLMBackend):
    """Serve labels a decisions shard already stored. Does not call Decide."""

    def __init__(self, labels: list[dict]) -> None:
        super().__init__(LLMBackendConfig(backend="decisions", model="vllm-sr/d3-flash"))
        self._by_key: dict[str, dict] = {}
        for row in labels:
            qualified = str(row.get("qualified_name") or "")
            bare = str(row.get("column_name") or "")
            if qualified:
                self._by_key[qualified] = row
            if bare and bare not in self._by_key:
                self._by_key[bare] = row

    def bind_taxonomy(self, category_set: Any) -> None:
        return None

    def health_check(self) -> bool:
        return bool(self._by_key)

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        if revisit_context:
            raise RuntimeError(
                f"{GURU_NOCAP} decisions replay does not run the thinking revisit"
            )
        classifications: list[ColumnClassification] = []
        for sample in samples:
            bare = str(getattr(sample, "name", "") or "")
            qualified = str(getattr(sample, "qualified_name", "") or "")
            if table_name and bare:
                qualified = f"{table_name}.{bare}"
            row = self._by_key.get(qualified) or self._by_key.get(bare)
            if row is None:
                raise RuntimeError(
                    f"{GURU_NOCAP} replay has no decisions label for "
                    f"{qualified or bare!r}"
                )
            code = row.get("predicted_code") or row.get("category_code") or ""
            classifications.append(ColumnClassification(
                column_name=bare or str(row.get("column_name") or ""),
                category_code=str(code) or None,
                confidence=float(row.get("confidence") or 0.0),
                evidence="",
                alternatives=list(row.get("alternatives") or []),
            ))
        return LLMResponse(
            classifications=classifications,
            input_tokens=0,
            output_tokens=0,
            model="vllm-sr/d3-flash",
            finish_reason="stop",
            partial=False,
        )
