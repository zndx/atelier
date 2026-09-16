"""Engine/Complete is the classification LLM; referee is not."""

from __future__ import annotations

import json

import pytest

from atelier.classify.complete_backend import CompleteLLMBackend
from atelier.classify.llm_backend import create_backend_from_cfg
from atelier.classify.sampler import ColumnSample
from atelier.config import AtelierConfig
from atelier.flows.lattice import (
    GURU_NOCAP,
    GURU_UNHEALTHY,
    LatticeComplete,
    assert_complete_capability,
    complete,
)
from atelier.engine.forwarder import NoPeerServes, Route


class _Fwd:
    def __init__(self, *, healthy=True, model="Qwen"):
        self.healthy = healthy
        self.model = model
        self.forwarded = []

    def resolve(self, cap: str) -> Route:
        if not self.healthy:
            raise NoPeerServes(cap, ["gaius@example"])
        return Route(capability=cap, peer="gaius", target="gaius.example:1", model=self.model)

    def forward(self, request, *, timeout=None):
        self.forwarded.append(request)
        from types import SimpleNamespace

        payload = {
            "classifications": [
                {
                    "column_name": "id",
                    "category_code": "1.1",
                    "confidence": 0.9,
                    "evidence": "identifier",
                    "alternatives": [],
                }
            ]
        }
        return SimpleNamespace(
            text=json.dumps(payload),
            model=self.model,
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=1.0,
            reasoning_content="",
            finish_reason="stop",
            fulfilled_by="gaius",
        )


def test_default_backend_is_engine_complete() -> None:
    cfg = AtelierConfig()
    assert cfg.classify_llm_backend == "engine_complete"
    assert cfg.has_classify_llm
    backend = create_backend_from_cfg(cfg)
    assert isinstance(backend, CompleteLLMBackend)


def test_classify_batch_uses_instruct_then_thinking_on_revisit() -> None:
    calls: list[str] = []

    def fake_complete(prompt, **kw):
        calls.append(kw["capability"])
        return LatticeComplete(
            text=json.dumps({
                "classifications": [{
                    "column_name": "id",
                    "category_code": "1.1",
                    "confidence": 0.8,
                    "evidence": "x",
                    "alternatives": [],
                }]
            }),
            model="Qwen",
            prompt_tokens=1,
            completion_tokens=2,
            latency_ms=1,
            reasoning_content="",
            finish_reason="stop",
        )

    be = CompleteLLMBackend(complete_fn=fake_complete)
    sample = ColumnSample(name="id", column_type="int", values=["1"])
    first = be.classify_batch([sample], "sys")
    assert first.classifications[0].category_code == "1.1"
    assert calls == ["instruct"]
    be.classify_batch([sample], "sys", revisit_context={"id": {"belief": 0.1}})
    assert calls == ["instruct", "thinking"]


def test_complete_discovers_peer_and_refuses_unhealthy() -> None:
    ok = complete("hi", capability="instruct", forwarder=_Fwd())
    assert ok.model == "Qwen"
    with pytest.raises(RuntimeError, match=GURU_UNHEALTHY):
        complete("hi", capability="instruct", forwarder=_Fwd(healthy=False))
    with pytest.raises(RuntimeError, match=GURU_NOCAP):
        assert_complete_capability("referee")
