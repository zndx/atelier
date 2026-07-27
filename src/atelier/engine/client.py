"""Thin capability-client — the ONLY way a workload reaches the engine:
``complete(prompt, capability=…)`` → text, over gRPC. No vLLM endpoint is ever
returned to the caller (strict layering).

Federation: setting ``engine.federate`` (HOCON; env ``ATELIER_ENGINE_FEDERATE``
via substitution) delegates capability requests to a sibling engine — Ægir's
(:50151) or Gaius's (:50051). The wire contract is the shared minimal-engine
proto; our ``json_schema`` extension is ignored by peers that predate it, so
callers using schemas must tolerate unconstrained text from federated targets.
"""
from __future__ import annotations

import grpc

from atelier.engine.config import load_engine_config
from atelier.engine.proto import atelier_engine_pb2 as pb
from atelier.engine.proto import atelier_engine_pb2_grpc as pbg


def _route() -> tuple[str, bool]:
    """(target, federated). Federated targets are foreign engines — reach them
    through the SHARED ``zndx.engine.v1.Engine`` service (per-project native
    services have distinct gRPC method paths; a foreign native stub gets
    UNIMPLEMENTED — the 2026-07-03 federation finding)."""
    cfg = load_engine_config()
    if cfg.federate:
        return cfg.federate, True
    return f"127.0.0.1:{cfg.grpc_port}", False


def complete_detailed(prompt: str, *, capability: str = "instruct",
                      system_prompt: str = "", max_tokens: int = 4096,
                      temperature: float = 0.7, json_schema: str = "",
                      timeout: float = 1200.0) -> dict:
    """Like :func:`complete`, but returns the full response — including the
    retained ``reasoning_content`` (thinking trace → referee audit trail) and
    ``finish_reason`` ('length' ⇒ raise max_tokens). ``json_schema`` requests
    schema-constrained decoding (enforced by any zndx.engine.v1 peer)."""
    target, federated = _route()
    with grpc.insecure_channel(target) as ch:
        if federated:
            from zndx.engine.v1 import engine_pb2 as zpb
            from zndx.engine.v1 import engine_pb2_grpc as zpbg
            r = zpbg.EngineStub(ch).Complete(zpb.CompleteRequest(
                capability=capability, prompt=prompt, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature,
                json_schema=json_schema), timeout=timeout)
        else:
            r = pbg.AtelierEngineStub(ch).Complete(pb.CompleteRequest(
                capability=capability, prompt=prompt, system_prompt=system_prompt,
                max_tokens=max_tokens, temperature=temperature,
                json_schema=json_schema), timeout=timeout)
        return {"text": r.text, "reasoning_content": r.reasoning_content,
                "finish_reason": r.finish_reason, "model": r.model,
                "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_ms": r.latency_ms}


def complete(prompt: str, *, capability: str = "instruct", system_prompt: str = "",
             max_tokens: int = 4096, temperature: float = 0.7,
             timeout: float = 1200.0) -> str:
    """Request the capability to complete ``prompt``; returns the answer text.
    (Long default timeout: the first call may trigger an on-demand vLLM load.)"""
    return complete_detailed(prompt, capability=capability,
                             system_prompt=system_prompt, max_tokens=max_tokens,
                             temperature=temperature, timeout=timeout)["text"]


def engine_status(timeout: float = 10.0):
    target, federated = _route()
    with grpc.insecure_channel(target) as ch:
        if federated:
            from zndx.engine.v1 import engine_pb2 as zpb
            from zndx.engine.v1 import engine_pb2_grpc as zpbg
            return zpbg.EngineStub(ch).Status(zpb.StatusRequest(), timeout=timeout)
        return pbg.AtelierEngineStub(ch).EngineStatus(
            pb.EngineStatusRequest(), timeout=timeout)
