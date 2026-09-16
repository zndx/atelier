"""Signals lattice helpers for Metaflow steps.

LLM is Engine/Complete thinking|instruct (OIP). No Anthropic, no Bedrock,
no referee. The peer is discovered from Status (CapabilityForwarder).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from atelier.engine.forwarder import NoPeerServes
from atelier.flows.platform_metaflow import (
    GURU_NOPLATFORM,
    PlatformMetaflowError,
    require_signals_metaflow as _require,
)

GURU_NOLATTICE = "#GR.00000002.NOLATTICE"
GURU_NOCAP = "#GR.00000003.NOCAP"
GURU_UNHEALTHY = "#EP.00000016.NOTREADY"
GURU_WRONGMODEL = "#EP.00000018.WRONGMODEL"
GURU_TRUNCATED = "#EP.00000017.TRUNCATED"

COMPLETE_CAPABILITIES = ("thinking", "instruct")


def require_signals_metaflow(environ: Mapping[str, str] | None = None) -> None:
    _require(environ)


def assert_complete_capability(capability: str) -> str:
    cap = (capability or "").strip().lower()
    if cap not in COMPLETE_CAPABILITIES:
        raise RuntimeError(
            f"{GURU_NOCAP} Complete capability must be thinking or instruct "
            f"(got {capability!r}); referee is not a lattice LLM."
        )
    return cap


@dataclass(frozen=True)
class LatticeComplete:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    reasoning_content: str
    finish_reason: str
    fulfilled_by: str = ""


def complete(
    prompt: str,
    *,
    capability: str = "instruct",
    system_prompt: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    json_schema: dict[str, Any] | str | None = None,
    timeout_s: float | None = None,
    forwarder: Any | None = None,
) -> LatticeComplete:
    """Blocking Complete for Metaflow steps. Discovers the serving peer."""
    from zndx.engine.v1 import engine_pb2 as zpb

    cap = assert_complete_capability(capability)
    schema = ""
    if json_schema is not None:
        schema = json_schema if isinstance(json_schema, str) else json.dumps(json_schema)
    if timeout_s is None:
        timeout_s = float(max(420, int(max_tokens) // 8 + 180))

    fwd = forwarder
    if fwd is None:
        from atelier.engine.forwarder import CapabilityForwarder

        fwd = CapabilityForwarder()
    try:
        route = fwd.resolve(cap)
    except NoPeerServes as e:
        raise RuntimeError(
            f"{GURU_UNHEALTHY} no peer advertises healthy capability={cap!r}.\n"
            f"  {e}\n"
            "  Complete does not fall back to hosted APIs."
        ) from e

    req = zpb.CompleteRequest(
        capability=cap,
        prompt=prompt,
        system_prompt=system_prompt or "",
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        json_schema=schema,
    )
    try:
        resp = fwd.forward(req, timeout=timeout_s)
    except Exception as e:
        raise RuntimeError(
            f"{GURU_NOLATTICE} Engine/Complete failed capability={cap!r} "
            f"via {route.label}: {e}\n"
            "  Discover the peer whose Status serves thinking/instruct."
        ) from e

    served = (resp.model or "").strip()
    if not served:
        raise RuntimeError(
            f"{GURU_NOCAP} Complete returned no model for capability={cap!r}."
        )
    expected = (route.model or "").strip()
    if expected and served != expected and expected not in served:
        raise RuntimeError(
            f"{GURU_WRONGMODEL} Complete served {served!r} for capability={cap!r}; "
            f"Status advertised {expected!r}."
        )
    finish = (resp.finish_reason or "").strip().lower()
    if finish == "length":
        raise RuntimeError(
            f"{GURU_TRUNCATED} Complete truncated capability={cap!r} model={served!r}."
        )
    text = resp.text or ""
    if json_schema is not None and not text.strip():
        raise RuntimeError(
            f"{GURU_NOCAP} Complete returned empty text for JSON capability={cap!r}."
        )
    return LatticeComplete(
        text=text,
        model=served,
        prompt_tokens=int(resp.prompt_tokens or 0),
        completion_tokens=int(resp.completion_tokens or 0),
        latency_ms=float(resp.latency_ms or 0.0),
        reasoning_content=resp.reasoning_content or "",
        finish_reason=resp.finish_reason or "",
        fulfilled_by=getattr(resp, "fulfilled_by", "") or route.label,
    )


__all__ = (
    "COMPLETE_CAPABILITIES",
    "GURU_NOLATTICE",
    "GURU_NOCAP",
    "GURU_NOPLATFORM",
    "GURU_TRUNCATED",
    "GURU_UNHEALTHY",
    "GURU_WRONGMODEL",
    "LatticeComplete",
    "PlatformMetaflowError",
    "assert_complete_capability",
    "complete",
    "require_signals_metaflow",
)
