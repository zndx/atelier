"""Capability forwarder — Atelier does not host `instruct`; the federation peer that hosts the model does.

Since 2026-09-08 `instruct` is an OPERATING PROFILE of the resident Qwen3.8-27B (thinking on,
``reasoning_effort=low``; signals-protocol ``capabilities.md`` §Operating profiles) served by the
peer that hosts it (Gaius). Atelier's local vLLM manager keeps only the capabilities it HOSTS
(``referee`` — the architecturally-independent judge family); every other capability is forwarded
VERBATIM over ``zndx.engine.v1`` to the peer whose ``Engine/Status`` lists it healthy. Same module
shape as Ægir's ``aegir/engine/forwarder.py``.

Resolution: peers = the Signals peer-contract lattice + directory seeds, self excluded; RESIDENT
offers (an Endpoint that pins ``gpu_ids``) first — a healthy row with no GPUs is a relay; peer
order breaks ties. Status answers are cached ``ttl_s``. **No fallback**: a capability nobody
advertises is :class:`NoPeerServes` (``FAILED_PRECONDITION`` at the faces); an ``instruct`` request
is never quietly served as ``thinking`` here — the SERVING engine owns the profile.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import grpc

from zndx.engine.v1 import engine_pb2 as zpb
from zndx.engine.v1 import engine_pb2_grpc as zpbg

log = logging.getLogger("atelier.engine.forwarder")

GURU_NOPEER = "#AT.00000001.NOPEER"
DEFAULT_CAPABILITY = "instruct"
ROUTE_CAPABILITIES: tuple[str, ...] = ("instruct", "thinking")


class NoPeerServes(RuntimeError):
    def __init__(self, capability: str, asked: list[str]) -> None:
        self.capability = capability
        self.asked = list(asked)
        who = ", ".join(asked) if asked else "no peers configured"
        super().__init__(
            f"{GURU_NOPEER} atelier does not host {capability!r} and no federation peer advertises it "
            f"healthy (asked: {who}).\n  The peer that hosts the model must list {capability!r} in "
            f"Engine/Status (capabilities.md §Operating profiles); atelier does not fall back.")


@dataclass(frozen=True)
class Route:
    capability: str
    peer: str
    target: str
    model: str
    gpu_ids: tuple[int, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.peer}@{self.target}"


@dataclass
class _Cached:
    at: float
    status: "zpb.StatusResponse | None"


def _default_peers() -> list[tuple[str, str]]:
    from atelier.engine.s2s import PROJECT, configured_peers, directory_seeds

    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for pid, tgt in (*configured_peers(None), *directory_seeds()):
        key = (pid or "").strip().lower() or tgt
        if not tgt or key == PROJECT or key in seen:
            continue
        seen.add(key)
        out.append((pid, tgt))
    return out


class CapabilityForwarder:
    """Test seams: ``peers`` (→ [(project, target)]), ``status_fn`` (target → StatusResponse),
    ``forward_fn`` (target, request, timeout → CompleteResponse)."""

    def __init__(self, *, peers=None, status_fn=None, forward_fn=None, ttl_s: float = 30.0,
                 status_timeout_s: float = 5.0, timeout_s: float = 1200.0, clock=time.monotonic) -> None:
        self._peers = peers or _default_peers
        self._status_fn = status_fn or self._status_over_grpc
        self._forward_fn = forward_fn or self._forward_over_grpc
        self._ttl, self._status_timeout, self._timeout, self._clock = ttl_s, status_timeout_s, timeout_s, clock
        self._cache: dict[str, _Cached] = {}
        self._channels: dict[str, grpc.Channel] = {}
        self._lock = threading.Lock()

    def _channel(self, target: str) -> grpc.Channel:
        with self._lock:
            ch = self._channels.get(target)
            if ch is None:
                ch = grpc.insecure_channel(target)
                self._channels[target] = ch
            return ch

    def _status_over_grpc(self, target: str) -> zpb.StatusResponse:
        return zpbg.EngineStub(self._channel(target)).Status(zpb.StatusRequest(), timeout=self._status_timeout)

    def _forward_over_grpc(self, target: str, request: zpb.CompleteRequest, timeout: float) -> zpb.CompleteResponse:
        return zpbg.EngineStub(self._channel(target)).Complete(request, timeout=timeout)

    def _status(self, target: str) -> "zpb.StatusResponse | None":
        now = self._clock()
        c = self._cache.get(target)
        if c is not None and now - c.at < self._ttl:
            return c.status
        try:
            st = self._status_fn(target)
        except Exception as e:  # noqa: BLE001 — unreachable peer offers nothing for one TTL
            log.info("peer status failed target=%s err=%s", target, type(e).__name__)
            st = None
        self._cache[target] = _Cached(at=now, status=st)
        return st

    def invalidate(self, target: "str | None" = None) -> None:
        if target is None:
            self._cache.clear()
        else:
            self._cache.pop(target, None)

    def resolve(self, capability: str) -> Route:
        cap = (capability or DEFAULT_CAPABILITY).strip()
        asked: list[str] = []
        resident: list[Route] = []
        relayed: list[Route] = []
        for pid, tgt in self._peers():
            st = self._status(tgt)
            asked.append(f"{pid or '?'}@{tgt}")
            if st is None:
                continue
            for ep in st.endpoints:
                if ep.capability != cap or not ep.healthy:
                    continue
                r = Route(capability=cap, peer=st.project or pid, target=tgt, model=ep.model, gpu_ids=tuple(ep.gpu_ids))
                (resident if ep.gpu_ids else relayed).append(r)
        if resident:
            return resident[0]
        if relayed:
            log.info("capability=%s: no resident offer; using non-resident %s", cap, relayed[0].label)
            return relayed[0]
        raise NoPeerServes(cap, asked)

    def routes(self, capabilities: tuple[str, ...] = ROUTE_CAPABILITIES) -> list[Route]:
        out: list[Route] = []
        for cap in capabilities:
            try:
                out.append(self.resolve(cap))
            except NoPeerServes:
                continue
        return out

    def forward(self, request: zpb.CompleteRequest, *, timeout: "float | None" = None) -> zpb.CompleteResponse:
        """Forward a zndx Complete VERBATIM (tools_json / messages_json / capabilities[] included)."""
        if not request.capability and not list(request.capabilities):
            request.capability = DEFAULT_CAPABILITY
        cap = request.capability or DEFAULT_CAPABILITY
        route = self.resolve(cap)
        try:
            resp = self._forward_fn(route.target, request, timeout or self._timeout)
        except grpc.RpcError:
            self.invalidate(route.target)  # stale health → re-resolve next time; the error is the answer
            raise
        if not resp.fulfilled_by:
            resp.fulfilled_by = f"forwarded@{route.label}"
        return resp

    def complete(self, capability: str, prompt: str, system_prompt: str = "", max_tokens: int = 512,
                 temperature: float = 0.7, json_schema: str = "") -> dict:
        """The dict form the native face speaks (same keys as VllmManager.complete)."""
        req = zpb.CompleteRequest(capability=capability or DEFAULT_CAPABILITY, prompt=prompt,
                                  system_prompt=system_prompt or "", max_tokens=int(max_tokens or 0),
                                  temperature=float(temperature or 0.0), json_schema=json_schema or "")
        t0 = time.time()
        r = self.forward(req)
        return {"text": r.text, "model": r.model, "reasoning_content": r.reasoning_content,
                "finish_reason": r.finish_reason, "prompt_tokens": r.prompt_tokens,
                "completion_tokens": r.completion_tokens,
                "latency_ms": r.latency_ms or (time.time() - t0) * 1000.0,
                "fulfilled_by": r.fulfilled_by}

    def shutdown(self) -> None:
        with self._lock:
            for ch in self._channels.values():
                try:
                    ch.close()
                except Exception:  # noqa: BLE001
                    pass
            self._channels.clear()
