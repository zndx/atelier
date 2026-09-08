"""CapabilityForwarder — Atelier forwards what it does not host; resolve by peer Status, forward verbatim, no fallback."""
from __future__ import annotations

import grpc
import pytest

from atelier.engine.forwarder import GURU_NOPEER, CapabilityForwarder, NoPeerServes, Route
from zndx.engine.v1 import engine_pb2 as zpb


def _status(project, *eps):
    st = zpb.StatusResponse(project=project)
    for cap, healthy, model in eps:
        st.endpoints.add(capability=cap, healthy=healthy, model=model, gpu_ids=[0, 1, 2, 3])
    return st


class _Rpc(grpc.RpcError):
    def __init__(self, code, details=""):
        self._code, self._details = code, details

    def code(self):
        return self._code

    def details(self):
        return self._details


def _fwd(statuses: dict, peers=None, forward=None, **kw):
    peers = peers or [(p, f"{p}:5") for p in statuses]
    return CapabilityForwarder(
        peers=lambda: list(peers),
        status_fn=lambda tgt: statuses[tgt.split(":")[0]](),
        forward_fn=forward or (lambda tgt, req, timeout: zpb.CompleteResponse(
            text=f"ok from {tgt}", model="Qwen/Qwen3.8-27B", finish_reason="stop",
            reasoning_content="brief", prompt_tokens=3, completion_tokens=2,
            profile=zpb.OperatingProfile(capability=req.capability, thinking=True, reasoning_effort="low"))),
        **kw,
    )


def test_resolves_to_the_peer_whose_status_serves_the_capability_healthy():
    f = _fwd({
        "hermes": lambda: _status("hermes"),  # no inference endpoints
        "gaius": lambda: _status("gaius", ("thinking", True, "Qwen/Qwen3.8-27B"),
                                 ("instruct", True, "Qwen/Qwen3.8-27B")),
    })
    r = f.resolve("instruct")
    assert isinstance(r, Route) and r.peer == "gaius" and r.model == "Qwen/Qwen3.8-27B"
    assert r.target == "gaius:5" and r.gpu_ids == (0, 1, 2, 3)


def test_no_fallback_when_nobody_serves_it():
    f = _fwd({"gaius": lambda: _status("gaius", ("thinking", True, "Qwen/Qwen3.8-27B"))})
    with pytest.raises(NoPeerServes) as ei:
        f.resolve("instruct")  # thinking is served — instruct is NOT quietly mapped onto it
    assert GURU_NOPEER in str(ei.value) and "gaius@gaius:5" in str(ei.value)
    assert ei.value.capability == "instruct"


def test_unhealthy_offer_does_not_count():
    f = _fwd({"gaius": lambda: _status("gaius", ("instruct", False, "Qwen/Qwen3.8-27B"))})
    with pytest.raises(NoPeerServes):
        f.resolve("instruct")


def test_unreachable_peer_is_skipped_not_fatal():
    def boom():
        raise _Rpc(grpc.StatusCode.UNAVAILABLE, "down")
    f = _fwd({"atelier": boom, "gaius": lambda: _status("gaius", ("instruct", True, "m"))})
    assert f.resolve("instruct").peer == "gaius"


def test_status_is_cached_for_ttl_then_refreshed():
    calls = {"n": 0}
    now = {"t": 100.0}

    def st():
        calls["n"] += 1
        return _status("gaius", ("instruct", True, "m"))
    f = _fwd({"gaius": st}, ttl_s=30.0, clock=lambda: now["t"])
    f.resolve("instruct"); f.resolve("instruct")
    assert calls["n"] == 1
    now["t"] += 31
    f.resolve("instruct")
    assert calls["n"] == 2


def test_forward_passes_the_request_verbatim_and_defaults_capability():
    seen = {}

    def fwd(tgt, req, timeout):
        seen["tgt"], seen["req"], seen["timeout"] = tgt, req, timeout
        return zpb.CompleteResponse(text="hi", model="m", fulfilled_by="")
    f = _fwd({"gaius": lambda: _status("gaius", ("instruct", True, "m"))}, forward=fwd, timeout_s=77.0)
    req = zpb.CompleteRequest(prompt="p", tools_json='[{"type":"function"}]', messages_json='[{"role":"user"}]')
    resp = f.forward(req)
    assert seen["req"] is req and seen["req"].capability == "instruct"  # empty → the default capability
    assert seen["req"].tools_json and seen["req"].messages_json  # nothing flattened or dropped
    assert seen["tgt"] == "gaius:5" and seen["timeout"] == 77.0
    assert resp.fulfilled_by == "forwarded@gaius@gaius:5"


def test_complete_dict_carries_profile_and_fulfilled_by():
    f = _fwd({"gaius": lambda: _status("gaius", ("instruct", True, "Qwen/Qwen3.8-27B"))})
    out = f.complete("instruct", "hello", max_tokens=8, temperature=0.1)
    assert out["text"].startswith("ok from") and out["model"] == "Qwen/Qwen3.8-27B"
    assert out["reasoning_content"] == "brief"  # effort=low still yields a model layer
    assert out["fulfilled_by"] == "forwarded@gaius@gaius:5"
    assert out["prompt_tokens"] == 3 and out["completion_tokens"] == 2


def test_peer_rpc_error_propagates_and_invalidates_its_status():
    n = {"status": 0}

    def st():
        n["status"] += 1
        return _status("gaius", ("instruct", True, "m"))

    def fwd(tgt, req, timeout):
        raise _Rpc(grpc.StatusCode.RESOURCE_EXHAUSTED, "busy")
    f = _fwd({"gaius": st}, forward=fwd)
    with pytest.raises(grpc.RpcError):
        f.forward(zpb.CompleteRequest(capability="instruct", prompt="p"))
    f.resolve("instruct")
    assert n["status"] == 2  # re-resolved after the failure, not served from stale cache


def test_routes_skip_unserved():
    f = _fwd({"gaius": lambda: _status("gaius", ("instruct", True, "m"))})
    assert [r.capability for r in f.routes(("instruct", "thinking", "vision"))] == ["instruct"]


def test_resident_offer_preferred_over_a_gpu_less_relay():
    """A peer that lists `instruct` healthy but pins no GPUs is a relay/over-claim; the engine that
    names its GPUs is the host. Peer order alone would have picked the relay."""
    relay = zpb.StatusResponse(project="metabase")
    relay.endpoints.add(capability="instruct", healthy=True, model="Qwen/Qwen3.8-27B")  # no gpu_ids
    f = _fwd({
        "metabase": lambda: relay,
        "gaius": lambda: _status("gaius", ("instruct", True, "Qwen/Qwen3.8-27B")),
    })
    assert f.resolve("instruct").peer == "gaius"
    # ...but a relay is still an honest advertised offer when nobody resident serves it
    g = _fwd({"metabase": lambda: relay, "gaius": lambda: _status("gaius", ("thinking", True, "m"))})
    assert g.resolve("instruct").peer == "metabase"
