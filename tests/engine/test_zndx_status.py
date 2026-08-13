"""Tests for the zndx.engine.v1 lattice face: Status body + server reflection.

Lattice accept is Engine/Status (project=atelier, capability=referee) at gRPC
bind — no vLLM, no live :50251. Reflection is the external grpcurl surface.
"""
from __future__ import annotations

from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

import grpc

from atelier.engine.config import ModelSpec
from atelier.engine.server import (
    PROJECT,
    ZndxEngineServicer,
    enable_reflection,
)
from zndx.engine.v1 import engine_pb2 as zpb
from zndx.engine.v1 import engine_pb2_grpc as zpbg

_REPO = Path(__file__).resolve().parents[2]


def _native(capabilities=None, live=()):
    if capabilities is None:
        capabilities = {
            "instruct": ModelSpec(model="org/instruct"),
            "referee": ModelSpec(model="org/referee"),
        }
    return SimpleNamespace(
        cfg=SimpleNamespace(capabilities=capabilities),
        mgr=SimpleNamespace(status=lambda: list(live)),
    )


def _live(capability, model="org/live", healthy=True, gpu_ids=()):
    return SimpleNamespace(
        capability=capability,
        spec=SimpleNamespace(model=model),
        healthy=healthy,
        gpu_ids=list(gpu_ids),
    )


class TestZndxServicerStatus:
    def test_always_advertises_atelier_and_referee(self):
        resp = ZndxEngineServicer(_native(capabilities={})).Status(
            zpb.StatusRequest(), None
        )
        assert resp.project == PROJECT
        assert any(ep.capability == "referee" for ep in resp.endpoints)

    def test_configured_caps_are_placeholders_until_live(self):
        resp = ZndxEngineServicer(_native()).Status(zpb.StatusRequest(), None)
        caps = {ep.capability: ep for ep in resp.endpoints}
        assert set(caps) >= {"instruct", "referee"}
        assert caps["instruct"].model == "org/instruct"
        assert caps["instruct"].healthy is False
        assert "configured" in caps["referee"].detail

    def test_overlays_live_referee_endpoint(self):
        resp = ZndxEngineServicer(
            _native(live=[_live("referee", model="org/live", healthy=False, gpu_ids=[0, 1])])
        ).Status(zpb.StatusRequest(), None)
        referee = next(ep for ep in resp.endpoints if ep.capability == "referee")
        assert referee.model == "org/live"
        assert referee.healthy is False
        assert list(referee.gpu_ids) == [0, 1]
        assert sum(1 for ep in resp.endpoints if ep.capability == "referee") == 1


class TestReflection:
    def test_lists_zndx_engine(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
        zpbg.add_EngineServicer_to_server(ZndxEngineServicer(_native()), server)
        enable_reflection(server)
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        try:
            from grpc_reflection.v1alpha import reflection_pb2, reflection_pb2_grpc

            ch = grpc.insecure_channel(f"127.0.0.1:{port}")
            stub = reflection_pb2_grpc.ServerReflectionStub(ch)
            req = reflection_pb2.ServerReflectionRequest(list_services="")
            replies = list(stub.ServerReflectionInfo(iter([req])))
            names = {s.name for r in replies for s in r.list_services_response.service}
            assert "zndx.engine.v1.Engine" in names
            assert "atelier.engine.AtelierEngine" in names
            assert "grpc.reflection.v1alpha.ServerReflection" in names
            r = zpbg.EngineStub(ch).Status(zpb.StatusRequest(), timeout=3)
            assert r.project == "atelier"
            assert any(ep.capability == "referee" for ep in r.endpoints)
            ch.close()
        finally:
            server.stop(grace=0)


class TestUnitWrappers:
    def test_start_is_engine_only(self):
        text = (_REPO / "scripts" / "systemd_start.sh").read_text()
        assert "python -m atelier.engine.server" in text
        assert "engine-supervise" not in text
        assert "50251" in text
        assert "50071" in text  # documented as off the unit path
        assert not any(
            ln.lstrip().startswith("just up")
            or "&& just up" in ln
            or "|| just up" in ln
            for ln in text.splitlines()
        )

    def test_soft_stop_does_not_touch_sibling_leases(self):
        text = (_REPO / "scripts" / "systemd_stop.sh").read_text()
        assert "atelier.engine" in text
        assert "teardown" not in text.lower() or "Does NOT" in text
        assert "gpu-deep-cleanup" not in text
        assert "gaius" not in text.lower() or "Does NOT touch" in text
        assert "aegir" not in text.lower() or "Does NOT touch" in text
        # Lease dir may be named in a comment; never mutated.
        for token in ("rm ", "unlink", "rmdir", ">", "mkdir"):
            if "/tmp/zndx-gpu-leases" in text:
                lease_lines = [
                    ln for ln in text.splitlines() if "zndx-gpu-leases" in ln
                ]
                assert lease_lines
                assert all(
                    ln.lstrip().startswith("#") for ln in lease_lines
                ), lease_lines
        assert "just down" not in text
        assert "just up" not in text
