"""Engine unit tests — hermetic (no GPU, no vLLM, no gRPC server).

Covers the HOCON capability registry, the proof-of-progress log scanner, the
thinking-trace split, and the GPU-lease guard's conflict paths.
"""
from __future__ import annotations

import json

import pytest

from atelier.engine.config import EngineConfig, ModelSpec, load_engine_config
from atelier.engine.gpu_guard import GpuClaimError, _foreign_holders, claim_gpus
from atelier.engine.vllm_manager import VllmManager, split_thinking

import atelier.engine.gpu_guard as gpu_guard


# ── Config ───────────────────────────────────────────────────────────

_CONF = """
engine {
  grpc_port = 55555
  vllm_base_port = 9300
  hf_hub_cache = "/tmp/hf"
  log_dir = "/tmp/eng-test"
  capabilities {
    instruct { model = "org/model-fp8", tp = 2, gpu_mem = 0.85,
               max_model_len = 4096, max_num_seqs = 2, tool_parser = "hermes" }
    referee  { model = "org/referee-bf16", tp = 4 }
  }
}
"""


def test_engine_config_parses_capabilities(tmp_path):
    conf = tmp_path / "base.conf"
    conf.write_text(_CONF)
    cfg = load_engine_config(conf)

    assert cfg.grpc_port == 55555
    assert cfg.vllm_base_port == 9300
    inst = cfg.capabilities["instruct"]
    assert inst.model == "org/model-fp8"
    assert inst.tensor_parallel_size == 2
    assert inst.gpu_memory_utilization == 0.85
    assert inst.extra_args == [
        "--max-model-len", "4096", "--max-num-seqs", "2",
        "--enable-auto-tool-choice", "--tool-call-parser", "hermes",
    ]
    # referee: no tool parser (guided-json / soft-parse path, not tool calls)
    ref = cfg.capabilities["referee"]
    assert ref.tensor_parallel_size == 4
    assert "--enable-auto-tool-choice" not in ref.extra_args


def test_engine_port_env_overrides_hocon(tmp_path, monkeypatch):
    conf = tmp_path / "base.conf"
    conf.write_text(_CONF)
    monkeypatch.setenv("ATELIER_ENGINE_PORT", "50251")
    cfg = load_engine_config(conf)
    assert cfg.grpc_port == 50251


def test_engine_config_defaults_without_block(tmp_path):
    conf = tmp_path / "empty.conf"
    conf.write_text("app { name = x }")
    cfg = load_engine_config(conf)
    assert cfg.grpc_port == 50251
    assert cfg.vllm_base_port == 8200
    assert cfg.capabilities == {}


def test_repo_base_conf_defines_both_capabilities():
    """The shipped config carries the two-stack referee-independence boundary."""
    cfg = load_engine_config()
    assert set(cfg.capabilities) >= {"instruct", "referee"}
    # Distinct model families — the architectural-independence invariant.
    inst, ref = cfg.capabilities["instruct"].model, cfg.capabilities["referee"].model
    assert inst.split("/")[0] != ref.split("/")[0]


# ── Proof-of-progress log scanner ────────────────────────────────────

def _mgr(tmp_path) -> VllmManager:
    cfg = EngineConfig(log_dir=str(tmp_path / "logs"))
    cfg.capabilities = {"instruct": ModelSpec(model="m")}
    return VllmManager(cfg)


def test_scan_log_tracks_progress(tmp_path):
    from atelier.engine.vllm_manager import Endpoint
    mgr = _mgr(tmp_path)
    log = tmp_path / "v.log"
    ep = Endpoint(capability="instruct", spec=ModelSpec(model="m"),
                  port=9999, gpu_ids=[0], log_path=log)

    log.write_text("INFO Loading model weights...\n")
    assert mgr._scan_log(ep) == ("loading weights", None)
    log.write_text("Loading model weights\nCapturing CUDA graph shapes\n")
    assert mgr._scan_log(ep) == ("capturing CUDA graphs", None)
    log.write_text("Uvicorn running on http://127.0.0.1:9999\n")
    assert mgr._scan_log(ep)[0] == "API server up"


def test_scan_log_fatal_short_circuits(tmp_path):
    from atelier.engine.vllm_manager import Endpoint
    mgr = _mgr(tmp_path)
    log = tmp_path / "v.log"
    log.write_text("Loading model weights\ntorch.OutOfMemoryError: CUDA out of memory\n")
    ep = Endpoint(capability="instruct", spec=ModelSpec(model="m"),
                  port=9999, gpu_ids=[0], log_path=log)
    _label, fatal = mgr._scan_log(ep)
    assert fatal == "GPU out of memory"


# ── Thinking-trace split ─────────────────────────────────────────────

def test_split_thinking_partitions_trace():
    content = "<think>step 1... step 2...</think>The answer is 42."
    answer, trace = split_thinking(content, "")
    assert answer == "The answer is 42."
    assert trace == "step 1... step 2..."


def test_split_thinking_defers_to_parser_output():
    answer, trace = split_thinking("plain answer", "parser trace")
    assert (answer, trace) == ("plain answer", "parser trace")
    # No delimiter, no parser output → everything stays in the answer.
    assert split_thinking("no tags here", "") == ("no tags here", "")


# ── GPU guard ────────────────────────────────────────────────────────

def test_foreign_holders_filters_by_gpu_and_threshold(monkeypatch):
    monkeypatch.setattr(gpu_guard, "_compute_apps", lambda: [
        (0, 111, 9000),   # foreign, on target, big → holder
        (1, 222, 100),    # under threshold → ignored
        (5, 333, 20000),  # off-target GPU → ignored
    ])
    holders = _foreign_holders([0, 1], min_mib=512)
    assert holders == [(0, 111, 9000)]


def test_default_threshold_counts_bare_cuda_context(monkeypatch):
    # Zero-share policy: an idle 384 MiB CUDA context (no model loaded)
    # is still occupancy at the defaults (guard signature + EngineConfig).
    monkeypatch.setattr(gpu_guard, "_compute_apps", lambda: [
        (0, 111, 384),
    ])
    import inspect
    guard_default = inspect.signature(claim_gpus).parameters["min_mib"].default
    for default in (guard_default, EngineConfig().gpu_min_free_mib):
        assert default < 300, "default must sit below bare-CUDA-context size"
        assert _foreign_holders([0], min_mib=default) == [(0, 111, 384)]


def test_claim_gpus_conflict_names_holder(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_guard, "_compute_apps", lambda: [])
    with claim_gpus([0, 1], lock_dir=tmp_path, role="test-a"):
        owner = json.loads((tmp_path / "gpu-0-1.owner.json").read_text())
        assert owner["role"] == "test-a"
        with pytest.raises(GpuClaimError, match="leased"):
            with claim_gpus([0, 1], lock_dir=tmp_path, role="test-b"):
                pass
    # Released after exit — a fresh claim succeeds.
    with claim_gpus([0, 1], lock_dir=tmp_path, role="test-c"):
        pass


def test_claim_gpus_refuses_foreign_vram(tmp_path, monkeypatch):
    monkeypatch.setattr(gpu_guard, "_compute_apps", lambda: [(2, 999, 8000)])
    with pytest.raises(GpuClaimError, match="foreign compute"):
        with claim_gpus([2, 3], lock_dir=tmp_path):
            pass
    # The advisory lock must have been released on refusal.
    with monkeypatch.context() as m:
        m.setattr(gpu_guard, "_compute_apps", lambda: [])
        with claim_gpus([2, 3], lock_dir=tmp_path):
            pass


# ── zndx.engine.v1 shared service (signals-protocol) ─────────────────


def test_zndx_service_registered_beside_native(tmp_path):
    """Both services answer on one port; the shared face reports project."""
    from concurrent import futures

    import grpc

    from atelier.engine.proto import atelier_engine_pb2 as pb
    from atelier.engine.proto import atelier_engine_pb2_grpc as pbg
    from atelier.engine.server import AtelierEngineServicer, ZndxEngineServicer
    from zndx.engine.v1 import engine_pb2 as zpb
    from zndx.engine.v1 import engine_pb2_grpc as zpbg

    class _FakeMgr:
        def status(self):
            return []

        def complete(self, cap, prompt, system_prompt, max_tokens,
                     temperature, json_schema=""):
            return {"text": f"echo:{cap}:{prompt}", "model": "fake",
                    "reasoning_content": "", "finish_reason": "stop",
                    "prompt_tokens": 1, "completion_tokens": 1,
                    "latency_ms": 0.1}

    native = AtelierEngineServicer.__new__(AtelierEngineServicer)
    native.mgr = _FakeMgr()
    native.cfg = EngineConfig()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_AtelierEngineServicer_to_server(native, server)
    zpbg.add_EngineServicer_to_server(ZndxEngineServicer(native), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        with grpc.insecure_channel(f"127.0.0.1:{port}") as ch:
            # Native service path.
            rn = pbg.AtelierEngineStub(ch).Complete(
                pb.CompleteRequest(capability="referee", prompt="hi"), timeout=5)
            assert rn.text == "echo:referee:hi"
            # Shared federation path — same port, different service identity.
            rz = zpbg.EngineStub(ch).Complete(
                zpb.CompleteRequest(capability="referee", prompt="hi"), timeout=5)
            assert rz.text == "echo:referee:hi"
            st = zpbg.EngineStub(ch).Status(zpb.StatusRequest(), timeout=5)
            assert st.project == "atelier"
            assert any(ep.capability == "referee" for ep in st.endpoints)
    finally:
        server.stop(grace=None)


# ── Event stream (telemetry substrate) ───────────────────────────────


def test_emit_writes_value_free_jsonl(tmp_path):
    from atelier.engine.events import EVENTS_FILE, emit

    emit(tmp_path, "endpoint_launch", capability="referee",
         model="org/m", port=8200, gpu_ids=[0, 1])
    emit(tmp_path, "complete", capability="referee", model="org/m",
         prompt_tokens=10, completion_tokens=5, latency_ms=12.5,
         finish_reason="stop", schema_constrained=True,
         reasoning_retained=False)

    lines = (tmp_path / EVENTS_FILE).read_text().strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(ln) for ln in lines)
    assert first["event"] == "endpoint_launch"
    assert first["run_id"] == second["run_id"]
    assert second["seq"] == first["seq"] + 1
    # The whole record vocabulary is structural — no content-bearing keys.
    assert "text" not in second and "prompt" not in second


def test_emit_refuses_non_allowlisted_keys(tmp_path):
    from atelier.engine.events import emit

    # The egress-membrane guard: content-shaped fields are refused loudly.
    with pytest.raises(ValueError, match="non-allowlisted"):
        emit(tmp_path, "complete", capability="referee",
             prompt="SELECT * FROM users")
