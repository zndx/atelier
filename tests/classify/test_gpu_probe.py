"""GPU preflight probe — zero-share occupancy behavior.

Under foreign compute-app occupancy the probe must never call
``torch.cuda.mem_get_info`` (it would pin a bare CUDA context per device
on the calling process for its whole life) and must report VRAM from
nvidia-smi instead. Hermetic: nvidia-smi and torch.cuda are stubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import atelier.classify.gpu as gpu
from atelier.classify.gpu import _GPU_INFO_FIELDS, GpuInfo, preflight_gpu


def _fake_nvidia_smi(compute_app_pids: str):
    def run(argv, **kwargs):
        joined = " ".join(argv)
        if "query-gpu=name,memory.total,memory.free" in joined:
            out = "NVIDIA GeForce RTX 4090, 24091, 1788\nNVIDIA GeForce RTX 4090, 24091, 1790\n"
        elif "query-gpu=driver_version" in joined:
            out = "570.148.08\n"
        elif "query-compute-apps=pid" in joined:
            out = compute_app_pids
        else:  # bare `nvidia-smi` — CUDA version header parse
            out = "| NVIDIA-SMI 570.148.08    Driver Version: 570.148.08    CUDA Version: 12.8 |\n"
        return SimpleNamespace(returncode=0, stdout=out)
    return run


@pytest.fixture
def probe_env(monkeypatch):
    monkeypatch.setattr(gpu, "_gpu_info_cache", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    return monkeypatch


def test_occupied_probe_skips_torch_vram_and_reports_smi(probe_env):
    probe_env.setattr("subprocess.run", _fake_nvidia_smi("2167014\n2167015\n"))

    def _forbidden(_i):
        raise AssertionError("mem_get_info must not be called under occupancy")
    probe_env.setattr(torch.cuda, "mem_get_info", _forbidden)

    info = preflight_gpu()
    assert info.occupied is True
    assert info.available is True
    assert info.vram_total_mib == [24091, 24091]
    assert info.vram_free_mib == [1788, 1790]
    assert any("zero-share" in w for w in info.warnings)


def test_free_cards_probe_uses_runtime_vram(probe_env):
    probe_env.setattr("subprocess.run", _fake_nvidia_smi(""))
    probe_env.setattr(
        torch.cuda, "mem_get_info",
        lambda i: (20_000 * 1024 * 1024, 24_000 * 1024 * 1024),
    )

    info = preflight_gpu()
    assert info.occupied is False
    assert info.vram_total_mib == [24_000, 24_000]
    assert info.vram_free_mib == [20_000, 20_000]
    assert not any("zero-share" in w for w in info.warnings)


def test_gpuinfo_roundtrips_through_isolated_probe_fields(probe_env):
    probe_env.setattr("subprocess.run", _fake_nvidia_smi("2167014\n"))
    probe_env.setattr(torch.cuda, "mem_get_info", lambda i: (0, 0))

    info = preflight_gpu()
    data = info.to_dict()
    rebuilt = GpuInfo(**{k: data[k] for k in _GPU_INFO_FIELDS})
    assert rebuilt == info
