"""GPU detection and CUDA preflight validation.

Ported from Signals (sigint/config.py) — identical detection logic,
different import paths. Results are cached for process lifetime since
GPU hardware doesn't change mid-run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuInfo:
    """GPU detection result from preflight."""

    available: bool
    device_count: int = 0
    driver_version: str = ""
    driver_cuda_version: str = ""
    pytorch_cuda_version: str = ""
    devices: list[str] = field(default_factory=list)
    vram_total_mib: list[int] = field(default_factory=list)
    vram_free_mib: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Foreign compute processes hold the cards (co-tenant engine serving).
    # When True, VRAM numbers came from nvidia-smi and no CUDA context was
    # created by this probe (zero-share policy).
    occupied: bool = False

    @property
    def resolved_device(self) -> str:
        """Return 'cuda' if GPUs are usable, else 'cpu'."""
        return "cuda" if self.available else "cpu"

    @property
    def resolved_devices(self) -> list[str]:
        """Return ['cuda:0', 'cuda:1', ...] for all usable GPUs, or ['cpu']."""
        if not self.available:
            return ["cpu"]
        return [f"cuda:{i}" for i in range(self.device_count)]

    def summary(self) -> str:
        """Human-readable GPU status string."""
        if not self.device_count:
            return "No NVIDIA GPUs detected"
        if not self.available:
            return (
                f"{self.device_count}x GPU detected but CUDA unavailable "
                f"(driver CUDA {self.driver_cuda_version}, "
                f"PyTorch CUDA {self.pytorch_cuda_version})"
            )
        # Use the first device name as a model descriptor — all devices
        # are typically the same SKU, so reporting each is noise.
        model = self.devices[0] if self.devices else "GPU"
        if self.vram_total_mib:
            vram_gb = self.vram_total_mib[0] / 1024
            return (
                f"{self.device_count}x {model} ({vram_gb:.0f} GB each), "
                f"CUDA {self.driver_cuda_version}"
            )
        return f"{self.device_count}x {model}, CUDA {self.driver_cuda_version}"

    def to_dict(self) -> dict:
        """Dict form for the /api/acceleration endpoint."""
        return {
            "available": self.available,
            "device_count": self.device_count,
            "devices": self.devices,
            "vram_total_mib": self.vram_total_mib,
            "vram_free_mib": self.vram_free_mib,
            "driver_version": self.driver_version,
            "driver_cuda_version": self.driver_cuda_version,
            "pytorch_cuda_version": self.pytorch_cuda_version,
            "resolved_device": self.resolved_device,
            "resolved_devices": self.resolved_devices,
            "warnings": list(self.warnings),
            "occupied": self.occupied,
            "summary": self.summary(),
        }


_gpu_info_cache: GpuInfo | None = None


def preflight_gpu() -> GpuInfo:
    """Detect GPU availability and validate CUDA driver compatibility.

    Checks:
    1. nvidia-smi reachable -> GPU count, driver version, CUDA version
    2. torch.cuda.is_available() -> runtime compatibility
    3. Version mismatch detection with actionable fix guidance

    This runs at config load time so the resolved device is known before
    any model loading.  Warnings are surfaced in the preflight report
    but never block startup (CPU fallback is always safe).

    Results are cached for the process lifetime (GPU hardware doesn't
    change mid-run).
    """
    global _gpu_info_cache
    if _gpu_info_cache is not None:
        return _gpu_info_cache

    import os
    import re
    import shutil
    import subprocess

    warnings: list[str] = []
    device_count = 0
    driver_version = ""
    driver_cuda = ""
    pytorch_cuda = ""
    device_names: list[str] = []
    cuda_available = False
    smi_total: list[int] = []
    smi_free: list[int] = []
    foreign_pids: set[int] = set()

    # ── Step 1: Probe nvidia-smi for hardware ────────────────────
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(",")]
                    device_names.append(parts[0])
                    try:
                        smi_total.append(int(parts[1]))
                        smi_free.append(int(parts[2]))
                    except (IndexError, ValueError):
                        pass
                device_count = len(device_names)

            # Compute-apps occupancy (context-free). Any foreign process
            # holding a card means the zero-share window is not ours —
            # the torch VRAM loop below must be skipped.
            apps = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if apps.returncode == 0:
                foreign_pids = {
                    int(p) for p in apps.stdout.split() if p.strip().isdigit()
                } - {os.getpid()}

            # Get driver version
            result2 = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result2.returncode == 0:
                driver_version = result2.stdout.strip().splitlines()[0].strip()

            # Parse CUDA version from nvidia-smi header
            result3 = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result3.returncode == 0:
                for line in result3.stdout.splitlines():
                    if "CUDA Version" in line:
                        m = re.search(r"CUDA Version:\s*([\d.]+)", line)
                        if m:
                            driver_cuda = m.group(1)
                        break
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    # ── Step 2: Check PyTorch CUDA runtime ───────────────────────
    vram_total: list[int] = []
    vram_free: list[int] = []
    try:
        import torch

        pytorch_cuda = torch.version.cuda or ""
        cuda_available = torch.cuda.is_available()

        if cuda_available and foreign_pids:
            # Zero-share policy: foreign compute processes hold the cards
            # (co-tenant engine serving). torch.cuda.mem_get_info would pin
            # a bare CUDA context per device on THIS process for its whole
            # life — report nvidia-smi's numbers instead and create no
            # context at all. is_available()/version above are context-free.
            vram_total = list(smi_total)
            vram_free = list(smi_free)
            warnings.append(
                f"{len(foreign_pids)} compute process(es) hold the GPUs — "
                f"VRAM reported via nvidia-smi; no CUDA context created "
                f"(zero-share policy)"
            )
        elif cuda_available:
            # Probe per-device VRAM via the runtime (more accurate than
            # nvidia-smi which doesn't reflect the current process's
            # point-in-time visibility under CUDA_VISIBLE_DEVICES).
            for i in range(torch.cuda.device_count()):
                try:
                    free, total = torch.cuda.mem_get_info(i)
                    vram_total.append(int(total // (1024 * 1024)))
                    vram_free.append(int(free // (1024 * 1024)))
                except Exception:
                    break
            # When torch sees a different device count than nvidia-smi
            # (e.g. CUDA_VISIBLE_DEVICES filtering), trust torch — those
            # are the devices the pipeline can actually use.
            if torch.cuda.device_count() != device_count:
                device_count = torch.cuda.device_count()

        if not cuda_available and device_count > 0 and pytorch_cuda:
            # GPUs present but torch can't see them
            if driver_cuda and pytorch_cuda and driver_cuda.split(".")[0] == pytorch_cuda.split(".")[0]:
                # Same major version — likely a library path issue (common in nix)
                warnings.append(
                    f"{device_count}x GPU detected, CUDA {driver_cuda} (driver) / "
                    f"{pytorch_cuda} (PyTorch) — but torch.cuda.is_available()=False. "
                    f"Check LD_LIBRARY_PATH includes libcuda.so.1 directory."
                )
            else:
                warnings.append(
                    f"CUDA version mismatch: driver supports CUDA {driver_cuda}, "
                    f"PyTorch built for CUDA {pytorch_cuda}. "
                    f"Upgrade driver: sudo apt install nvidia-driver-570-open && sudo reboot"
                )
    except ImportError:
        if device_count > 0:
            warnings.append(
                "PyTorch not installed — GPUs detected but cannot be used. "
                "Install: uv add torch"
            )

    _gpu_info_cache = GpuInfo(
        available=cuda_available,
        device_count=device_count,
        driver_version=driver_version,
        driver_cuda_version=driver_cuda,
        pytorch_cuda_version=pytorch_cuda,
        devices=device_names,
        vram_total_mib=vram_total,
        vram_free_mib=vram_free,
        warnings=warnings,
        occupied=bool(foreign_pids),
    )
    return _gpu_info_cache


_gpu_info_isolated_cache: GpuInfo | None = None

_GPU_INFO_FIELDS = (
    "available", "device_count", "driver_version", "driver_cuda_version",
    "pytorch_cuda_version", "devices", "vram_total_mib", "vram_free_mib",
    "warnings", "occupied",
)


def preflight_gpu_isolated(timeout: float = 60.0) -> GpuInfo:
    """``preflight_gpu()`` in a child process — the caller never touches CUDA.

    ``torch.cuda.mem_get_info`` pins a bare CUDA context (~384 MiB) on every
    device for the life of the calling process. Under the zero-share GPU
    policy an idle context counts as occupancy, so long-lived processes (the
    gateway) must probe through a child that exits. Probe failures are NOT
    cached, so a transient failure recovers on the next call.
    """
    global _gpu_info_isolated_cache
    if _gpu_info_isolated_cache is not None:
        return _gpu_info_isolated_cache

    import json
    import subprocess
    import sys

    try:
        out = subprocess.run(
            [sys.executable, "-m", "atelier.classify.gpu"],
            capture_output=True, text=True, timeout=timeout,
        )
        # Last stdout line — import-time log noise must not break the parse.
        data = json.loads(out.stdout.strip().splitlines()[-1])
        info = GpuInfo(**{k: data[k] for k in _GPU_INFO_FIELDS})
    except Exception as exc:
        return GpuInfo(available=False, warnings=[f"isolated GPU probe failed: {exc}"])
    _gpu_info_isolated_cache = info
    return info


if __name__ == "__main__":
    import json
    print(json.dumps(preflight_gpu().to_dict()))
