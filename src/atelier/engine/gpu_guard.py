"""GPU-exclusivity lease — belt-and-braces guard against double-committing VRAM.

Mirrors Ægir's ``gpu_guard`` with one deliberate divergence: the default lock
dir is the SHARED ``/tmp/zndx-gpu-leases`` (proposed cross-project convention,
running-observations note §7) rather than a per-project dir — advisory
filelocks only protect holders who look in the same place. The authoritative
cross-process check remains the ``nvidia-smi`` compute-apps probe, which sees
Ægir/Gaius vLLM workers regardless of lock-dir convention.

Two mechanisms:
1. **Lock file** — one ``filelock`` per GPU set (``gpu-0-1-2-3.lock``) plus an
   ``.owner.json`` recording pid/project/host/time. Advisory; auto-released
   when the holder dies.
2. **nvidia-smi probe** — refuses the claim when a foreign process holds ≥
   ``min_mib`` on any target GPU. Degrades to lock-only when nvidia-smi is
   absent (CPU boxes, CI).

The default ``min_mib`` sits BELOW bare-CUDA-context size (~300–500 MiB on
consumer cards): an idle context counts as occupancy. Zero-share policy —
memory-sharing on consumer GPUs doesn't work well enough to be worth it;
a holder must fully vacate (context gone from compute-apps), not merely
free VRAM.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GpuClaimError(RuntimeError):
    """A target GPU is held by a foreign process or lease."""


def _compute_apps() -> list[tuple[int, int, int]]:
    """[(gpu_index, pid, used_mib)] from nvidia-smi; [] when unavailable."""
    try:
        uuid_map = {}
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        for line in out.stdout.strip().splitlines():
            idx, uuid = [p.strip() for p in line.split(",", 1)]
            uuid_map[uuid] = int(idx)
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory,gpu_uuid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return []
        apps = []
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            pid_s, mib_s, uuid = [p.strip() for p in line.split(",", 2)]
            apps.append((uuid_map.get(uuid, -1), int(pid_s), int(mib_s)))
        return apps
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return []


def _foreign_holders(gpu_ids: list[int], min_mib: int) -> list[tuple[int, int, int]]:
    """Compute-apps on the target GPUs that aren't us or our children."""
    me = os.getpid()
    holders = []
    for gpu, pid, mib in _compute_apps():
        if gpu in gpu_ids and mib >= min_mib and pid != me:
            holders.append((gpu, pid, mib))
    return holders


@contextlib.contextmanager
def claim_gpus(
    gpu_ids: list[int],
    *,
    lock_dir: str | Path = "/tmp/zndx-gpu-leases",
    min_mib: int = 64,
    role: str = "atelier-engine",
):
    """Hold an exclusive lease on ``gpu_ids`` for the duration of the context.

    Raises :class:`GpuClaimError` (naming the holder) when the lease or the
    nvidia-smi probe says the GPUs are taken.
    """
    from filelock import FileLock, Timeout

    lock_dir = Path(lock_dir)
    lock_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(str(g) for g in sorted(gpu_ids))
    lock_path = lock_dir / f"gpu-{tag}.lock"
    owner_path = lock_dir / f"gpu-{tag}.owner.json"

    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0.5)
    except Timeout:
        owner = "?"
        if owner_path.exists():
            with contextlib.suppress(Exception):
                owner = owner_path.read_text().strip()
        raise GpuClaimError(f"GPUs {gpu_ids} leased by another engine: {owner}") from None

    try:
        holders = _foreign_holders(gpu_ids, min_mib)
        if holders:
            lock.release()
            desc = ", ".join(f"gpu{g}: pid {p} ({m} MiB)" for g, p, m in holders)
            raise GpuClaimError(
                f"GPUs {gpu_ids} hold foreign compute processes — {desc}. "
                f"Another engine (Ægir :50151? Gaius :50051?) is serving on them."
            )
        owner_path.write_text(json.dumps({
            "pid": os.getpid(), "role": role, "host": socket.gethostname(),
        }))
        logger.info("claimed GPUs %s (%s)", gpu_ids, lock_path.name)
        yield
    finally:
        if lock.is_locked:
            lock.release()
        with contextlib.suppress(OSError):
            owner_path.unlink(missing_ok=True)
