"""Periodic forensics sampler for the gateway process.

Captures memory pressure, CPU load, FSM state, task-queue counts, and
per-process RSS into a JSONL file on a regular cadence.  Runs as a
daemon task under the gateway lifespan — no separate process or
external supervisor required.

Each sample is one JSON line in ``.app/forensics/samples.jsonl``.  When
the file exceeds ``_MAX_SAMPLES_BYTES``, it rotates to
``samples.jsonl.1`` (overwriting any prior rotation).  Bounded growth.

Format mirrors the ad-hoc samples written during a prior LLM_SWEEP
hang incident (only the ``digest.py`` reader was committed; the writer
was lost).  That reader keeps working unchanged against samples
produced here.

Operator surface (web terminal):
    python .app/forensics/digest.py 20      # show last 20 samples
    tail -f .app/forensics/samples.jsonl    # live stream

Sample shape (per line)::

  {
    "ts": "<ISO 8601 UTC>",
    "mem_cur": <bytes>, "mem_peak": <bytes>, "mem_limit": <bytes>,
    "load": "<1m> <5m> <15m>",
    "rss_kb": {"gateway": ..., "grpc": ..., "qdrant": ..., "pglite": ...},
    "pids":   {"gw": "...", "gr": "...", "qd": "...", "pg": "...", "sup": "..."},
    "fsm":    {"id": ..., "state": ..., "phase": ..., "heartbeat": ...,
               "err": ..., "http_ok": <0|1>},
    "sup":    {"state": ..., "restart_count": ...},
    "queue":  {"pending": ..., "running": ..., "done": ..., "failed": ...}
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────


_SAMPLES_PATH = Path(".app/forensics/samples.jsonl")
_DEFAULT_INTERVAL_S = 10.0
_MAX_SAMPLES_BYTES = 50 * 1024 * 1024  # 50 MB → rotate


# Process discovery: command-line patterns matched against /proc/<pid>/cmdline.
# Order matters — the first hit wins per slot.
_PROCESS_PATTERNS: dict[str, tuple[str, ...]] = {
    "qd":  ("qdrant/qdrant", "qdrant",),
    "pg":  ("pglite-server.mjs",),
    "gr":  ("atelier.server",),
    "sup": ("pglite_supervisor", "supervisor.py"),
}


# ── /proc readers ────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_int(p: Path, default: int = -1) -> int:
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return default


def _read_load() -> str:
    try:
        parts = Path("/proc/loadavg").read_text().split()[:3]
        return " ".join(parts)
    except OSError:
        return "n/a"


def _rss_kb(pid: int | None) -> int:
    if pid is None:
        return 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return 0
    return 0


def _find_pids() -> dict[str, int | None]:
    """Discover sibling-process PIDs by /proc/<pid>/cmdline pattern match."""
    pids: dict[str, int | None] = {
        "gw": os.getpid(),
        "qd": None, "pg": None, "gr": None, "sup": None,
    }
    try:
        proc = Path("/proc")
        entries = list(proc.iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_text().replace("\x00", " ")
        except OSError:
            continue
        if not cmdline:
            continue
        for slot, patterns in _PROCESS_PATTERNS.items():
            if pids[slot] is not None:
                continue
            if any(pat in cmdline for pat in patterns):
                try:
                    pids[slot] = int(entry.name)
                except ValueError:
                    pass
                break
    return pids


def _cgroup_mem() -> tuple[int, int, int]:
    """Return (cur_bytes, peak_bytes, limit_bytes).

    Tries cgroup v2 first (``/sys/fs/cgroup/memory.{current,peak,max}``),
    falls back to cgroup v1 (``/sys/fs/cgroup/memory/memory.usage_in_bytes``),
    then to ``/proc/meminfo`` for a host-level estimate.  All values
    are bytes; ``limit_bytes`` is 0 when unlimited / unknown.
    """
    # cgroup v2 (unified hierarchy)
    base_v2 = Path("/sys/fs/cgroup")
    cur = _read_int(base_v2 / "memory.current")
    if cur > 0:
        peak = _read_int(base_v2 / "memory.peak", default=cur)
        limit_raw = _read_int(base_v2 / "memory.max")
        # cgroup v2 returns "max" string when unlimited — _read_int returns -1
        limit = limit_raw if limit_raw > 0 else 0
        return cur, max(peak, cur), limit

    # cgroup v1
    base_v1 = Path("/sys/fs/cgroup/memory")
    cur = _read_int(base_v1 / "memory.usage_in_bytes")
    if cur > 0:
        peak = _read_int(base_v1 / "memory.max_usage_in_bytes", default=cur)
        limit = _read_int(base_v1 / "memory.limit_in_bytes", default=0)
        # v1 represents "no limit" as a huge sentinel
        if limit > (1 << 60):
            limit = 0
        return cur, max(peak, cur), limit

    # /proc/meminfo fallback (host-level)
    try:
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            if v:
                info[k.strip()] = int(v.strip().split()[0]) * 1024
        used = info.get("MemTotal", 0) - info.get("MemAvailable", 0)
        return used, used, info.get("MemTotal", 0)
    except (OSError, ValueError):
        return 0, 0, 0


# ── State snapshots ──────────────────────────────────────────────


def _fsm_snapshot() -> dict:
    """Read the FSM's current state via the DAO.

    Robust to PGlite-down scenarios: returns ``{"state": "error", ...}``
    rather than raising, so the sampler keeps running.
    """
    try:
        from atelier.classify import get_fsm
        from atelier.db.dao import AtelierDao
        fsm = get_fsm(dao=AtelierDao())
        status = fsm.get_status()
        if status is None:
            return {"state": "n/a"}
        state_obj = getattr(status, "state", None)
        return {
            "id": getattr(status, "id", None),
            "state": getattr(state_obj, "value", None) if state_obj else None,
            "phase": getattr(status, "phase", None),
            "heartbeat": str(getattr(status, "heartbeat", "") or ""),
            "err": (getattr(status, "error", "") or "")[:200],
            "http_ok": 1,
        }
    except Exception as exc:  # noqa: BLE001 — never let sampler crash
        return {"state": "error", "err": str(exc)[:200], "http_ok": 0}


def _queue_snapshot() -> dict:
    """Per-bucket pending/running/done/failed counts from the task queue."""
    try:
        from atelier.task_queue import DIRS
        return {
            bucket: len(list(DIRS[bucket].glob("*.json")))
            for bucket in ("pending", "running", "done", "failed")
        }
    except Exception:  # noqa: BLE001
        return {}


def _supervisor_snapshot() -> dict:
    """Read the PGlite supervisor's published state file.

    Format is JSON (see ``bin/start-app.sh`` / supervisor publisher).
    Extracts the fields the original digest.py reader expects (state,
    restart_count) plus a couple of additional signals (consecutive
    failures, circuit-broken flag) useful for hang diagnosis.
    """
    try:
        state_path = Path(".app/pglite-supervisor.state")
        if not state_path.is_file():
            return {"state": "unknown", "restart_count": 0}
        raw = json.loads(state_path.read_text())
        return {
            "state": raw.get("state", "unknown"),
            "restart_count": int(raw.get("restart_count") or 0),
            "consecutive_failures": int(
                raw.get("consecutive_failures") or 0
            ),
            "circuit_broken": bool(raw.get("circuit_broken") or False),
        }
    except (OSError, json.JSONDecodeError, ValueError):
        return {"state": "unknown", "restart_count": 0}


# ── Sampling loop ────────────────────────────────────────────────


def _take_sample() -> dict:
    """Build one sample dict; no I/O to the samples file."""
    pids = _find_pids()
    cur_mem, peak_mem, limit_mem = _cgroup_mem()
    return {
        "ts": _now_iso(),
        "mem_cur": cur_mem,
        "mem_peak": peak_mem,
        "mem_limit": limit_mem,
        "load": _read_load(),
        "rss_kb": {
            "gateway": _rss_kb(pids["gw"]),
            "grpc": _rss_kb(pids["gr"]),
            "qdrant": _rss_kb(pids["qd"]),
            "pglite": _rss_kb(pids["pg"]),
        },
        "pids": {k: (str(v) if v else "") for k, v in pids.items()},
        "fsm": _fsm_snapshot(),
        "sup": _supervisor_snapshot(),
        "queue": _queue_snapshot(),
    }


def _rotate_if_needed() -> None:
    try:
        size = _SAMPLES_PATH.stat().st_size
    except OSError:
        return
    if size <= _MAX_SAMPLES_BYTES:
        return
    rotated = _SAMPLES_PATH.parent / "samples.jsonl.1"
    try:
        if rotated.exists():
            rotated.unlink()
        _SAMPLES_PATH.rename(rotated)
        logger.info("forensics samples rotated → %s", rotated.name)
    except OSError as exc:
        logger.warning("forensics rotation failed: %s", exc)


def _write_sample(sample: dict) -> None:
    """Append one sample as a JSONL line.  Rotates if oversize first."""
    _SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed()
    line = json.dumps(sample, default=str)
    with _SAMPLES_PATH.open("a") as f:
        f.write(line + "\n")


async def _sampling_loop(interval_s: float) -> None:
    """Take samples forever, every ``interval_s`` seconds."""
    logger.info(
        "forensics: sampling every %.1fs → %s (rotates at %d MB)",
        interval_s, _SAMPLES_PATH, _MAX_SAMPLES_BYTES // (1024 * 1024),
    )
    while True:
        try:
            sample = await asyncio.to_thread(_take_sample)
            await asyncio.to_thread(_write_sample, sample)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning("forensics sampler tick failed: %s", exc)
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            raise


def start_sampling_task(interval_s: float | None = None) -> asyncio.Task:
    """Spawn the sampling loop as a daemon task; return the task handle.

    Call from the gateway's lifespan after services are up.  The task
    is daemon-style: cancellation during lifespan shutdown stops it
    cleanly without flushing a final sample.
    """
    interval = (
        interval_s if interval_s is not None
        else float(os.environ.get("ATELIER_FORENSICS_INTERVAL_S",
                                  _DEFAULT_INTERVAL_S))
    )
    return asyncio.create_task(
        _sampling_loop(interval), name="forensics-sampler",
    )
