# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Rolling TTFT + throughput stats for the Web Terminal Agent.

Each completed terminal query records an observation keyed by
``(provider, model_id)``.  Median-based summaries over the last 20
observations per key surface on the Status-page panel so operators
can see, at a glance, how responsive each provider is for them.

Kept deliberately simple: in-memory ``deque`` per key, thread-safe
writes via a single lock (stats are low-volume — one write per
terminal prompt — so contention is nil).

Stats reset on gateway restart. Not persisted; the panel is an
operational read-out, not an archive.
"""

from __future__ import annotations

import statistics
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


_WINDOW_SIZE = 20
_lock = threading.Lock()


@dataclass
class _Sample:
    ttft_ms: float
    duration_ms: float
    input_tokens: int
    output_tokens: int


# Key: (provider, model_id) — model_id is the catalog entry id so
# stats are tied to the UX-facing identity, not the underlying ARN.
_buffers: dict[tuple[str, str], deque] = {}


def record(
    provider: str,
    model_id: str,
    *,
    ttft_ms: float,
    duration_ms: float,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Append a single observation to the rolling window.

    Negative / zero values are clamped rather than rejected so a
    bad measurement doesn't silently drop the sample — downstream
    summary math handles zeros gracefully.
    """
    key = (provider, model_id)
    sample = _Sample(
        ttft_ms=max(0.0, float(ttft_ms)),
        duration_ms=max(0.0, float(duration_ms)),
        input_tokens=max(0, int(input_tokens)),
        output_tokens=max(0, int(output_tokens)),
    )
    with _lock:
        buf = _buffers.get(key)
        if buf is None:
            buf = deque(maxlen=_WINDOW_SIZE)
            _buffers[key] = buf
        buf.append(sample)


def _tokps(s: _Sample) -> float:
    """Steady-state tok/s = output / (duration − ttft).

    Subtracting TTFT isolates the generation phase so startup cost
    doesn't deflate the number the operator cares about.  Returns
    ``0.0`` when the sample is too short to give a meaningful rate
    (avoids ``ZeroDivisionError`` and implausibly large values).
    """
    gen_ms = s.duration_ms - s.ttft_ms
    if gen_ms <= 100.0 or s.output_tokens <= 0:
        return 0.0
    return s.output_tokens / (gen_ms / 1000.0)


def summary(provider: str, model_id: str) -> dict[str, Any]:
    """Return ``{n, ttft_ms_p50, tokps_p50}`` for a key.

    ``n`` is the current window size (capped at 20); zero when the
    key has never been observed.  Medians use ``statistics.median``
    which handles the odd sample count naturally.  When ``n < 3``
    the medians are still computed but the UI should render them as
    "—" to keep operators from over-trusting a two-sample trend.
    """
    with _lock:
        buf = _buffers.get((provider, model_id))
        samples = list(buf) if buf else []

    if not samples:
        return {"n": 0, "ttft_ms_p50": 0.0, "tokps_p50": 0.0}

    ttfts = [s.ttft_ms for s in samples]
    tokpses = [_tokps(s) for s in samples]
    return {
        "n": len(samples),
        "ttft_ms_p50": round(statistics.median(ttfts), 1),
        "tokps_p50": round(statistics.median(tokpses), 2),
    }


def reset(provider: str | None = None, model_id: str | None = None) -> None:
    """Clear one key, all keys for a provider, or everything.

    Used by tests; no operator-facing surface today.
    """
    with _lock:
        if provider is None and model_id is None:
            _buffers.clear()
            return
        keys_to_drop = [
            k for k in _buffers
            if (provider is None or k[0] == provider)
            and (model_id is None or k[1] == model_id)
        ]
        for k in keys_to_drop:
            del _buffers[k]


def all_summaries() -> dict[tuple[str, str], dict[str, Any]]:
    """Snapshot of every observed key → summary.  Convenience for the
    gateway endpoint that stamps stats onto catalog entries."""
    with _lock:
        keys = list(_buffers.keys())
    return {k: summary(*k) for k in keys}
