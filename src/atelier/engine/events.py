"""Engine event stream — the substrate for the non-OTel telemetry channel.

Append-only JSONL at ``<log_dir>/events.jsonl``. A MiNiFi sidecar tails this
file and ships events to the signals NiFi (design:
docs/notes/2026-07-03 MiNiFi sidecar note); until then it is a local audit
trail the supervisor and operators can read directly.

**Value-free by construction (egress-membrane invariant):** events carry
lifecycle facts and numeric metadata ONLY — capability names, model ids,
token counts, latencies, ports, GPU ids, finish reasons. Never prompt text,
never completion text, never sample values. ``emit()`` enforces a key
allowlist so a future call site cannot accidentally widen the schema; new
keys are added HERE, deliberately, or the event is refused.

Correlation: every event carries ``seq`` (per-process monotonic) and the
engine ``run_id`` (one UUID per engine process — the Gaius correlation-id
pattern), so NiFi-side FlowFiles stitch to engine lifetimes.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

EVENTS_FILE = "events.jsonl"

# The complete event vocabulary. Adding a key here is a schema decision —
# review against the egress membrane (no value-bearing content, ever).
_ALLOWED_KEYS = {
    "capability", "model", "port", "gpu_ids", "healthy", "detail_kind",
    "prompt_tokens", "completion_tokens", "latency_ms", "finish_reason",
    "schema_constrained", "reasoning_retained",
    "rc", "reason", "label", "elapsed_s", "grpc_port", "capabilities",
    "project",
}

_lock = threading.Lock()
_seq = 0
RUN_ID = uuid.uuid4().hex[:12]


def emit(log_dir: str | Path, event: str, **fields) -> None:
    """Append one value-free event. Refuses keys outside the allowlist."""
    global _seq
    bad = set(fields) - _ALLOWED_KEYS
    if bad:
        raise ValueError(
            f"event {event!r} carries non-allowlisted keys {sorted(bad)} — "
            f"extend _ALLOWED_KEYS deliberately (egress-membrane review) "
            f"instead of passing them."
        )
    with _lock:
        _seq += 1
        record = {
            "ts": time.time(),
            "run_id": RUN_ID,
            "seq": _seq,
            "pid": os.getpid(),
            "event": event,
            **fields,
        }
        path = Path(log_dir) / EVENTS_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # telemetry must never break serving
            logger.warning("event emit failed (%s): %s", event, exc)
