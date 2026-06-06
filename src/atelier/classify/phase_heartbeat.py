"""Phase heartbeat — proof-of-progress observability for pre-LLM phases.

The bootstrap LLM-sweep loop already advances ``FSM.updated_at`` on
every batch via ``pipeline._sweep_progress``, so nautilus's stall
detector can distinguish a running sweep from a stalled one.  But the
phases that run BEFORE the sweep (LOADING_VOCAB, DISCOVERING,
SAMPLING) advance the FSM only at phase boundaries — a Hive query
that hangs for five minutes inside ``load_annotations_from_hive``
leaves ``updated_at`` frozen and looks identical to a dead thread.

This module's :func:`phase_heartbeat` is a context manager that spawns
a daemon thread to ping the FSM every ``interval_s`` seconds while
the body runs.  Each ping advances ``updated_at`` (FSM.advance always
does) so nautilus's existing ``stall_threshold_s`` machinery catches
pre-LLM stalls without any nautilus changes.

Pairs with the lightweight liveness + metadata probes in
:mod:`atelier.classify.sampler` (``_probe_connection``,
``_probe_table_shape``).  Probes catch obviously-broken setups
upfront with a short fixed timeout; the heartbeat keeps the FSM
ticking through legitimately-slow heavy queries so nautilus can
distinguish "actively waiting on Hive" from "thread died."

Project memory: this is the proof-of-progress paradigm — "instead of
fixed timeouts (not ready in 30s), use progress-based timeouts (no
verifiable progress in 10s)."  Originally applied to PGlite startup
in ``bin/start-app.sh``; here it generalizes to Hive-backed pipeline
phases.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from atelier.classify.fsm import AgentFSM, FSMState

logger = logging.getLogger(__name__)


@contextmanager
def phase_heartbeat(
    fsm: "AgentFSM",
    run_id: str,
    state: "FSMState",
    *,
    interval_s: float = 5.0,
    label: str | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager that heartbeats the FSM during a long-running phase.

    Spawns a daemon thread that calls ``fsm.advance(run_id, state,
    progress={...})`` every ``interval_s`` seconds while the body
    runs.  ``progress["phase_heartbeat_at"]`` carries an ISO timestamp;
    ``progress["phase_seconds"]`` carries elapsed seconds.  The body
    can also stash custom counters into the yielded dict; those are
    merged into each subsequent heartbeat payload so the body's own
    progress (e.g. "tables sampled so far") shows up alongside the
    timestamp.

    Args:
        fsm: The AgentFSM instance to ping.
        run_id: The run to advance.
        state: The current FSM state — pinged as a same-state advance.
        interval_s: How often to ping.  5s is the default; tighten if
            you want nautilus to react faster, loosen if the phase is
            short and you don't need fine-grained ticks.
        label: Optional human-readable phase label for log lines.

    Yields:
        A mutable dict the body can write to.  Keys merged into each
        heartbeat payload as ``progress["phase_<key>"] = value``.

    Example::

        with phase_heartbeat(fsm, run_id, FSMState.SAMPLING,
                             interval_s=5.0, label="sampling") as ctx:
            for i, table in enumerate(tables):
                sample_table_metadata(...)
                ctx["tables_sampled"] = i + 1
    """
    started_at = time.monotonic()
    stop = threading.Event()
    ctx: dict[str, Any] = {}

    def _beat() -> None:
        try:
            elapsed = round(time.monotonic() - started_at, 1)
            payload: dict[str, Any] = {
                "phase_heartbeat_at": datetime.now(timezone.utc).isoformat(),
                "phase_seconds": elapsed,
            }
            if label:
                payload["phase_label"] = label
                # Structured sub-phase block consumed by the nested
                # progress UI's tree-builder.  Reads ``current`` /
                # ``total`` from the body's ``ctx`` when present —
                # otherwise the row renders as indeterminate (spinner
                # only, no bar).  See
                # ``.claude/plans/lovely-doodling-badger.md`` (nested
                # progress visibility plan).
                sub_phase: dict[str, Any] = {
                    "id": label,
                    "elapsed_s": elapsed,
                }
                if "current" in ctx and "total" in ctx:
                    sub_phase["current"] = ctx["current"]
                    sub_phase["total"] = ctx["total"]
                payload["sub_phase"] = sub_phase
            for k, v in ctx.items():
                payload[f"phase_{k}"] = v
            fsm.advance(run_id, state, progress=payload)
        except Exception:
            # Heartbeat is observability — never let it abort the phase.
            logger.debug("phase_heartbeat ping failed (non-fatal)", exc_info=True)

    def _loop() -> None:
        # Emit one ping immediately so updated_at moves at body entry,
        # then poll until stop is signaled.
        _beat()
        while not stop.wait(interval_s):
            _beat()

    thread = threading.Thread(
        target=_loop, name=f"phase-heartbeat-{label or state.name}",
        daemon=True,
    )
    thread.start()
    try:
        yield ctx
    finally:
        stop.set()
        # Best-effort join — don't block phase exit if the thread is
        # mid-FSM-call.  The daemon flag ensures it dies with the
        # process at worst.
        thread.join(timeout=interval_s + 2.0)
