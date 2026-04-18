"""Nautilus — mid-run classification pipeline watcher.

A background thread that polls the FSM + ``BootstrapState.batch_audit``
and decides when the run is going sideways enough that a supervisor
intervention is warranted.  The thread itself is observation + decision
framing; actually invoking the supervisor overwatch lives in Pillar 3
and is delivered through the ``intervene_callback`` hook so nautilus
stays testable without tool-using agents in the loop.

Triggers (any one fires an intervention):

* **Stall** — no new batch_audit activity for ``stall_threshold_s`` while
  the FSM is still in an active (non-terminal) state.  Catches the
  20-minute hangs observed in UAT where the LLM sweep froze on a
  problem batch and no heartbeat advanced.
* **Slow LLM_SWEEP** — ``fsm.state == LLM_SWEEP`` for more than
  ``llm_sweep_threshold_s`` regardless of batch progress.  Sets a
  ceiling on how long we let the sweep grind before re-evaluating
  columns-per-call / batching strategy.
* **Accumulated batch failures** — count of audit entries with status
  ``failed`` / ``fatal`` exceeds ``failed_batch_threshold``.  The
  halving retry (Pillar 1) preserves 100% coverage at the column
  level but a shower of failures still signals the config is wrong.
* **ERROR state** — pipeline transitioned to ``ERROR``.  Unconditional;
  supervisor reads the error and decides whether to retry with an
  overlay or escalate.

Each trigger records an ``InterventionRecord``; the supervisor overwatch
(Pillar 3) consumes the latest record when it runs.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from atelier.classify.bootstrap import BootstrapState
    from atelier.classify.fsm import AgentFSM

logger = logging.getLogger(__name__)


# ── Shared observation registry ──────────────────────────────────
#
# Pipeline registers its live BootstrapState by run_id; nautilus looks
# the state up by run_id on each poll.  A lock guards insert/remove
# and lookup so we never observe a partially-destructed state during
# pipeline teardown.

_registry_lock = threading.Lock()
_state_registry: dict[str, "BootstrapState"] = {}


def register_state(run_id: str, state: "BootstrapState") -> None:
    """Register the live pipeline state so nautilus can observe it."""
    with _registry_lock:
        _state_registry[run_id] = state


def unregister_state(run_id: str) -> None:
    """Drop the reference after the run ends."""
    with _registry_lock:
        _state_registry.pop(run_id, None)


def get_state(run_id: str) -> "BootstrapState | None":
    """Return the live state for a run, or None if not registered."""
    with _registry_lock:
        return _state_registry.get(run_id)


# ── Records ──────────────────────────────────────────────────────


@dataclass
class NautilusConfig:
    """Trigger thresholds for the nautilus watcher."""

    enabled: bool = True
    poll_interval_s: float = 10.0
    stall_threshold_s: float = 120.0
    llm_sweep_threshold_s: float = 300.0
    failed_batch_threshold: int = 10
    # When True, nautilus is authorized to request cancellation of the
    # run (cooperative: pipeline checks ``state.cancelled``).  Gated by
    # ``overwatch.autonomy == "autonomous"`` upstream.
    can_cancel: bool = False


def nautilus_config_from_cfg(cfg) -> NautilusConfig:
    """Build NautilusConfig from an AtelierConfig."""
    autonomy = getattr(cfg, "overwatch_autonomy", "propose")
    return NautilusConfig(
        enabled=getattr(cfg, "overwatch_nautilus_enabled", True),
        poll_interval_s=getattr(cfg, "overwatch_nautilus_poll_interval_s", 10.0),
        stall_threshold_s=getattr(cfg, "overwatch_nautilus_stall_threshold_s", 120.0),
        llm_sweep_threshold_s=getattr(cfg, "overwatch_nautilus_llm_sweep_threshold_s", 300.0),
        failed_batch_threshold=getattr(cfg, "overwatch_nautilus_failed_batch_threshold", 10),
        can_cancel=(autonomy == "autonomous"),
    )


# Trigger kinds — stable strings for the UI and supervisor prompt
TRIGGER_STALL = "stall"
TRIGGER_SLOW_SWEEP = "slow_llm_sweep"
TRIGGER_FAILED_BATCHES = "failed_batches"
TRIGGER_FSM_ERROR = "fsm_error"


@dataclass
class InterventionRecord:
    """Single nautilus intervention decision.

    Stored in ``NautilusWatcher.interventions`` so the UI and the
    supervisor overwatch can reconstruct why an intervention fired and
    what (if anything) was requested of the classifier.
    """

    run_id: str
    trigger: str
    trigger_detail: str
    fsm_state: str
    fsm_updated_at: str
    batch_audit_len: int
    failed_batch_count: int
    at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision: str = "observed"  # "observed" | "intervened" | "cancelled"
    supervisor_response: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "trigger": self.trigger,
            "trigger_detail": self.trigger_detail,
            "fsm_state": self.fsm_state,
            "fsm_updated_at": self.fsm_updated_at,
            "batch_audit_len": self.batch_audit_len,
            "failed_batch_count": self.failed_batch_count,
            "at": self.at,
            "decision": self.decision,
            "supervisor_response": self.supervisor_response,
        }


@dataclass
class _Heartbeat:
    """Per-poll snapshot used for stall detection across polls."""

    fsm_state: str = ""
    fsm_updated_at: str = ""
    audit_len: int = 0
    last_audit_change: float = 0.0
    phase_entered_at: float = 0.0


# ── Watcher ──────────────────────────────────────────────────────


_TERMINAL_STATES = {"IDLE", "CONVERGED", "ERROR"}


class NautilusWatcher:
    """Background watcher for a single pipeline run.

    Start one with :meth:`start`; it polls on a daemon thread until
    :meth:`stop` is called or the FSM enters a terminal state.  The
    trigger evaluation is pure: callers can exercise
    :meth:`evaluate_triggers` directly against a seeded heartbeat + run
    for tests without spinning threads.
    """

    def __init__(
        self,
        run_id: str,
        fsm: "AgentFSM",
        cfg: NautilusConfig,
        *,
        intervene_callback: Callable[[InterventionRecord], dict[str, Any] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.run_id = run_id
        self._fsm = fsm
        self._cfg = cfg
        self._intervene = intervene_callback
        self._clock = clock
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._heartbeat = _Heartbeat()
        self._fired_triggers: set[str] = set()
        self.interventions: list[InterventionRecord] = []

    # ── lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background poll thread."""
        if not self._cfg.enabled:
            logger.info("nautilus disabled for run %s", self.run_id)
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name=f"nautilus-{self.run_id}", daemon=True,
        )
        self._thread.start()
        logger.info("nautilus started for run %s", self.run_id)

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the poll thread to exit and wait briefly."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ── poll loop ──────────────────────────────────────────────

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                try:
                    done = self.tick()
                except Exception:
                    logger.exception("nautilus tick failed for %s", self.run_id)
                    done = False
                if done:
                    break
                self._stop.wait(self._cfg.poll_interval_s)
        finally:
            logger.info("nautilus exiting for run %s", self.run_id)

    def tick(self) -> bool:
        """One poll cycle.  Returns True when the run has terminated."""
        run = self._fsm.get_status(self.run_id)
        if run is None:
            return False

        state_name = run.state.value if hasattr(run.state, "value") else str(run.state)
        now = self._clock()

        # Heartbeat advance.  Phase change (includes first observation,
        # where the previous fsm_state is "") resets the phase-entry
        # clock and the "last audit activity" clock so thresholds are
        # measured from when we started watching this phase, and
        # re-arms the phase-scoped triggers so they fire once per phase.
        if state_name != self._heartbeat.fsm_state:
            self._heartbeat.phase_entered_at = now
            self._heartbeat.last_audit_change = now
            self._fired_triggers.discard(TRIGGER_SLOW_SWEEP)
            self._fired_triggers.discard(TRIGGER_STALL)

        audit_len, failed_count = self._observe_audit()
        if audit_len > self._heartbeat.audit_len:
            self._heartbeat.last_audit_change = now

        self._heartbeat.fsm_state = state_name
        self._heartbeat.fsm_updated_at = run.updated_at
        self._heartbeat.audit_len = audit_len

        triggers = self.evaluate_triggers(
            state_name=state_name,
            now=now,
            failed_count=failed_count,
        )

        for trigger, detail in triggers:
            # Each trigger kind fires at most once per phase — prevents
            # a stall detector from firing every 10 s once the threshold
            # is exceeded.  Reset when the phase changes.
            if trigger in self._fired_triggers:
                continue
            self._fired_triggers.add(trigger)
            rec = InterventionRecord(
                run_id=self.run_id,
                trigger=trigger,
                trigger_detail=detail,
                fsm_state=state_name,
                fsm_updated_at=run.updated_at,
                batch_audit_len=audit_len,
                failed_batch_count=failed_count,
            )
            self._dispatch(rec)

        return state_name in _TERMINAL_STATES

    # ── trigger evaluation (pure) ─────────────────────────────

    def evaluate_triggers(
        self,
        *,
        state_name: str,
        now: float,
        failed_count: int,
    ) -> list[tuple[str, str]]:
        """Return [(trigger, detail), ...] that fired for this poll.

        Pure function of heartbeat + cfg; separated from the thread so
        tests can drive trigger decisions deterministically without
        relying on wall-clock timing.
        """
        triggered: list[tuple[str, str]] = []
        cfg = self._cfg
        hb = self._heartbeat

        if state_name == "ERROR":
            triggered.append((TRIGGER_FSM_ERROR, "FSM transitioned to ERROR"))
            return triggered  # nothing else to evaluate once errored

        if state_name in _TERMINAL_STATES:
            return triggered  # CONVERGED / IDLE — no intervention needed

        if state_name == "LLM_SWEEP":
            in_phase = now - hb.phase_entered_at
            if in_phase > cfg.llm_sweep_threshold_s:
                triggered.append((
                    TRIGGER_SLOW_SWEEP,
                    f"LLM_SWEEP has run {in_phase:.0f}s "
                    f"(threshold {cfg.llm_sweep_threshold_s:.0f}s)",
                ))

        # Stall: no audit activity for > stall_threshold while FSM not terminal.
        since_change = now - hb.last_audit_change
        if since_change > cfg.stall_threshold_s:
            triggered.append((
                TRIGGER_STALL,
                f"no batch_audit activity for {since_change:.0f}s "
                f"(threshold {cfg.stall_threshold_s:.0f}s) "
                f"in state={state_name}",
            ))

        if failed_count >= cfg.failed_batch_threshold:
            triggered.append((
                TRIGGER_FAILED_BATCHES,
                f"{failed_count} failed/fatal batch entries exceeds "
                f"threshold {cfg.failed_batch_threshold}",
            ))

        return triggered

    # ── dispatch ───────────────────────────────────────────────

    def _dispatch(self, rec: InterventionRecord) -> None:
        """Invoke the intervene callback and record the response."""
        logger.warning(
            "nautilus trigger fired: run=%s trigger=%s detail=%s",
            rec.run_id, rec.trigger, rec.trigger_detail,
        )
        self.interventions.append(rec)
        if self._intervene is None:
            return
        try:
            response = self._intervene(rec)
        except Exception:
            logger.exception("intervene callback failed for %s", rec.run_id)
            return
        if response is None:
            return
        rec.supervisor_response = response
        decision = response.get("decision", "observed")
        rec.decision = decision

        if decision == "cancelled" and self._cfg.can_cancel:
            self._request_cancel(
                reason=response.get("reason", rec.trigger_detail),
            )

    def _request_cancel(self, *, reason: str) -> None:
        """Cooperative cancellation — set the state flag the pipeline checks."""
        state = get_state(self.run_id)
        if state is None:
            logger.warning(
                "cancel requested for %s but no live state registered", self.run_id,
            )
            return
        state.cancelled = True
        state.cancellation_reason = reason
        logger.warning("nautilus requested cancel for %s: %s", self.run_id, reason)

    # ── introspection ──────────────────────────────────────────

    def _observe_audit(self) -> tuple[int, int]:
        """Return (len(batch_audit), count(failed|fatal))."""
        state = get_state(self.run_id)
        if state is None:
            return 0, 0
        audit = state.batch_audit
        failed = sum(
            1 for a in audit if a.status in ("failed", "fatal")
        )
        return len(audit), failed

    def status(self) -> dict[str, Any]:
        """Snapshot for the gateway /api/overwatch/nautilus/{run_id} route."""
        audit_len, failed = self._observe_audit()
        state = get_state(self.run_id)
        cancelled = bool(state and getattr(state, "cancelled", False))
        return {
            "run_id": self.run_id,
            "running": self.running,
            "enabled": self._cfg.enabled,
            "heartbeat": {
                "fsm_state": self._heartbeat.fsm_state,
                "fsm_updated_at": self._heartbeat.fsm_updated_at,
                "audit_len": audit_len,
                "failed_count": failed,
            },
            "cancelled": cancelled,
            "interventions": [r.to_dict() for r in self.interventions],
        }


# ── Global watcher tracking for the gateway ─────────────────────
#
# The gateway hangs onto the live watcher so /api/overwatch/nautilus
# and later a POST cancel endpoint can reach it without plumbing the
# reference through pipeline internals.

_watcher_lock = threading.Lock()
_active_watchers: dict[str, NautilusWatcher] = {}


def set_active_watcher(run_id: str, watcher: NautilusWatcher) -> None:
    with _watcher_lock:
        _active_watchers[run_id] = watcher


def clear_active_watcher(run_id: str) -> None:
    with _watcher_lock:
        _active_watchers.pop(run_id, None)


def get_active_watcher(run_id: str) -> NautilusWatcher | None:
    with _watcher_lock:
        return _active_watchers.get(run_id)


def list_active_watchers() -> list[NautilusWatcher]:
    with _watcher_lock:
        return list(_active_watchers.values())
