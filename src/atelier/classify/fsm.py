"""AgentFSM — Finite State Machine for the classification pipeline.

Governs the background classification process visible on the Status page.
State transitions are persisted to PostgreSQL so the frontend can poll
/api/fsm/status for live progress.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class FSMState(str, Enum):
    """Classification pipeline states."""

    IDLE = "IDLE"
    LOADING_VOCAB = "LOADING_VOCAB"
    DISCOVERING = "DISCOVERING"
    SAMPLING = "SAMPLING"
    GENERATING_SYNTH = "GENERATING_SYNTH"
    TRAINING = "TRAINING"
    CLASSIFYING = "CLASSIFYING"
    FUSING = "FUSING"
    EVALUATING = "EVALUATING"
    CONVERGED = "CONVERGED"
    ERROR = "ERROR"


# Valid state transitions
_TRANSITIONS: dict[FSMState, set[FSMState]] = {
    FSMState.IDLE: {FSMState.LOADING_VOCAB},
    FSMState.LOADING_VOCAB: {FSMState.DISCOVERING, FSMState.ERROR},
    FSMState.DISCOVERING: {FSMState.SAMPLING, FSMState.ERROR},
    FSMState.SAMPLING: {FSMState.CLASSIFYING, FSMState.GENERATING_SYNTH, FSMState.ERROR},
    FSMState.GENERATING_SYNTH: {FSMState.TRAINING, FSMState.ERROR},
    FSMState.TRAINING: {FSMState.CLASSIFYING, FSMState.ERROR},
    FSMState.CLASSIFYING: {FSMState.FUSING, FSMState.ERROR},
    FSMState.FUSING: {FSMState.EVALUATING, FSMState.ERROR},
    FSMState.EVALUATING: {FSMState.CONVERGED, FSMState.IDLE, FSMState.ERROR},
    FSMState.CONVERGED: {FSMState.IDLE},
    FSMState.ERROR: {FSMState.IDLE},
}


@dataclass
class FSMRun:
    """A single classification pipeline run."""

    id: str
    state: FSMState = FSMState.IDLE
    started_at: str = ""
    updated_at: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state.value,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "config": self.config,
            "progress": self.progress,
            "error": self.error,
            "result_path": self.result_path,
        }


class AgentFSM:
    """Manages classification pipeline state with DB persistence.

    Usage::

        fsm = AgentFSM(dao)
        run = fsm.start_run(config={"connection_name": "default-impala"})
        fsm.advance(run.id, FSMState.LOADING_VOCAB, progress={"step": "loading"})
        ...
        status = fsm.get_status(run.id)
    """

    def __init__(self, dao=None):
        self._dao = dao
        self._runs: dict[str, FSMRun] = {}
        self._current_run_id: str | None = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_run(self, config: dict[str, Any] | None = None) -> FSMRun:
        """Create and persist a new classification run."""
        run_id = str(uuid.uuid4())[:8]
        now = self._now()
        run = FSMRun(
            id=run_id,
            state=FSMState.IDLE,
            started_at=now,
            updated_at=now,
            config=config or {},
        )
        self._runs[run_id] = run
        self._current_run_id = run_id
        self._persist(run)
        return run

    def advance(
        self,
        run_id: str,
        new_state: FSMState,
        *,
        progress: dict[str, Any] | None = None,
        error: str | None = None,
        result_path: str | None = None,
    ) -> FSMRun:
        """Transition a run to a new state."""
        run = self._runs.get(run_id)
        if run is None:
            run = self._load(run_id)
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            self._runs[run_id] = run

        valid = _TRANSITIONS.get(run.state, set())
        if new_state not in valid:
            raise ValueError(
                f"Invalid transition: {run.state.value} -> {new_state.value}. "
                f"Valid: {[s.value for s in valid]}"
            )

        run.state = new_state
        run.updated_at = self._now()
        if progress:
            run.progress.update(progress)
        if error:
            run.error = error
        if result_path:
            run.result_path = result_path

        self._persist(run)
        return run

    def get_status(self, run_id: str | None = None) -> FSMRun | None:
        """Get the current status of a run (or the latest run)."""
        if run_id is None:
            run_id = self._current_run_id
        if run_id is None:
            return None
        run = self._runs.get(run_id)
        if run is None:
            run = self._load(run_id)
        return run

    def get_current_run_id(self) -> str | None:
        """Return the ID of the most recent run."""
        return self._current_run_id

    def list_runs(self) -> list[FSMRun]:
        """List all runs (from memory + DB)."""
        if self._dao:
            return self._list_from_db()
        return list(self._runs.values())

    # ── Persistence ──────────────────────────────────────────────────

    def _persist(self, run: FSMRun) -> None:
        """Save run state to the database."""
        if self._dao is None:
            return
        try:
            self._dao.upsert_fsm_run(
                run_id=run.id,
                state=run.state.value,
                started_at=run.started_at,
                updated_at=run.updated_at,
                config=json.dumps(run.config),
                progress=json.dumps(run.progress),
                error=run.error,
                result_path=run.result_path,
            )
        except Exception:
            pass  # Graceful — FSM works in-memory even without DB

    def _load(self, run_id: str) -> FSMRun | None:
        """Load a run from the database."""
        if self._dao is None:
            return None
        try:
            row = self._dao.get_fsm_run(run_id)
            if row is None:
                return None
            return FSMRun(
                id=row["id"],
                state=FSMState(row["state"]),
                started_at=row.get("started_at", ""),
                updated_at=row.get("updated_at", ""),
                config=json.loads(row.get("config") or "{}"),
                progress=json.loads(row.get("progress") or "{}"),
                error=row.get("error"),
                result_path=row.get("result_path"),
            )
        except Exception:
            return None

    def _list_from_db(self) -> list[FSMRun]:
        """Load all runs from the database."""
        try:
            rows = self._dao.list_fsm_runs()
            return [
                FSMRun(
                    id=row["id"],
                    state=FSMState(row["state"]),
                    started_at=row.get("started_at", ""),
                    updated_at=row.get("updated_at", ""),
                    config=json.loads(row.get("config") or "{}"),
                    progress=json.loads(row.get("progress") or "{}"),
                    error=row.get("error"),
                    result_path=row.get("result_path"),
                )
                for row in rows
            ]
        except Exception:
            return list(self._runs.values())
