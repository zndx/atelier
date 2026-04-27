# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Supervisor session state — attempts, outcomes, and the final summary.

A supervisor session spans multiple FSM runs: the first pipeline run,
then any reruns the supervisor overwatch requests after a failure or a
remediated proposal.  Session state lives under
``build/results/session_{id}/`` so the UI and the final
``supervisor_summary.md`` can reconstruct the chronology — "every
failure mode observed and what eventually worked".

The session is deliberately append-only on disk: every attempt,
intervention, and proposal is written as soon as it happens.  If the
supervisor aborts midway, an operator can read the partial session
state and finish the job manually.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


_session_lock = threading.Lock()


@dataclass
class AttemptRecord:
    """Single pipeline attempt within a supervisor session."""

    run_id: str
    started_at: str
    ended_at: str | None = None
    outcome: str = "in_progress"  # "converged" | "errored" | "cancelled" | "in_progress"
    llm_coverage: float | None = None
    llm_agreement: float | None = None
    mean_gap: float | None = None
    accuracy: float | None = None  # populated when a curated reference is configured
    failed_columns: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ProposalRecord:
    """Supervisor-generated overlay proposal for a subsequent attempt."""

    at: str
    after_run_id: str
    overlay: dict[str, Any]
    rationale: str
    expected_effect: str
    trigger: str
    applied: bool = False
    applied_at: str | None = None


@dataclass
class InterventionLog:
    """Mid-run intervention surfaced by nautilus."""

    at: str
    run_id: str
    trigger: str
    detail: str
    decision: str
    supervisor_response: dict[str, Any] | None = None


@dataclass
class SupervisorSession:
    """State for a multi-run supervisor session."""

    session_id: str
    created_at: str
    max_retries: int = 3
    attempts: list[AttemptRecord] = field(default_factory=list)
    proposals: list[ProposalRecord] = field(default_factory=list)
    interventions: list[InterventionLog] = field(default_factory=list)
    summary_path: str | None = None
    status: str = "active"  # "active" | "complete" | "exhausted" | "aborted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "max_retries": self.max_retries,
            "attempts": [asdict(a) for a in self.attempts],
            "proposals": [asdict(p) for p in self.proposals],
            "interventions": [asdict(i) for i in self.interventions],
            "summary_path": self.summary_path,
            "status": self.status,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def session_dir(session_id: str, build_root: Path) -> Path:
    return build_root / "results" / f"session_{session_id}"


def _session_json(session_id: str, build_root: Path) -> Path:
    return session_dir(session_id, build_root) / "supervisor_session.json"


def create_session(
    *, build_root: Path, max_retries: int = 3, session_id: str | None = None,
) -> SupervisorSession:
    """Start a new supervisor session and materialize its directory."""
    sid = session_id or uuid.uuid4().hex[:8]
    sess = SupervisorSession(
        session_id=sid,
        created_at=_now_iso(),
        max_retries=max_retries,
    )
    _persist(sess, build_root=build_root)
    return sess


def load_session(session_id: str, build_root: Path) -> SupervisorSession | None:
    """Read an existing session state from disk."""
    path = _session_json(session_id, build_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        logger.exception("failed to load session %s", session_id)
        return None
    sess = SupervisorSession(
        session_id=data["session_id"],
        created_at=data["created_at"],
        max_retries=data.get("max_retries", 3),
        status=data.get("status", "active"),
        summary_path=data.get("summary_path"),
    )
    for a in data.get("attempts", []):
        sess.attempts.append(AttemptRecord(**a))
    for p in data.get("proposals", []):
        sess.proposals.append(ProposalRecord(**p))
    for i in data.get("interventions", []):
        sess.interventions.append(InterventionLog(**i))
    return sess


def _persist(sess: SupervisorSession, *, build_root: Path) -> None:
    with _session_lock:
        d = session_dir(sess.session_id, build_root)
        d.mkdir(parents=True, exist_ok=True)
        _session_json(sess.session_id, build_root).write_text(
            json.dumps(sess.to_dict(), indent=2, default=str) + "\n",
        )


def record_attempt_start(
    sess: SupervisorSession, *, run_id: str, build_root: Path,
) -> AttemptRecord:
    """Append a new in-progress attempt to the session."""
    att = AttemptRecord(run_id=run_id, started_at=_now_iso())
    sess.attempts.append(att)
    _persist(sess, build_root=build_root)
    return att


def record_attempt_outcome(
    sess: SupervisorSession,
    *,
    run_id: str,
    outcome: str,
    build_root: Path,
    llm_coverage: float | None = None,
    llm_agreement: float | None = None,
    mean_gap: float | None = None,
    accuracy: float | None = None,
    failed_columns: list[str] | None = None,
    notes: str = "",
) -> AttemptRecord | None:
    """Close out an attempt with its outcome + key metrics."""
    att = next(
        (a for a in sess.attempts if a.run_id == run_id and a.outcome == "in_progress"),
        None,
    )
    if att is None:
        logger.warning("no in-progress attempt for run %s", run_id)
        return None
    att.ended_at = _now_iso()
    att.outcome = outcome
    att.llm_coverage = llm_coverage
    att.llm_agreement = llm_agreement
    att.mean_gap = mean_gap
    att.accuracy = accuracy
    if failed_columns is not None:
        att.failed_columns = list(failed_columns)
    att.notes = notes
    _persist(sess, build_root=build_root)
    return att


def record_proposal(
    sess: SupervisorSession,
    *,
    after_run_id: str,
    overlay: dict[str, Any],
    rationale: str,
    expected_effect: str,
    trigger: str,
    build_root: Path,
) -> ProposalRecord:
    """Append a supervisor-generated overlay proposal to the session."""
    p = ProposalRecord(
        at=_now_iso(),
        after_run_id=after_run_id,
        overlay=dict(overlay),
        rationale=rationale,
        expected_effect=expected_effect,
        trigger=trigger,
    )
    sess.proposals.append(p)
    _persist(sess, build_root=build_root)
    return p


def mark_proposal_applied(
    sess: SupervisorSession, *, after_run_id: str, build_root: Path,
) -> ProposalRecord | None:
    """Mark the most recent proposal for a given run as applied."""
    p = next(
        (x for x in reversed(sess.proposals)
         if x.after_run_id == after_run_id and not x.applied),
        None,
    )
    if p is None:
        return None
    p.applied = True
    p.applied_at = _now_iso()
    _persist(sess, build_root=build_root)
    return p


def record_intervention(
    sess: SupervisorSession,
    *,
    run_id: str,
    trigger: str,
    detail: str,
    decision: str,
    build_root: Path,
    supervisor_response: dict[str, Any] | None = None,
) -> InterventionLog:
    """Log a nautilus intervention for inclusion in the final summary."""
    i = InterventionLog(
        at=_now_iso(),
        run_id=run_id,
        trigger=trigger,
        detail=detail,
        decision=decision,
        supervisor_response=supervisor_response,
    )
    sess.interventions.append(i)
    _persist(sess, build_root=build_root)
    return i


def finalize(
    sess: SupervisorSession,
    *,
    status: str,
    build_root: Path,
    summary_markdown: str | None = None,
) -> Path | None:
    """Close out a session; optionally write the final summary markdown."""
    sess.status = status
    summary_path: Path | None = None
    if summary_markdown is not None:
        summary_path = session_dir(sess.session_id, build_root) / "supervisor_summary.md"
        summary_path.write_text(summary_markdown)
        sess.summary_path = str(summary_path)
    _persist(sess, build_root=build_root)
    return summary_path


def should_stop(sess: SupervisorSession) -> bool:
    """True when the session has consumed its retry budget."""
    return len(sess.attempts) >= sess.max_retries


def default_summary_markdown(sess: SupervisorSession) -> str:
    """A plain, no-LLM-needed rendering of the session — fallback when
    the supervisor agent didn't produce its own summary.

    Populates the "every failure mode observed and what eventually
    worked" narrative from the structured records.  Used when the
    agent's synthesis turn fails or is disabled.
    """
    lines: list[str] = []
    lines.append(f"# Supervisor Session `{sess.session_id}`\n")
    lines.append(f"_Created: {sess.created_at} · Status: **{sess.status}**_\n\n")

    lines.append(f"## Attempts ({len(sess.attempts)})\n\n")
    for i, att in enumerate(sess.attempts, 1):
        lines.append(f"### Attempt {i} — run `{att.run_id}`\n")
        lines.append(f"- Outcome: **{att.outcome}**\n")
        lines.append(f"- Duration: {att.started_at} → {att.ended_at or '(still running)'}\n")
        if att.llm_coverage is not None:
            lines.append(f"- LLM Coverage: {att.llm_coverage:.1%}\n")
        if att.llm_agreement is not None:
            lines.append(f"- LLM Agreement: {att.llm_agreement:.1%}\n")
        if att.mean_gap is not None:
            lines.append(f"- Mean belief gap: {att.mean_gap:.3f}\n")
        if att.accuracy is not None:
            lines.append(f"- Accuracy vs curated reference: {att.accuracy:.1%}\n")
        if att.failed_columns:
            lines.append(f"- Failed columns ({len(att.failed_columns)}): "
                         f"{', '.join(att.failed_columns[:5])}"
                         f"{'…' if len(att.failed_columns) > 5 else ''}\n")
        if att.notes:
            lines.append(f"- Notes: {att.notes}\n")
        lines.append("\n")

    if sess.interventions:
        lines.append(f"## Mid-run interventions ({len(sess.interventions)})\n\n")
        for i in sess.interventions:
            lines.append(f"- `{i.at}` run `{i.run_id}` — **{i.trigger}** "
                         f"({i.decision}): {i.detail}\n")
        lines.append("\n")

    if sess.proposals:
        lines.append(f"## Proposals ({len(sess.proposals)})\n\n")
        for p in sess.proposals:
            applied = "applied" if p.applied else "pending"
            lines.append(f"### Proposal after `{p.after_run_id}` — {applied}\n")
            lines.append(f"- Trigger: {p.trigger}\n")
            lines.append(f"- Rationale: {p.rationale}\n")
            lines.append(f"- Expected effect: {p.expected_effect}\n")
            keys = ", ".join(sorted(p.overlay.keys())) or "(none)"
            lines.append(f"- Overlay keys: {keys}\n\n")

    if sess.attempts and sess.attempts[-1].outcome == "converged":
        lines.append("## Outcome\n\n")
        lines.append(f"Final attempt `{sess.attempts[-1].run_id}` converged. "
                     "Earlier attempts (if any) are documented above.\n")
    elif sess.status == "exhausted":
        lines.append("## Outcome\n\n")
        lines.append(f"Retry budget ({sess.max_retries}) exhausted without "
                     "convergence. Operator review required.\n")

    return "".join(lines)
