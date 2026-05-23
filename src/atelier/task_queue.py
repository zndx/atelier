"""Restart-ready idempotent task queue.

A file-based work queue that survives AMP restarts and runs deferred
operations on the App pod without operator intervention.  Designed
for the post-restart "pick up where we left off" pattern: pre-restart,
the operator enqueues a chain of known tasks from the Session pod;
on boot, the gateway's lifespan kicks off a daemon processor that
walks the queue to completion.

Idempotency is per-handler — each handler checks for its expected
output artifact (a manifest, a verify JSON, a markdown guide, etc.)
before doing work.  Re-running a completed task is a no-op.  Crash
recovery uses PID liveness: an interrupted "running" task is moved
back to "pending" with attempt+1 on the next boot.

On-disk layout (under ``build/data/task_queue/``)::

  pending/<task_id>.json   waiting to run (sorted by enqueued_at)
  running/<task_id>.json   currently executing (carries pid + started_at)
  done/<task_id>.json      completed (carries outcome + duration)
  failed/<task_id>.json    failed after max_attempts (carries error)

Each task JSON::

  {
    "task_id": "<uuid>",
    "task_type": "<registered handler>",
    "params": {...},
    "enqueued_at": "<ISO>",
    "depends_on": ["<task_id>", ...],
    "max_attempts": 3,
    "attempt": 0,
    "idempotency_summary": "<short string for logs>"
  }

Operator surfaces (web terminal fallbacks):
  python -m atelier.task_queue list           # show queue state
  python -m atelier.task_queue retry <id>     # move failed → pending
  python -m atelier.task_queue cancel <id>    # delete a pending task
  python -m atelier.task_queue run-once       # synchronously drain
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid as _uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ── Paths ─────────────────────────────────────────────────────────


QUEUE_ROOT = Path("build/data/task_queue")
DIRS = {
    "pending": QUEUE_ROOT / "pending",
    "running": QUEUE_ROOT / "running",
    "done": QUEUE_ROOT / "done",
    "failed": QUEUE_ROOT / "failed",
}


def _ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Handler registry ──────────────────────────────────────────────


@dataclass
class TaskResult:
    """Return value from a handler.

    ``skipped`` set when the handler detected its output already
    exists (idempotent no-op).  ``outcome`` is a free-form dict the
    handler records for the audit trail.
    """
    skipped: bool = False
    outcome: dict = field(default_factory=dict)


HandlerFn = Callable[[dict], TaskResult]
_HANDLERS: dict[str, HandlerFn] = {}


def register(task_type: str) -> Callable[[HandlerFn], HandlerFn]:
    """Decorator: register a handler for a task_type."""
    def deco(fn: HandlerFn) -> HandlerFn:
        _HANDLERS[task_type] = fn
        return fn
    return deco


def list_handlers() -> list[str]:
    return sorted(_HANDLERS.keys())


# ── Enqueue ──────────────────────────────────────────────────────


def enqueue(
    task_type: str,
    params: dict,
    *,
    depends_on: list[str] | None = None,
    max_attempts: int = 3,
    idempotency_summary: str | None = None,
) -> str:
    """Write a pending task to disk.  Returns the task_id."""
    _ensure_dirs()
    task_id = str(_uuid.uuid4())
    payload = {
        "task_id": task_id,
        "task_type": task_type,
        "params": params,
        "enqueued_at": _now(),
        "depends_on": depends_on or [],
        "max_attempts": max_attempts,
        "attempt": 0,
        "idempotency_summary": idempotency_summary
            or f"{task_type}({list(params.keys())})",
    }
    path = DIRS["pending"] / f"{task_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Enqueued task %s (%s)", task_id[:8], task_type)
    return task_id


# ── State transitions ────────────────────────────────────────────


def _move(task: dict, from_bucket: str, to_bucket: str,
          *, augment: dict | None = None) -> Path:
    """Move a task JSON between buckets, optionally augmenting payload."""
    if augment:
        task = {**task, **augment}
    src = DIRS[from_bucket] / f"{task['task_id']}.json"
    dst = DIRS[to_bucket] / f"{task['task_id']}.json"
    dst.write_text(json.dumps(task, indent=2, default=str))
    if src.exists() and src != dst:
        src.unlink()
    return dst


def _scan(bucket: str) -> list[dict]:
    """Load every task JSON from one bucket, sorted by enqueued_at."""
    out = []
    for f in DIRS[bucket].glob("*.json"):
        try:
            out.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Bad task JSON %s: %s", f, exc)
    out.sort(key=lambda t: t.get("enqueued_at", ""))
    return out


# ── Crash recovery ───────────────────────────────────────────────


def _pid_alive(pid: int) -> bool:
    """POSIX liveness probe."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def recover_orphaned() -> int:
    """Move running/ tasks whose PID is dead back to pending/."""
    _ensure_dirs()
    recovered = 0
    for task in _scan("running"):
        pid = task.get("pid")
        if pid is not None and _pid_alive(pid):
            continue
        task["attempt"] = task.get("attempt", 0) + 1
        if task["attempt"] > task.get("max_attempts", 3):
            _move(task, "running", "failed", augment={
                "error": "exceeded_max_attempts_after_crash",
                "failed_at": _now(),
            })
        else:
            _move(task, "running", "pending", augment={
                "recovered_at": _now(),
                "recovered_from_pid": pid,
            })
            recovered += 1
    return recovered


# ── Dispatcher ───────────────────────────────────────────────────


def _deps_satisfied(task: dict, done_ids: set[str]) -> bool:
    deps = task.get("depends_on") or []
    return all(d in done_ids for d in deps)


def _next_runnable() -> dict | None:
    """Return the oldest pending task whose dependencies are all done."""
    done_ids = {t["task_id"] for t in _scan("done")}
    for task in _scan("pending"):
        if _deps_satisfied(task, done_ids):
            return task
    return None


def execute_one(task: dict) -> bool:
    """Dispatch one task.  Returns True if the queue should continue."""
    handler = _HANDLERS.get(task["task_type"])
    if handler is None:
        _move(task, "pending", "failed", augment={
            "error": f"unknown task_type {task['task_type']!r}",
            "failed_at": _now(),
        })
        return True

    started_at = _now()
    _move(task, "pending", "running", augment={
        "started_at": started_at,
        "pid": os.getpid(),
        "attempt": task.get("attempt", 0) + 1,
    })

    t0 = time.monotonic()
    try:
        result = handler(task.get("params") or {})
    except Exception as exc:  # noqa: BLE001
        elapsed = time.monotonic() - t0
        attempt = task.get("attempt", 0) + 1
        if attempt >= task.get("max_attempts", 3):
            _move({**task, "attempt": attempt}, "running", "failed", augment={
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "failed_at": _now(),
                "elapsed_s": round(elapsed, 2),
            })
            logger.error("Task %s failed permanently: %s",
                         task["task_id"][:8], exc)
        else:
            _move({**task, "attempt": attempt}, "running", "pending", augment={
                "last_error": f"{type(exc).__name__}: {exc}",
                "last_attempt_at": _now(),
            })
            logger.warning("Task %s attempt %d/%d failed (will retry): %s",
                           task["task_id"][:8], attempt,
                           task.get("max_attempts", 3), exc)
        return True
    elapsed = time.monotonic() - t0
    _move({**task}, "running", "done", augment={
        "completed_at": _now(),
        "elapsed_s": round(elapsed, 2),
        "skipped": result.skipped,
        "outcome": result.outcome,
    })
    logger.info(
        "Task %s (%s) %s in %.2fs",
        task["task_id"][:8], task["task_type"],
        "skipped (idempotent)" if result.skipped else "done",
        elapsed,
    )
    return True


def drain(*, max_iterations: int = 100) -> dict:
    """Run pending tasks until none remain runnable or budget exhausted.

    Returns a summary of what ran.  Crash recovery happens once at start.
    """
    _ensure_dirs()
    recovered = recover_orphaned()
    if recovered:
        logger.info("Recovered %d orphaned task(s) on drain start", recovered)
    ran = 0
    skipped = 0
    failed_now = 0
    for _ in range(max_iterations):
        task = _next_runnable()
        if task is None:
            break
        before_done = len(list(DIRS["done"].glob("*.json")))
        before_failed = len(list(DIRS["failed"].glob("*.json")))
        execute_one(task)
        after_done = len(list(DIRS["done"].glob("*.json")))
        after_failed = len(list(DIRS["failed"].glob("*.json")))
        if after_done > before_done:
            ran += 1
        if after_failed > before_failed:
            failed_now += 1
    pending = len(list(DIRS["pending"].glob("*.json")))
    return {
        "recovered_orphans": recovered,
        "ran": ran,
        "failed_this_drain": failed_now,
        "pending_remaining": pending,
    }


def drain_async() -> threading.Thread:
    """Spawn a daemon thread that drains the queue.  Returns the thread."""
    th = threading.Thread(
        target=lambda: _drain_with_log(),
        daemon=True, name="task-queue-drain",
    )
    th.start()
    return th


def _drain_with_log() -> None:
    try:
        result = drain()
        logger.info("Queue drain complete: %s", result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Queue drain crashed: %s", exc, exc_info=True)


# ── Gate helpers ─────────────────────────────────────────────────


def check_queue_clean() -> tuple[bool, str | None, dict]:
    """Inspect the queue for blockers that should refuse downstream actions.

    "Clean" means: no failed tasks, no pending tasks whose dependencies
    have failed (i.e., unrunnable pending).  An empty queue is clean.
    A queue with done-only tasks is clean (those are historical record).

    Returns ``(is_clean, error_message_or_None, state_summary)``.
    Callers that gate on queue state (e.g. fsm_start) call this AFTER
    draining and refuse the action if not clean — operator must resolve
    via ``python -m atelier.task_queue {retry|cancel}``.
    """
    _ensure_dirs()
    failed = _scan("failed")
    pending = _scan("pending")
    running = _scan("running")
    state = {
        "pending": len(pending),
        "running": len(running),
        "failed": len(failed),
    }

    if failed:
        ids_preview = ", ".join(
            f"{t['task_id'][:8]} ({t['task_type']})" for t in failed[:3]
        )
        more = "" if len(failed) <= 3 else f" (+{len(failed) - 3} more)"
        msg = (
            f"Task queue has {len(failed)} failed task(s): {ids_preview}"
            f"{more}.  Inspect via `python -m atelier.task_queue list`; "
            f"resolve root cause then `retry <task_id>` or `cancel <task_id>`."
        )
        return False, msg, state

    if pending:
        # If pending tasks remain after a drain attempt, their dependencies
        # are unsatisfied (dep failed previously, or dep is in running)
        # OR drain wasn't run yet.  Either way: not clean.
        ids_preview = ", ".join(
            f"{t['task_id'][:8]} ({t['task_type']})" for t in pending[:3]
        )
        more = "" if len(pending) <= 3 else f" (+{len(pending) - 3} more)"
        msg = (
            f"Task queue has {len(pending)} pending task(s) waiting to run: "
            f"{ids_preview}{more}.  Run `python -m atelier.task_queue run-once` "
            f"to drain, or `cancel <task_id>` to skip."
        )
        return False, msg, state

    return True, None, state


def drain_then_check(*, max_iterations: int = 100) -> tuple[bool, str | None, dict]:
    """Atomically drain pending tasks and report whether the queue is clean.

    Convenience wrapper for gate-callers (e.g. fsm_start): runs drain(),
    then check_queue_clean().  Returns the (is_clean, error, state) tuple
    from check_queue_clean, with the drain summary merged into state.
    """
    drain_summary = drain(max_iterations=max_iterations)
    is_clean, msg, state = check_queue_clean()
    state["drain"] = drain_summary
    return is_clean, msg, state


# ── CLI ──────────────────────────────────────────────────────────


def _cli_list() -> int:
    _ensure_dirs()
    for bucket in ("pending", "running", "done", "failed"):
        tasks = _scan(bucket)
        print(f"\n=== {bucket}: {len(tasks)} ===")
        for t in tasks:
            ttype = t.get("task_type", "?")
            tid = t.get("task_id", "?")[:8]
            summary = t.get("idempotency_summary", "")
            extra = ""
            if bucket == "done":
                if t.get("skipped"):
                    extra = "  (skipped idempotent)"
                else:
                    extra = f"  ({t.get('elapsed_s', '?')}s)"
            elif bucket == "failed":
                extra = f"  ERROR: {(t.get('error') or '')[:80]}"
            elif bucket == "running":
                extra = f"  pid={t.get('pid')}"
            print(f"  {tid}  {ttype:32}  {summary}{extra}")
    return 0


def _cli_retry(task_id: str) -> int:
    _ensure_dirs()
    # Search done and failed
    for bucket in ("failed", "done"):
        for f in DIRS[bucket].glob(f"{task_id}*.json"):
            task = json.loads(f.read_text())
            task["attempt"] = 0
            task["retried_at"] = _now()
            _move(task, bucket, "pending")
            print(f"Moved {task['task_id'][:8]} from {bucket} to pending")
            return 0
    print(f"Task {task_id} not found in done/failed", file=sys.stderr)
    return 1


def _cli_cancel(task_id: str) -> int:
    _ensure_dirs()
    for f in DIRS["pending"].glob(f"{task_id}*.json"):
        f.unlink()
        print(f"Cancelled pending task {f.stem[:8]}")
        return 0
    print(f"Task {task_id} not in pending", file=sys.stderr)
    return 1


def _cli_run_once() -> int:
    """Synchronously drain — for manual operator use."""
    # Ensure handlers are loaded
    from atelier import task_handlers  # noqa: F401
    print("Available handlers:", list_handlers())
    result = drain()
    print(f"\nDrain result: {json.dumps(result, indent=2)}")
    return 0 if result.get("pending_remaining", 0) == 0 else 1


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="atelier.task_queue")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="Show queue state")
    pr = sub.add_parser("retry", help="Move a failed/done task back to pending")
    pr.add_argument("task_id")
    pc = sub.add_parser("cancel", help="Delete a pending task")
    pc.add_argument("task_id")
    sub.add_parser("run-once", help="Synchronously drain the queue")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if args.cmd == "list":
        return _cli_list()
    if args.cmd == "retry":
        return _cli_retry(args.task_id)
    if args.cmd == "cancel":
        return _cli_cancel(args.task_id)
    if args.cmd == "run-once":
        return _cli_run_once()
    return 2


if __name__ == "__main__":
    sys.exit(main())
