"""CLI: request cooperative cancellation of an in-flight pipeline run.

Usage::

    uv run python -m atelier.overwatch.kill_run <run_id> \\
        --reason "<free-form reason>" [--session <session_id>]

Sets ``BootstrapState.cancelled`` via the nautilus registry and stops
the associated nautilus watcher.  The pipeline polls the flag between
LLM batches (see :mod:`atelier.classify.bootstrap`), so this is a
cooperative cancel — not a SIGKILL.  An in-flight LLM call finishes
before the pipeline exits cleanly.

Like the other controlled CLIs this is gated: in ``propose`` /
``monitor`` tiers the supervisor may not cancel; an operator must do
it from the UI.  Autonomous tier passes through.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="atelier.overwatch.kill_run",
        description="Request cooperative cancellation of a pipeline run.",
    )
    ap.add_argument("run_id", help="FSM run id to cancel")
    ap.add_argument(
        "--reason", required=True,
        help="Free-form reason (written to cancellation_reason).",
    )
    ap.add_argument("--session", help="Supervisor session id (for the intervention log).")
    args = ap.parse_args(argv)

    try:
        from atelier.config import load_config
        cfg = load_config()
    except Exception as exc:
        print(f"error: could not load config: {exc}", file=sys.stderr)
        return 2

    autonomy = getattr(cfg, "overwatch_autonomy", "propose")
    if autonomy != "autonomous":
        print(
            f"rejected: overwatch.autonomy = {autonomy!r}; kill_run is only "
            "permitted in autonomous mode.  Non-autonomous supervisors may "
            "only record observations; cancellation belongs to the operator.",
            file=sys.stderr,
        )
        return 4

    from atelier.overwatch.nautilus import (
        get_state, get_active_watcher,
    )

    state = get_state(args.run_id)
    if state is None:
        print(
            f"error: no live state registered for run {args.run_id!r}. The "
            "run has either not started yet or already terminated.",
            file=sys.stderr,
        )
        return 2

    state.cancelled = True
    state.cancellation_reason = args.reason

    watcher = get_active_watcher(args.run_id)
    watcher_stopped = False
    if watcher is not None:
        try:
            watcher.stop()
            watcher_stopped = True
        except Exception:
            pass

    # Append to the supervisor session intervention log when available.
    session_recorded = False
    if args.session:
        try:
            from atelier.overwatch.session import (
                load_session, record_intervention,
            )
            sess = load_session(args.session, build_root=_project_root() / "build")
            if sess is not None:
                record_intervention(
                    sess, run_id=args.run_id,
                    trigger="operator_kill_cli",
                    detail=args.reason,
                    decision="cancelled",
                    build_root=_project_root() / "build",
                )
                session_recorded = True
        except Exception as exc:
            print(
                f"warning: could not log intervention to session: {exc}",
                file=sys.stderr,
            )

    print(json.dumps({
        "status": "cancel_requested",
        "run_id": args.run_id,
        "reason": args.reason,
        "watcher_stopped": watcher_stopped,
        "session_recorded": session_recorded,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
