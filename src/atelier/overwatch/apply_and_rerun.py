# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""CLI: apply a supervisor proposal's overlay and start a new pipeline run.

Usage::

    uv run python -m atelier.overwatch.apply_and_rerun <run_id> \\
        [--source-id <id>] [--session <session_id>]

Reads ``build/results/<run_id>/proposed_overlay.json`` (written by
``write_proposal``), revalidates it against the Bedrock invariant and
SETTINGS_METADATA, applies it to the live config overlay, and invokes
``gateway._fsm_start_inline`` to spawn a new pipeline run.

This CLI is the **only** allowed side-effecting channel for the
autonomous tier.  The ``propose`` tier must never invoke it — the
gateway-level ``overwatch_autonomy`` gate is the authoritative check
and is re-verified here as belt-and-suspenders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from atelier.overwatch.remediation import (
    OverlayProposal, RemediationError, validate_proposal,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _results_dir(run_id: str) -> Path:
    return _project_root() / "build" / "results" / run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="atelier.overwatch.apply_and_rerun",
        description="Apply a supervisor proposal and kick off a new pipeline run.",
    )
    ap.add_argument("run_id", help="FSM run id whose proposed_overlay.json to apply")
    ap.add_argument("--source-id", help="Data source id for the rerun (optional)")
    ap.add_argument("--session", help="Supervisor session id — marks proposal applied")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Validate + show what would be applied, but do not start a rerun.",
    )
    args = ap.parse_args(argv)

    # Config check — gateway-level autonomy gate is authoritative;
    # CLI rejects when it's clearly not in autonomous mode.
    try:
        from atelier.config import load_config
        cfg = load_config()
    except Exception as exc:
        print(f"error: could not load config: {exc}", file=sys.stderr)
        return 2

    autonomy = getattr(cfg, "overwatch_autonomy", "propose")
    if autonomy != "autonomous":
        print(
            f"rejected: overwatch.autonomy = {autonomy!r}; apply_and_rerun "
            "is only permitted in autonomous mode.  Propose-tier supervisors "
            "must leave application to the operator.",
            file=sys.stderr,
        )
        return 4

    proposal_path = _results_dir(args.run_id) / "proposed_overlay.json"
    if not proposal_path.exists():
        print(
            f"error: no proposed_overlay.json at {proposal_path}",
            file=sys.stderr,
        )
        return 2

    try:
        raw = json.loads(proposal_path.read_text())
        proposal = OverlayProposal.from_dict(raw)
        validate_proposal(proposal)
    except RemediationError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"error: could not load proposal: {exc}", file=sys.stderr)
        return 2

    # Belt-and-suspenders invariant re-check on the already-validated
    # overlay — cheap and makes it impossible to slip something through
    # by editing proposed_overlay.json between write_proposal and here.
    from atelier.overwatch.remediation import enforce_bedrock_only
    enforce_bedrock_only(proposal.overlay)

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "run_id": proposal.run_id,
            "overlay": proposal.overlay,
        }, indent=2))
        return 0

    from atelier.config_overlay import set_overlay
    set_overlay(proposal.overlay)

    # Mark applied in the session before triggering the rerun so the
    # "pending" → "applied" transition is durable even if the rerun
    # fails to start.
    if args.session:
        from atelier.overwatch.session import load_session, mark_proposal_applied
        sess = load_session(args.session, build_root=_project_root() / "build")
        if sess is not None:
            mark_proposal_applied(
                sess, after_run_id=proposal.run_id,
                build_root=_project_root() / "build",
            )

    # Kick off the rerun via the gateway's FSM start path — this keeps
    # nautilus attachment consistent with operator-initiated runs.
    try:
        from atelier.gateway import fsm_start
        result = fsm_start(source_id=args.source_id)
    except Exception as exc:
        print(f"error: fsm_start failed: {exc}", file=sys.stderr)
        return 5

    print(json.dumps({
        "status": "applied_and_started",
        "applied_overlay": proposal.overlay,
        "fsm_start_result": result,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
