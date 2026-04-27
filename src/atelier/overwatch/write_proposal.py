# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""CLI: validate + persist a supervisor overlay proposal.

Usage::

    uv run python -m atelier.overwatch.write_proposal <run_id> \\
        --json '{"overlay": {...}, "rationale": "...", \\
                 "expected_effect": "...", "trigger": "..."}'

    uv run python -m atelier.overwatch.write_proposal <run_id> \\
        --from-file path/to/proposal.json

    uv run python -m atelier.overwatch.write_proposal <run_id> \\
        --json '...' --session <session_id>

This is one of the narrow output channels the supervisor agent is
allowed to invoke via its sandboxed Bash tool.  The CLI validates the
payload against SETTINGS_METADATA and the Bedrock-only invariant, then
writes ``proposed_overlay.json`` next to the run artifacts and (if a
session is provided) appends a ProposalRecord to the session state.

No ``--apply`` here — actually applying the overlay and restarting
the pipeline is a separate CLI (``apply_and_rerun``) so the autonomy
tier can gate it.
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


def _load_payload(args: argparse.Namespace) -> dict:
    if args.json is not None:
        return json.loads(args.json)
    if args.from_file is not None:
        return json.loads(Path(args.from_file).read_text())
    raise SystemExit("error: one of --json or --from-file is required")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="atelier.overwatch.write_proposal",
        description="Validate and persist a supervisor overlay proposal.",
    )
    ap.add_argument("run_id", help="FSM run id this proposal follows")
    ap.add_argument("--json", help="Proposal payload as a JSON string")
    ap.add_argument("--from-file", help="Path to a JSON file holding the payload")
    ap.add_argument("--session", help="Supervisor session id — appends a ProposalRecord")
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Validate only — do not write proposed_overlay.json or touch session state.",
    )
    args = ap.parse_args(argv)

    try:
        raw = _load_payload(args)
    except Exception as exc:
        print(f"error: could not read proposal payload: {exc}", file=sys.stderr)
        return 2

    # Let from_dict / validate_proposal produce the structured error.
    data = dict(raw)
    data.setdefault("run_id", args.run_id)
    try:
        proposal = OverlayProposal.from_dict(data)
        accepted = validate_proposal(proposal)
    except RemediationError as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 3

    if args.dry_run:
        print(json.dumps({
            "status": "dry_run_ok",
            "accepted_keys": accepted,
        }, indent=2))
        return 0

    out_dir = _results_dir(args.run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "proposed_overlay.json"
    out_path.write_text(json.dumps(proposal.to_dict(), indent=2) + "\n")

    session_path: Path | None = None
    if args.session:
        from atelier.overwatch.session import (
            load_session, record_proposal,
        )
        build_root = _project_root() / "build"
        sess = load_session(args.session, build_root=build_root)
        if sess is None:
            print(f"warning: session {args.session} not found; skipping "
                  "session log", file=sys.stderr)
        else:
            rec = record_proposal(
                sess,
                after_run_id=proposal.run_id,
                overlay=proposal.overlay,
                rationale=proposal.rationale,
                expected_effect=proposal.expected_effect,
                trigger=proposal.trigger,
                build_root=build_root,
            )
            session_path = build_root / "results" / f"session_{sess.session_id}" / "supervisor_session.json"
            del rec  # silence pyright unused

    print(json.dumps({
        "status": "ok",
        "proposed_overlay_path": str(out_path),
        "session_state_path": str(session_path) if session_path else None,
        "accepted_keys": accepted,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
