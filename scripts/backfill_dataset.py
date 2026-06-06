#!/usr/bin/env python3
"""CLI seam over ``atelier.db.sync.sync_filesystem_to_db``.

The Atelier gateway runs the same reconcile pass automatically at
lifespan startup (see ``_sync_orphaned_runs`` in ``gateway.py``), so
operator-driven backfill is no longer the primary path.  This script
exists for two cases the auto-sync can't cover:

  1. Forcing a specific source_id when the snapshot doesn't carry one
     (early runs from before the snapshot field landed) and the
     ``ml_artifact_sets`` row's source_id is wrong.
  2. Reconciling a single run on demand without restarting the AMP.

Run inside the Atelier Application pod (where ``ATELIER_DB_URL`` is
exported by ``bin/start-app.sh``):

    python scripts/backfill_dataset.py <run_id>
    python scripts/backfill_dataset.py <run_id> --source-id hive-poc/default
    python scripts/backfill_dataset.py --all                # whole tree
    python scripts/backfill_dataset.py <run_id> --dry-run

The legacy single-run, source-id-required form is preserved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "run_id", nargs="?",
        help="Run id to reconcile.  Omit with --all to scan the whole tree.",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Reconcile every run under the results directory.",
    )
    ap.add_argument(
        "--source-id", default=None,
        help="Force this source_id (otherwise resolved from settings_snapshot.json or the artifact set row).",
    )
    ap.add_argument(
        "--results-dir", default="build/results",
        help="Base results directory (default: build/results).",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the resolution plan without touching the DB.",
    )
    args = ap.parse_args()

    if args.all and args.run_id:
        ap.error("pass --all OR a run_id, not both")
    if not args.all and not args.run_id:
        ap.error("pass a run_id or --all")

    results_root = Path(args.results_dir)
    if not results_root.is_dir():
        sys.exit(f"error: {results_root} is not a directory")

    if args.dry_run:
        target = "all runs" if args.all else args.run_id
        print(json.dumps({
            "dry_run": True,
            "target": target,
            "results_root": str(results_root),
            "source_id_override": args.source_id,
            "hint": "auto-sync runs at gateway startup; this CLI is for explicit one-off reconciles",
        }, indent=2))
        return 0

    from atelier.db.sync import sync_filesystem_to_db

    report = sync_filesystem_to_db(
        results_root,
        source_id_override=args.source_id,
        only_run_id=args.run_id if not args.all else None,
    )

    print(report.summary_line())
    for outcome in report.outcomes:
        marker = (
            "✓" if (outcome.artifact_set == "registered"
                    or outcome.dataset == "registered")
            else "·" if outcome.artifact_set == "already_registered"
                       and outcome.dataset == "already_registered"
            else "!"
        )
        print(
            f"  {marker} {outcome.run_id}  "
            f"artifact_set={outcome.artifact_set:<20} "
            f"dataset={outcome.dataset:<20} "
            f"source_id={outcome.source_id!r}"
            + (f"  -- {outcome.note}" if outcome.note else "")
        )

    return 2 if report.errors else 0


if __name__ == "__main__":
    sys.exit(main())
