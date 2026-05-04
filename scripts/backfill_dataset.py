#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Backfill DB rows for a classify run whose post-pipeline registration failed.

Re-runs ``dao.upsert_dataset`` and ``dao.register_artifact_set`` from the
artifacts on disk, so a run that reached CONVERGED but failed its DB
write (e.g. transient PGlite outage during a multi-hour bootstrap) becomes
visible in the Embeddings page and usable as an Extend Classification base.

Run inside the Atelier Application pod (where ``ATELIER_DB_URL`` resolves):

    python scripts/backfill_dataset.py <run_id>
    python scripts/backfill_dataset.py <run_id> --source-id impala-poc/foo
    python scripts/backfill_dataset.py <run_id> --results-dir build/results

Without ``--source-id``, falls back to ``build/state/last_active_source.txt``
which the gateway maintains per source switch.  Pass ``--dry-run`` to print
the upsert payload without touching the DB.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_classifications(results_dir: Path) -> list[dict]:
    p = results_dir / "classifications.json"
    if not p.exists():
        sys.exit(f"error: {p} not found — cannot backfill without classifications")
    return json.loads(p.read_text())


def _resolve_source_id(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    state_file = Path("build/state/last_active_source.txt")
    if state_file.exists():
        val = state_file.read_text().strip()
        return val or None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_id", help="Run id (matches build/results/<run_id>/)")
    ap.add_argument(
        "--source-id",
        default=None,
        help="Data source id; defaults to build/state/last_active_source.txt",
    )
    ap.add_argument(
        "--results-dir",
        default="build/results",
        help="Base results directory (default: build/results)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the upsert payload but do not touch the DB",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir) / args.run_id
    if not results_dir.is_dir():
        sys.exit(f"error: {results_dir} is not a directory")

    parquet_path = results_dir / "atelier_embeddings.parquet"
    if not parquet_path.exists():
        sys.exit(f"error: {parquet_path} not found — pipeline did not finish")

    classifications = _load_classifications(results_dir)
    source_id = _resolve_source_id(args.source_id)
    if not source_id:
        print(
            "warning: no source_id resolved — dataset row will be unattached "
            "(use --source-id to bind it)",
            file=sys.stderr,
        )

    dataset_payload = {
        "dataset_id": args.run_id,
        "name": f"Classification {args.run_id[:8]}",
        "parquet_path": str(parquet_path),
        "description": f"{len(classifications)} columns classified",
        "row_count": len(classifications),
        "source_id": source_id,
        "is_active": True,
        "summary": f"{len(classifications)} columns (backfilled)",
        "fsm_run_id": args.run_id,
        "artifact_set_id": None,    # resolved below after artifact-set register
        "parent_dataset_id": None,
        "run_kind": "classify",
    }

    if args.dry_run:
        print(json.dumps({"dataset_upsert": dataset_payload}, indent=2))
        return 0

    from atelier.db.dao import AtelierDao

    dao = AtelierDao()

    # Order matters: ``datasets.artifact_set_id`` is a FK to
    # ``ml_artifact_sets(id)``.  Register the artifact set first so the
    # dataset row's FK has a target; if registration fails or there's
    # no artifact set to register, leave ``artifact_set_id=None`` and
    # the dataset still lands (Extend wiring incomplete but Embeddings
    # panel surfaces the run).
    artifact_set_registered = False
    try:
        from atelier.classify.artifact_set import build_artifact_set_record
        from atelier.config import load_config

        cfg = load_config()
        spec = build_artifact_set_record(
            run_id=args.run_id,
            results_dir=results_dir,
            cfg=cfg,
            n_columns=len(classifications),
            source_id=source_id,
            fsm_run_id=args.run_id,
        )
        if spec is None:
            print("• no ML artifact set to register (no catboost classes sidecar)")
        else:
            try:
                dao.register_artifact_set(**spec)
                artifact_set_registered = True
                print(f"✓ artifact set {args.run_id} registered")
            except Exception as exc:
                # Already-registered case is benign — the FK target
                # exists, which is what we need.
                msg = str(exc).lower()
                if "duplicate" in msg or "unique" in msg or "already exists" in msg:
                    artifact_set_registered = True
                    print(f"• artifact set {args.run_id} already registered — proceeding")
                else:
                    raise
    except Exception as exc:
        print(f"! artifact set registration failed: {exc}", file=sys.stderr)
        # Continue anyway — dataset will land with artifact_set_id=None.

    if artifact_set_registered:
        dataset_payload["artifact_set_id"] = args.run_id

    if source_id:
        dataset_payload["version_number"] = dao.next_version_number(source_id)
    else:
        dataset_payload["version_number"] = 1

    dao.upsert_dataset(**dataset_payload)
    if source_id:
        dao.set_active_version(source_id, args.run_id)
    print(
        f"✓ dataset {args.run_id} upserted "
        f"(source_id={source_id!r}, artifact_set_id="
        f"{dataset_payload['artifact_set_id']!r})"
    )

    # Mark the sidecar resolved (don't delete — keep audit trail).
    err_path = results_dir / "register_error.json"
    if err_path.exists():
        resolved = err_path.with_suffix(".json.resolved")
        err_path.rename(resolved)
        print(f"• moved register_error.json → {resolved.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
