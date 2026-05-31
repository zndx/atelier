#!/usr/bin/env python
"""Pre-restart helper: enqueue the apply → verify → guide chain.

Writes a dependency-chained set of tasks to the file-based queue at
``build/data/task_queue/pending/``.  On the next App pod boot, the
gateway lifespan's ``_kick_task_queue`` drains the queue automatically.
Each task is idempotent — re-running after a crash or operator-driven
restart is a no-op.

Usage::

    # Default: apply + verify + guide for the latest umbrella cohort
    python scripts/enqueue_evolution_chain.py \\
        build/enrichment_evolution/cohort_umbrellas_v3

    # Also kick off a fresh pipeline run after the chain completes
    python scripts/enqueue_evolution_chain.py \\
        build/enrichment_evolution/cohort_umbrellas_v3 --trigger-pipeline

    # Override the baseline run used by Phase 6 verify
    python scripts/enqueue_evolution_chain.py \\
        build/enrichment_evolution/cohort_umbrellas_v3 \\
        --baseline-run 7bbe4533

Operator surfaces (web terminal fallbacks, run from the App pod):

    python -m atelier.task_queue list
    python -m atelier.task_queue retry <task_id>
    python -m atelier.task_queue cancel <task_id>
    python -m atelier.task_queue run-once   # synchronous drain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from atelier.task_queue import enqueue  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("cohort_dir",
        help="Path to build/enrichment_evolution/cohort_<name>_v<N>/")
    parser.add_argument("--baseline-run", default=None,
        help="Run id (e.g. 7bbe4533) for Phase 6's baseline classifications")
    parser.add_argument("--acceptance", default=None,
        help="Optional acceptance JSON for the apply step")
    parser.add_argument("--trigger-pipeline", action="store_true",
        help="Append a trigger_pipeline_run task at the end of the chain")
    parser.add_argument("--source-id", default=None,
        help="Source id for the pipeline trigger (defaults to last "
             "user-selected from FSM runs)")
    args = parser.parse_args()

    cohort_dir = Path(args.cohort_dir)
    if not cohort_dir.is_dir():
        sys.exit(f"cohort_dir not found: {cohort_dir}")

    chain = []

    # 1. Apply enrichment transforms (Phase 5)
    apply_params = {"cohort_dir": str(cohort_dir)}
    if args.acceptance:
        apply_params["acceptance"] = args.acceptance
    apply_id = enqueue(
        "apply_enrichment_transforms", apply_params,
        idempotency_summary=f"apply: {cohort_dir.name}",
    )
    chain.append(("apply", apply_id))

    # 2. Verify transform apply (Phase 6) — depends on apply
    verify_params = {"cohort_dir": str(cohort_dir)}
    if args.baseline_run:
        verify_params["baseline_run"] = args.baseline_run
    verify_id = enqueue(
        "verify_transform_apply", verify_params,
        depends_on=[apply_id],
        idempotency_summary=f"verify: {cohort_dir.name}",
    )
    chain.append(("verify", verify_id))

    # 3. Render change management guide (Phase 7) — depends on verify
    guide_params = {"cohort_dir": str(cohort_dir)}
    guide_id = enqueue(
        "render_change_management_guide", guide_params,
        depends_on=[verify_id],
        idempotency_summary=f"guide: {cohort_dir.name}",
    )
    chain.append(("guide", guide_id))

    # 4. (Optional) trigger a fresh pipeline run against the new collection
    if args.trigger_pipeline:
        trigger_params = {}
        if args.source_id:
            trigger_params["source_id"] = args.source_id
        trigger_id = enqueue(
            "trigger_pipeline_run", trigger_params,
            depends_on=[guide_id],
            idempotency_summary=f"trigger: pipeline after {cohort_dir.name}",
        )
        chain.append(("trigger", trigger_id))

    print(f"\nEnqueued evolution chain ({len(chain)} tasks) for "
          f"{cohort_dir.name}:")
    for label, tid in chain:
        print(f"  {label:8}  {tid}")
    print(f"\nQueue dir: build/data/task_queue/pending/")
    print(f"\nOn the next App pod boot, the gateway lifespan will drain "
          f"these automatically.")
    print(f"To inspect from inside the pod:  python -m atelier.task_queue list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
