#!/usr/bin/env python
"""Sweep ``classify.cautious_review.bel_threshold`` across a grid.

.. warning:: **OBSOLETE — cautious review is proven harmful.**

   Run ce4f3777 (2026-05-20) demonstrated that cautious review destroys
   accuracy at every threshold tested: reroute 76.1% miss rate, backoff
   78.8% miss rate, net −13.6pp vs LLM-only.  The default config now
   disables cautious review entirely (enabled=false, bel_threshold=0.0).

   This sweep script is retained only for the unlikely case that someone
   re-implements cautious review from scratch and needs to re-validate.
   Running it against the current (disabled) config will produce
   identical results across all threshold values since review never fires.

Historical hypothesis: cautious-review is over-correcting at the current
default (0.80).  **Confirmed and worse**: it over-corrects at *every*
threshold.

What this does
--------------

For each value in the threshold grid:

  1. Sets ``ATELIER_CLASSIFY_CAUTIOUS_REVIEW_BEL_THRESHOLD`` in the
     subprocess environment.  The HOCON binding at
     ``config/base.conf:406`` makes this an env-driven override.
  2. Invokes ``run_classification_pipeline`` in a *fresh subprocess*
     so the cached ``AtelierConfig`` doesn't carry state between runs.
  3. Captures the resulting ``run_id`` and persists it.

Output: ``build/sweeps/bel_threshold-<utc-iso>.json`` with shape::

  {
    "started_at": "2026-05-15T01:23:45Z",
    "thresholds": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    "runs": [
      {"threshold": 0.55, "run_id": "...", "status": "ok", "duration_s": 612.4},
      ...
    ],
    "config": {
      "connection_name": "hive-poc",
      "database": "reference_corpus",
      "tables_limit": null,
      "source_id": null
    }
  }

Then feed the manifest to ``scripts/score_sweep.py`` to compare against
agent-mediated reference.

Dataset pinning
---------------

This script does NOT pin the table set by itself.  Either:

  * Use a branch that has ``classify.table_exclude_patterns`` and set
    it in ``config/base.conf`` before invoking, OR
  * Let ``score_sweep.py`` filter to agent-mediated-reference-known columns
    (recommended; works regardless of branch state).

Usage
-----

::

  uv run python scripts/sweep_bel_threshold.py \\
      --connection-name hive-poc \\
      --database reference_corpus \\
      --thresholds 0.55,0.60,0.65,0.70,0.75,0.80

  # then:
  uv run python scripts/score_sweep.py \\
      --manifest build/sweeps/bel_threshold-2026-05-15T01:23:45Z.json \\
      --agent-mediated path/to/agent_mediated.json

Cost
----

Each run is a full classify (LLM sweeps, bootstrap, fusion,
cautious-review).  Budget ~10 min/run on the OLD 20-table substrate
on a GPU host, longer on CPU.  Six thresholds => ~1h of compute.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_one(
    threshold: float,
    *,
    connection_name: str | None,
    database: str | None,
    tables_limit: int | None,
    source_id: str | None,
    python: str,
) -> dict:
    """Invoke a single pipeline run in a subprocess and return its summary."""
    env = os.environ.copy()
    env["ATELIER_CLASSIFY_CAUTIOUS_REVIEW_BEL_THRESHOLD"] = f"{threshold:.4f}"

    # Inline runner: stays in this file so the sweep tool is self-contained.
    # Writes a single-line JSON summary to stdout that the parent parses.
    inline = (
        "import json, sys\n"
        "from atelier.classify import get_fsm\n"
        "from atelier.classify.pipeline import run_classification_pipeline\n"
        "from atelier.config import load_config\n"
        "cfg = load_config()\n"
        "fsm = get_fsm()\n"
        "kwargs = {}\n"
        f"src = {source_id!r}\n"
        f"cn  = {connection_name!r}\n"
        f"db  = {database!r}\n"
        f"tl  = {tables_limit!r}\n"
        "if src is not None: kwargs['source_id'] = src\n"
        "if cn  is not None: kwargs['connection_name'] = cn\n"
        "if db  is not None: kwargs['database'] = db\n"
        "if tl  is not None: kwargs['tables_limit'] = tl\n"
        "result = run_classification_pipeline(cfg, fsm, **kwargs)\n"
        "run_id = result.get('run_id') if isinstance(result, dict) else None\n"
        "phase = None\n"
        "try:\n"
        "    status = fsm.get_status(run_id) if run_id else None\n"
        "    if status is not None:\n"
        "        phase = status.state.value if hasattr(status.state, 'value') else str(status.state)\n"
        "except Exception as _e:\n"
        "    phase = f'(phase-lookup-failed: {type(_e).__name__})'\n"
        "sys.stdout.write('\\n__SWEEP_RESULT__' + json.dumps({\n"
        "    'run_id': run_id,\n"
        "    'phase': phase,\n"
        "}) + '__SWEEP_RESULT__\\n')\n"
    )

    started = time.monotonic()
    proc = subprocess.run(
        [python, "-c", inline],
        env=env,
        capture_output=True,
        text=True,
    )
    duration_s = time.monotonic() - started

    summary: dict = {
        "threshold": threshold,
        "duration_s": round(duration_s, 1),
        "returncode": proc.returncode,
    }

    if proc.returncode != 0:
        summary["status"] = "failed"
        summary["stderr_tail"] = proc.stderr[-2000:]
        return summary

    marker = "__SWEEP_RESULT__"
    if marker in proc.stdout:
        payload_str = proc.stdout.split(marker)[1]
        try:
            payload = json.loads(payload_str)
            summary["run_id"] = payload.get("run_id")
            summary["final_phase"] = payload.get("phase")
            summary["status"] = "ok" if payload.get("run_id") else "no_run_id"
        except json.JSONDecodeError as exc:
            summary["status"] = "parse_error"
            summary["error"] = str(exc)
    else:
        summary["status"] = "no_marker"
        summary["stdout_tail"] = proc.stdout[-1000:]

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{t:g}" for t in DEFAULT_THRESHOLDS),
        help="Comma-separated grid (default: 0.55,0.60,0.65,0.70,0.75,0.80).",
    )
    parser.add_argument("--connection-name", default=None)
    parser.add_argument("--database", default=None)
    parser.add_argument("--tables-limit", type=int, default=None)
    parser.add_argument(
        "--source-id",
        default=None,
        help="When set, takes priority over connection-name/database.",
    )
    parser.add_argument(
        "--output-dir",
        default="build/sweeps",
        help="Directory for the sweep manifest (created if absent).",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter for the subprocess runs.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Carry on after a failed run instead of aborting.",
    )
    args = parser.parse_args()

    thresholds = sorted({float(t) for t in args.thresholds.split(",") if t.strip()})
    if not thresholds:
        print("No thresholds parsed from --thresholds", file=sys.stderr)
        return 2

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"bel_threshold-{_utc_iso()}.json"

    manifest = {
        "started_at": _utc_iso(),
        "thresholds": thresholds,
        "runs": [],
        "config": {
            "connection_name": args.connection_name,
            "database": args.database,
            "tables_limit": args.tables_limit,
            "source_id": args.source_id,
        },
    }

    print(f"Sweep -> {manifest_path}", file=sys.stderr)
    print(f"Thresholds: {thresholds}", file=sys.stderr)

    for t in thresholds:
        print(f"\n=== Running bel_threshold={t:.2f} ===", file=sys.stderr)
        summary = _run_one(
            t,
            connection_name=args.connection_name,
            database=args.database,
            tables_limit=args.tables_limit,
            source_id=args.source_id,
            python=args.python,
        )
        manifest["runs"].append(summary)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(
            f"  status={summary['status']} run_id={summary.get('run_id')} "
            f"duration={summary['duration_s']}s",
            file=sys.stderr,
        )
        if summary["status"] != "ok" and not args.continue_on_failure:
            print(
                f"Aborting after failed run at threshold={t}.  Pass "
                f"--continue-on-failure to override.",
                file=sys.stderr,
            )
            manifest["aborted_at"] = _utc_iso()
            manifest_path.write_text(json.dumps(manifest, indent=2))
            return 1

    manifest["finished_at"] = _utc_iso()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone. Manifest: {manifest_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
