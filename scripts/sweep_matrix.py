#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""N-axis Cartesian sweep over arbitrary HOCON parameters.

Generalizes ``sweep_bel_threshold.py`` from "vary one threshold" to
"vary K parameters across an N-dimensional grid."  Each axis maps a
HOCON path (e.g. ``classify.cautious_review.bel_threshold``) to its
canonical env-var name (the one HOCON's ``${?VAR}`` substitution reads)
so the subprocess inherits the right override without us touching the
HOCON file between runs.

Why a new script instead of widening sweep_bel_threshold.py
-----------------------------------------------------------

The single-axis script produced every historical manifest you have on
disk; backwards compatibility matters for the score_sweep.py path and
for the ``project_cat_a_baseline`` memory note that references that
shape.  The new matrix script:

  * Writes manifests with a different shape (``axes`` + per-run
    ``params`` dict) so score_sweep.py knows to read them differently.
  * Supports an optional ``--name`` for the manifest stem (defaults
    to ``matrix-{axes_joined}``).
  * Keeps a single-axis invocation equivalent: ``--axis
    classify.cautious_review.bel_threshold=0.55,0.65,0.75,0.80`` is
    structurally the same as a one-axis sweep.

The harm-first framing changes WHAT we sweep
--------------------------------------------

"First do no harm" reframes the objective.  Strict accuracy alone
isn't the loss function — preventable harm (wrong_subtree,
hallucinated_annotation) outweighs the gain from a small strict-pct
bump.  This script doesn't compute the composite — that lives in
``score_sweep.py`` so the same accuracy function can rank both
matrix and single-axis sweeps — but it DOES persist the axis
parameter values per run so score_sweep can index the results space.

Usage
-----

::

    # 2D bel × gap grid
    uv run python scripts/sweep_matrix.py \\
        --source-id hive-poc/reference_corpus \\
        --axis classify.cautious_review.bel_threshold=0.55,0.65,0.75,0.80 \\
        --axis classify.bootstrap.gap_threshold=0.12,0.18,0.24 \\
        --name bel_x_gap \\
        --continue-on-failure

    # 3D bel × gap × fusion
    uv run python scripts/sweep_matrix.py \\
        --source-id hive-poc/reference_corpus \\
        --axis classify.cautious_review.bel_threshold=0.60,0.70,0.80 \\
        --axis classify.bootstrap.gap_threshold=0.15,0.21 \\
        --axis classify.fusion_strategy=dempster,yager \\
        --name bel_x_gap_x_fusion \\
        --continue-on-failure

Once the manifest lands::

    uv run python scripts/score_sweep.py \\
        --manifest build/sweeps/bel_x_gap-2026-05-16T...Z.json \\
        --agent-mediated build/data/agent_mediated/agent_mediated.json \\
        --harm-aware
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ── HOCON path → env-var map ─────────────────────────────────────────
#
# Only paths with an existing ``${?VAR}`` binding in config/base.conf
# are listed here — without the HOCON binding the env override is
# silently ignored.  Verified against the conf as of 2026-05-16.
# To add a new axis: first add ``= ${?VAR}`` to base.conf, then
# extend this map.  An assertion in main() refuses unknown paths.
HOCON_TO_ENV: dict[str, str] = {
    # Cautious-review post-bootstrap override (most-watched knob)
    "classify.cautious_review.bel_threshold": "ATELIER_CLASSIFY_CAUTIOUS_REVIEW_BEL_THRESHOLD",
    # Bootstrap convergence — all bindable per config/base.conf:301-384
    "classify.bootstrap.gap_threshold": "ATELIER_BOOTSTRAP_GAP_THRESHOLD",
    "classify.bootstrap.bel_floor": "ATELIER_BOOTSTRAP_BEL_FLOOR",
    "classify.bootstrap.k_threshold": "ATELIER_BOOTSTRAP_K_THRESHOLD",
    "classify.bootstrap.clarity_target": "ATELIER_BOOTSTRAP_CLARITY_TARGET",
    "classify.bootstrap.indep_revisit_mass_threshold": "ATELIER_BOOTSTRAP_INDEP_REVISIT_MASS_THRESHOLD",
    # DST fusion strategy (dempster | yager)
    "classify.fusion_strategy": "ATELIER_FUSION_STRATEGY",
    # NOTE: classify.discount.* and classify.monte_carlo.* paths are
    # not yet bindable via env — adding them requires a base.conf
    # change first.  Listed here as a TODO so we don't silently
    # accept their sweep specs.  See base.conf:475-488 (discounts)
    # and base.conf monte_carlo block.
}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_axis(spec: str) -> tuple[str, list[str]]:
    """Parse ``hocon.path=val1,val2,val3`` into (path, [str values]).

    Values stay as strings here — the subprocess inherits them as env
    vars, and HOCON does the typed coercion on the other side.
    """
    if "=" not in spec:
        raise ValueError(
            f"--axis {spec!r}: missing '=' (expected 'hocon.path=v1,v2,...')"
        )
    path, values = spec.split("=", 1)
    path = path.strip()
    if path not in HOCON_TO_ENV:
        raise ValueError(
            f"--axis {path!r}: not in HOCON_TO_ENV.  Add it to "
            f"scripts/sweep_matrix.py:HOCON_TO_ENV after verifying "
            f"config/base.conf has the corresponding ${{?ENV}} binding."
        )
    parts = [v.strip() for v in values.split(",") if v.strip()]
    if not parts:
        raise ValueError(f"--axis {spec!r}: empty value list")
    return path, parts


def _run_one(
    params: dict[str, str],
    *,
    connection_name: str | None,
    database: str | None,
    tables_limit: int | None,
    source_id: str | None,
    python: str,
) -> dict:
    """Invoke a single pipeline run with the given param overrides."""
    env = os.environ.copy()
    for hocon_path, value in params.items():
        env_var = HOCON_TO_ENV[hocon_path]
        env[env_var] = value

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
        "params": dict(params),
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


def _short_path(path: str) -> str:
    """Render a HOCON path as a short slug for manifest filenames."""
    return path.split(".")[-1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--axis",
        action="append",
        required=True,
        help=(
            "Sweep axis: 'hocon.path=v1,v2,v3'.  Repeatable; the "
            "Cartesian product across all --axis specs is the sweep "
            "grid.  HOCON path must be in HOCON_TO_ENV (see source)."
        ),
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
        "--name",
        default=None,
        help=(
            "Manifest filename stem (default: 'matrix-<axes-joined>'). "
            "Final shape: '{name}-{utc-iso}.json'."
        ),
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the grid plan + env overrides without invoking runs. "
            "Useful for confirming a sweep before kicking off."
        ),
    )
    args = parser.parse_args()

    # Parse axes.  Insertion-order preserves the operator's intent; the
    # Cartesian product respects it for deterministic run ordering.
    axes: list[tuple[str, list[str]]] = []
    for spec in args.axis:
        try:
            axes.append(_parse_axis(spec))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    grid_size = 1
    for _, vals in axes:
        grid_size *= len(vals)

    name_stem = args.name or (
        "matrix-" + "_x_".join(_short_path(p) for p, _ in axes)
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"{name_stem}-{_utc_iso()}.json"

    manifest = {
        "schema_version": 2,  # 1 = sweep_bel_threshold.py; 2 = matrix
        "kind": "matrix",
        "started_at": _utc_iso(),
        "axes": [
            {"path": p, "env_var": HOCON_TO_ENV[p], "values": v}
            for p, v in axes
        ],
        "grid_size": grid_size,
        "runs": [],
        "config": {
            "connection_name": args.connection_name,
            "database": args.database,
            "tables_limit": args.tables_limit,
            "source_id": args.source_id,
        },
    }

    print(f"Sweep -> {manifest_path}", file=sys.stderr)
    print(f"Grid size: {grid_size} runs across {len(axes)} axis/axes", file=sys.stderr)
    for p, vals in axes:
        print(f"  axis {p}: {vals}", file=sys.stderr)

    # Enumerate the Cartesian product into (param dict) tuples.
    value_lists = [vals for _, vals in axes]
    paths = [p for p, _ in axes]
    grid: list[dict[str, str]] = [
        dict(zip(paths, combo)) for combo in itertools.product(*value_lists)
    ]

    if args.dry_run:
        print(f"\n--dry-run: would invoke {grid_size} runs:", file=sys.stderr)
        for i, params in enumerate(grid, 1):
            envs = ", ".join(
                f"{HOCON_TO_ENV[p]}={v}" for p, v in params.items()
            )
            print(f"  run {i:>2}: {envs}", file=sys.stderr)
        return 0

    for i, params in enumerate(grid, 1):
        param_str = " ".join(f"{_short_path(p)}={v}" for p, v in params.items())
        print(f"\n=== Run {i}/{grid_size}: {param_str} ===", file=sys.stderr)
        summary = _run_one(
            params,
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
                f"Aborting after failed run {i}.  Pass "
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
