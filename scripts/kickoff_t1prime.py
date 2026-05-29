#!/usr/bin/env python
"""Kick off T1' bracket from the AMP pod's running gateway.

Use case: this session's launcher dies (session timeout, kill, etc.) — operator
runs this script from the Web Terminal Agent inside the AMP pod to either
start the run fresh OR monitor an in-progress run.

Why HTTP instead of importing fsm_start
---------------------------------------

In the session pod, our launcher (t1prime_launch.py) imported fsm_start
directly — the pipeline ran as a daemon thread inside the launcher process,
so if the launcher died the pipeline died with it.

In the AMP pod, the GATEWAY is the long-running process.  This script just
triggers /api/fsm/start over HTTP; the gateway daemon-threads the pipeline.
If this script exits (or the operator's terminal disconnects), the pipeline
keeps running.  Re-running the script monitors the in-flight run instead of
starting a new one.

Usage from the AMP pod's terminal
---------------------------------

    python scripts/kickoff_t1prime.py
    python scripts/kickoff_t1prime.py --source-id hive-poc/reference_corpus
    python scripts/kickoff_t1prime.py --no-overlay   # use whatever Settings has
    python scripts/kickoff_t1prime.py --monitor-only # don't try to start

Calibration overlay applied (Phase 1.5 operating point)
-------------------------------------------------------

See build/runs/calibration/resweep_b60d5a4e/new_operating_point.md.
The values are PATCHed via /api/settings (runtime overlay, no restart).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# Phase 1.5 operating point — match what the resweep landed on
CALIBRATION_OVERLAY = {
    "classify_mass_calibration_cosine_alpha": 0.5,
    "classify_mass_calibration_svm_alpha": 0.5,
    "classify_mass_calibration_catboost_alpha": 0.7,
    "classify_mass_calibration_llm_alpha": 0.1,
    "classify_cosine_union_focal_k": 3,
    "classify_cosine_union_focal_alpha": 0.45,
    "classify_svm_source": "registered",
}


def gateway_url() -> str:
    """Resolve the gateway URL.  AMP exports CDSW_APP_PORT=8090."""
    port = os.environ.get("CDSW_APP_PORT", "8090")
    return f"http://127.0.0.1:{port}"


def http(method: str, path: str, data: dict | None = None,
         timeout: int = 30) -> dict:
    """Minimal HTTP client — stdlib only so this works without `requests`."""
    url = f"{gateway_url()}{path}"
    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {body[:300]}") from e


def fmt_elapsed(start: float) -> str:
    el = int(time.time() - start)
    return f"+{el//60:>3}m{el%60:>02d}s"


def find_current_run() -> str | None:
    """Return run_id if an FSM run is in progress, else None."""
    try:
        state = http("GET", "/api/fsm/status", timeout=5)
    except Exception:
        return None
    cur_state = (state.get("state") or "").upper()
    run_id = state.get("id") or state.get("run_id")
    if cur_state and cur_state not in ("IDLE", "CONVERGED", "ERROR") and run_id:
        return run_id
    return None


def apply_overlay(overlay: dict) -> dict:
    """PATCH /api/settings with the Phase 1.5 calibration values.

    Falls back gracefully when the gateway predates the calibration keys —
    surfaces the missing keys instead of failing the whole run.
    """
    accepted: dict = {}
    rejected: dict = {}
    for key, val in overlay.items():
        try:
            resp = http("PATCH", "/api/settings", data={key: val}, timeout=10)
            accepted[key] = val
        except RuntimeError as e:
            rejected[key] = str(e)
    return {"accepted": accepted, "rejected": rejected}


def start_pipeline(source_id: str) -> str:
    """Trigger fsm_start; return the new run_id (poll briefly to discover it)."""
    result = http("POST", f"/api/fsm/start?source_id={source_id}", timeout=30)
    if not result.get("started"):
        raise RuntimeError(f"fsm_start failed: {result}")
    # run_id is created inside the daemon thread; poll for it
    for _ in range(30):
        time.sleep(2)
        state = http("GET", "/api/fsm/status", timeout=5)
        run_id = state.get("id") or state.get("run_id")
        cur_state = (state.get("state") or "").upper()
        if run_id and cur_state not in ("IDLE", "CONVERGED", "ERROR"):
            return run_id
    raise RuntimeError("pipeline started but no run_id within 60s")


def monitor(run_id: str, poll_interval: int = 30) -> str:
    """Block until terminal state.  Returns final state ('CONVERGED'/'ERROR')."""
    results_dir = Path(f"build/results/{run_id}")
    start = time.time()
    last_state = None
    last_progress = ""
    print(f"\n[{time.strftime('%H:%M:%S')}] monitoring run_id={run_id}", flush=True)
    print(f"  artifacts: {results_dir}/\n", flush=True)
    consecutive_poll_errors = 0
    while True:
        try:
            state = http("GET", "/api/fsm/status", timeout=10)
            cur = (state.get("state") or "?").upper()
            prog = json.dumps(state.get("progress") or {}, sort_keys=True)[:140]
            if cur != last_state or prog != last_progress:
                print(f"[{time.strftime('%H:%M:%S')}] {fmt_elapsed(start)}  "
                      f"{cur:25} {prog}", flush=True)
                last_state = cur
                last_progress = prog
            if cur in ("CONVERGED", "ERROR"):
                return cur
            consecutive_poll_errors = 0
        except Exception as e:
            consecutive_poll_errors += 1
            print(f"  poll error #{consecutive_poll_errors}: {e}", flush=True)
            if consecutive_poll_errors >= 10:
                print(f"  ✗ too many consecutive poll errors — gateway may be down")
                print(f"  → check build/results/{run_id}/fsm_state.json directly")
                return "POLL_FAILED"
        time.sleep(poll_interval)


def report(run_id: str, terminal_state: str) -> None:
    results_dir = Path(f"build/results/{run_id}")
    print(f"\n=== Run {run_id} → {terminal_state} ===")
    for f in ("scoring_summary.md", "evaluation_report.json",
              "overwatch.md", "classifications.json"):
        p = results_dir / f
        sz = p.stat().st_size if p.exists() else 0
        print(f"  {f}: {'✓ ' + str(sz) + ' bytes' if sz else '✗ MISSING'}")

    summary = results_dir / "scoring_summary.md"
    if summary.exists():
        print(f"\n=== {summary.name} (head) ===")
        for line in summary.read_text().splitlines()[:25]:
            print(f"  {line}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-id", default="hive-poc/reference_corpus",
                    help="Source for the pipeline run (default: reference_corpus)")
    ap.add_argument("--no-overlay", action="store_true",
                    help="Skip the Phase 1.5 calibration overlay PATCH "
                         "(use whatever Settings already has)")
    ap.add_argument("--monitor-only", action="store_true",
                    help="Don't start a new run; just monitor whatever is in progress")
    ap.add_argument("--poll-interval", type=int, default=30,
                    help="Seconds between FSM state polls (default 30)")
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] T1' AMP-pod launcher", flush=True)
    print(f"  gateway: {gateway_url()}")
    print(f"  source:  {args.source_id}")
    print()

    # Probe gateway
    try:
        health = http("GET", "/api/health", timeout=5)
        print(f"  gateway health: {health}")
    except Exception as e:
        print(f"  ✗ gateway not reachable at {gateway_url()}: {e}")
        print(f"    is the AMP pod's gateway running? check `python -m atelier.gateway` "
              f"or the AMP Application status in CAI Workspace.")
        return 1

    # Detect in-flight run
    existing = find_current_run()
    if existing:
        print(f"\n  ⚠ classification already in progress: run_id={existing}")
        print(f"  → monitoring existing run (overlay will NOT be applied "
              f"to the in-flight run)")
        run_id = existing
    elif args.monitor_only:
        print(f"\n  ✗ --monitor-only set, but no run in progress")
        return 1
    else:
        # Apply overlay
        if not args.no_overlay:
            print(f"\n[{time.strftime('%H:%M:%S')}] applying Phase 1.5 calibration overlay")
            result = apply_overlay(CALIBRATION_OVERLAY)
            print(f"  ✓ accepted: {sorted(result['accepted'].keys())}")
            if result["rejected"]:
                print(f"  ⚠ rejected: {sorted(result['rejected'].keys())}")
                print(f"    (gateway may predate Phase 2 wiring — check deployment)")
        else:
            print(f"\n[{time.strftime('%H:%M:%S')}] --no-overlay: using current Settings")

        # Start run
        print(f"\n[{time.strftime('%H:%M:%S')}] dispatching pipeline...")
        try:
            run_id = start_pipeline(args.source_id)
        except RuntimeError as e:
            print(f"  ✗ {e}")
            return 1
        print(f"  ✓ run_id: {run_id}")

    terminal = monitor(run_id, poll_interval=args.poll_interval)
    report(run_id, terminal)

    print(f"\nTo see the full report: cat build/results/{run_id}/scoring_summary.md")
    print(f"To compare vs 5ef4868c baseline:")
    print(f"  diff <(jq .per_category build/results/5ef4868c/evaluation_report.json) \\")
    print(f"       <(jq .per_category build/results/{run_id}/evaluation_report.json)")
    return 0 if terminal == "CONVERGED" else 1


if __name__ == "__main__":
    sys.exit(main())
