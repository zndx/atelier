"""Snapshot classifier orchestrator.

Fans out N concurrent per-table subagents against Bedrock Sonnet.
Maintains ``manifest.json`` as the single source of truth for what's
pending / in-flight / complete / failed, so the run is resumable.

Basic flow::

    scripts/snap_classify_orchestrator.py \\
        --concurrency 4 \\
        --model "$ANTHROPIC_SUBAGENT_MODEL" \\
        --only ab_test_results,customers  # optional smoke-test scope

Progress / failures are printed to stdout and also appended to
``build/snapshots/hive-poc__synth/orchestrator.log``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from snap_worker import ClaudeAgentWorker, WorkerBackend, WorkerTask, WorkerResult  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SNAP_ROOT = REPO_ROOT / "build" / "snapshots" / "hive-poc__synth"
MANIFEST = SNAP_ROOT / "manifest.json"
LOG_PATH = SNAP_ROOT / "orchestrator.log"

DEFAULT_MODEL = os.environ.get(
    "ANTHROPIC_SUBAGENT_MODEL",
    "arn:aws:bedrock:us-east-1:440464140575:inference-profile/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
)


# ---------------------------------------------------------------------------
# Manifest helpers (asynchronous-safe via single-writer orchestrator loop)
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def save_manifest(m: dict) -> None:
    m["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(m, indent=2, default=str) + "\n")
    tmp.replace(MANIFEST)


def log_event(msg: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

async def _run_one(
    backend: WorkerBackend, table: str, entry: dict, manifest: dict, sem: asyncio.Semaphore
) -> WorkerResult:
    async with sem:
        attempt = int(entry.get("attempts", 0)) + 1
        task = WorkerTask(
            table=table,
            input_path=SNAP_ROOT / "input" / f"{table}.json",
            output_path=SNAP_ROOT / "output" / f"{table}.jsonl",
            vocabulary_path=SNAP_ROOT / "vocabulary.json",
            expected_columns=int(entry.get("columns", 0)),
            attempt=attempt,
        )

        # Transition to in-flight
        entry["status"] = "in_flight"
        entry["attempts"] = attempt
        entry["last_started"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        save_manifest(manifest)
        log_event(f"[start] {table} (attempt {attempt}, {task.expected_columns} cols)")

        result = await backend.run_table(task)

        # Reconcile
        entry["status"] = "complete" if result.status == "ok" else result.status
        entry["last_result"] = {
            "status": result.status,
            "classified": result.classified_columns,
            "expected": result.expected_columns,
            "elapsed_s": round(result.elapsed_s, 1),
            "error": result.error,
            "usage": result.usage,
        }
        save_manifest(manifest)

        icon = {"ok": "[ok ]", "truncated": "[trn]", "timeout": "[tmo]", "error": "[err]"}.get(result.status, "[?]")
        log_event(
            f"{icon} {table} attempt={attempt} "
            f"{result.classified_columns}/{result.expected_columns} cols "
            f"{result.elapsed_s:.0f}s "
            f"{('err=' + (result.error or '')[:120]) if result.error else ''}"
        )
        return result


async def run_orchestrator(args: argparse.Namespace) -> int:
    manifest = load_manifest()
    all_tables = manifest["tables"]

    if args.only:
        wanted = {t.strip() for t in args.only.split(",") if t.strip()}
        candidates = [(t, e) for t, e in all_tables.items() if t in wanted]
    else:
        # Skip tables already complete; retry truncated/timeout/error unless --no-retry
        retriable = {"pending", "truncated", "timeout", "error", "in_flight"}
        if args.no_retry:
            retriable = {"pending"}
        candidates = [(t, e) for t, e in all_tables.items() if e.get("status", "pending") in retriable]

    if args.max_attempts:
        candidates = [(t, e) for t, e in candidates if int(e.get("attempts", 0)) < args.max_attempts]

    if args.limit:
        candidates = candidates[: args.limit]

    log_event(
        f"[plan] concurrency={args.concurrency} model={args.model.split('/')[-1]} "
        f"tables={len(candidates)}"
    )
    if not candidates:
        log_event("[plan] nothing to do — manifest shows no retriable tables")
        return 0

    backend = ClaudeAgentWorker(model_arn=args.model, effort=args.effort, max_budget_usd=args.budget)
    sem = asyncio.Semaphore(args.concurrency)

    # Wire Ctrl-C so we write manifest state before exiting
    loop = asyncio.get_event_loop()
    stopping = asyncio.Event()

    def _handle_stop(*_):
        if not stopping.is_set():
            log_event("[signal] stop requested — will finish in-flight tasks then exit")
            stopping.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, _handle_stop)
        except NotImplementedError:
            pass  # Windows

    t0 = time.time()
    # Launch all tasks up-front; semaphore caps true parallelism.
    coros = [_run_one(backend, t, e, manifest, sem) for t, e in candidates]
    results: list[WorkerResult] = []
    for fut in asyncio.as_completed(coros):
        if stopping.is_set():
            break
        r = await fut
        results.append(r)

    elapsed = time.time() - t0
    ok = sum(1 for r in results if r.status == "ok")
    bad = len(results) - ok
    log_event(f"[done] ok={ok} bad={bad} total={len(results)} elapsed={elapsed:.0f}s")
    return 0 if bad == 0 else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", default="", help="Comma list of table names")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--no-retry", action="store_true", help="Only pick pending tables; skip failed")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--budget", type=float, default=5.0, help="USD cap per subagent")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"manifest not found: {MANIFEST} — run snap_extract_inputs.py first", file=sys.stderr)
        return 1

    SNAP_ROOT.mkdir(parents=True, exist_ok=True)
    return asyncio.run(run_orchestrator(args))


if __name__ == "__main__":
    raise SystemExit(main())
