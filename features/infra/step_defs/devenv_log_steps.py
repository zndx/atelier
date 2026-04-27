# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for devenv stack log-health scenarios.

Reads process-compose state (via CLI) and the captured process-compose
log from ``.devenv/state/process-compose/process-compose.log``.  Parses
JSONL events and surfaces Python tracebacks + known-bad signatures.
"""

from __future__ import annotations

import collections
import json
import re
import subprocess
from pathlib import Path

import time

from behave import given, when, then


_LOG_PATH = (
    Path(__file__).resolve().parents[3]
    / ".devenv" / "state" / "process-compose" / "process-compose.log"
)


# ── Process state ──────────────────────────────────────────────────


@when("I query process-compose for the current process states")
def step_query_processes(context):
    r = subprocess.run(
        ["process-compose", "process", "list", "-o", "json"],
        capture_output=True, text=True, timeout=15,
    )
    assert r.returncode == 0, (
        f"process-compose list failed: exit={r.returncode}, stderr={r.stderr!r}"
    )
    context.processes = {p["name"]: p for p in json.loads(r.stdout)}


@then('each of "{names}" is reachable')
def step_reachable(context, names):
    expected = [n.strip() for n in names.split(",") if n.strip()]
    context.expected_names = expected
    missing = [n for n in expected if n not in context.processes]
    assert not missing, (
        f"process-compose doesn't know about: {missing}. "
        f"Known: {sorted(context.processes)}"
    )


@then("every expected process is currently running")
def step_all_running(context):
    # Only enforce the invariant on processes we explicitly know about
    # (the earlier step populates context.expected_names).  process-compose
    # retains the ``exit_code`` from the most recent exit even on a
    # currently-running replica, so the authoritative signal is
    # ``is_running`` plus ``status == "Running"`` — not exit_code.
    names = getattr(context, "expected_names", list(context.processes))
    unhealthy = []
    for name in names:
        p = context.processes.get(name)
        if not p:
            continue
        if p.get("status") != "Running" or not p.get("is_running"):
            unhealthy.append((name, p.get("status"), p.get("is_running"), p.get("exit_code")))
    assert not unhealthy, (
        "processes not currently running:\n  "
        + "\n  ".join(
            f"{n} status={s} is_running={r} last_exit={e}"
            for n, s, r, e in unhealthy
        )
    )


# ── Log scanning ───────────────────────────────────────────────────


@given("I mark the current position of the process-compose log")
def step_mark_log(context):
    assert _LOG_PATH.exists(), f"log not found: {_LOG_PATH}"
    # Byte offset at "now" — any bytes written after this are new output.
    # Reading the incremental slice avoids being confounded by historical
    # crashes already on disk.
    context.log_mark = _LOG_PATH.stat().st_size


@when("I wait {seconds:d} seconds for the stack to emit heartbeats")
def step_wait(context, seconds):
    time.sleep(seconds)
    with open(_LOG_PATH, "rb") as f:
        f.seek(context.log_mark)
        fresh = f.read().decode("utf-8", errors="replace")
    context.log_lines = fresh.splitlines()


def _extract_messages(lines: list[str]) -> list[tuple[str, str]]:
    """Return (process, message) for each parseable JSON line."""
    out: list[tuple[str, str]] = []
    for raw in lines:
        try:
            e = json.loads(raw)
        except json.JSONDecodeError:
            continue
        out.append((e.get("process", "?"), str(e.get("message", ""))))
    return out


def _find_hits(context, pattern: re.Pattern) -> list[tuple[str, str]]:
    return [
        (proc, msg) for proc, msg in _extract_messages(context.log_lines)
        if pattern.search(msg)
    ]


@then('the log lines since the mark contain no "{needle}"')
def step_no_needle(context, needle):
    pat = re.compile(re.escape(needle))
    hits = _find_hits(context, pat)
    _fail_on_hits(hits, f"{needle!r} occurrences")


def _fail_on_hits(hits: list[tuple[str, str]], label: str) -> None:
    if not hits:
        return
    # Summarise by (process, first 80 chars) for readable failure output.
    counts = collections.Counter((proc, msg[:80]) for proc, msg in hits)
    summary = "\n  ".join(
        f"{n}x [{proc}] {msg}" for (proc, msg), n in counts.most_common(8)
    )
    raise AssertionError(
        f"Found {len(hits)} {label} in recent log:\n  {summary}"
    )
