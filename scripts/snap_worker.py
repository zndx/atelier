# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Worker backends for the snapshot classifier.

Defines a narrow ``WorkerBackend`` interface and ships one implementation
(``ClaudeAgentWorker``) that runs a single ``claude -p`` subprocess per
table against Bedrock Sonnet.  A Cerebras-SDK backend (non-Claude model,
OpenAI-compatible endpoint) is out of scope for this attempt but the
interface is deliberately shaped so it can drop in later without
orchestrator changes.

The orchestrator calls ``run_table(task)`` which returns a ``WorkerResult``
describing what happened; the orchestrator decides whether to retry.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_TEMPLATE = REPO_ROOT / "scripts" / "snap_subagent_prompt.md"


@dataclass
class WorkerTask:
    """One unit of work handed to a backend."""

    table: str
    input_path: Path
    output_path: Path
    vocabulary_path: Path
    expected_columns: int
    attempt: int = 1
    timeout_s: int = 900  # 15 min default per table


@dataclass
class WorkerResult:
    table: str
    attempt: int
    status: str  # "ok" | "truncated" | "timeout" | "error"
    classified_columns: int
    expected_columns: int
    elapsed_s: float
    exit_code: Optional[int] = None
    error: Optional[str] = None
    raw_stdout_tail: Optional[str] = None
    raw_stderr_tail: Optional[str] = None
    usage: dict = field(default_factory=dict)


class WorkerBackend(Protocol):
    async def run_table(self, task: WorkerTask) -> WorkerResult: ...


# ---------------------------------------------------------------------------
# Claude Agent SDK backend (Bedrock)
# ---------------------------------------------------------------------------


def _bedrock_env(model_arn: str) -> dict:
    """Env overlay for a Bedrock-routed ``claude -p`` subprocess."""
    env = os.environ.copy()
    env["CLAUDE_CODE_USE_BEDROCK"] = "1"
    env["ANTHROPIC_MODEL"] = model_arn
    # Sub-model dispatches (tool search, etc.) also need a Bedrock target.
    env.setdefault("ANTHROPIC_DEFAULT_SONNET_MODEL", model_arn)
    env.setdefault("ANTHROPIC_DEFAULT_HAIKU_MODEL", os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL", model_arn))
    # Experimental betas dispatch to direct API — disable on Bedrock path.
    env.setdefault("CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS", "1")
    env.setdefault("ENABLE_TOOL_SEARCH", "false")
    # AWS creds must already be in the parent env (from atelier.env sourcing).
    return env


def _count_valid_jsonl(path: Path) -> tuple[int, bool]:
    """Return (record_count_excluding_sentinel, has_sentinel)."""
    if not path.exists():
        return 0, False
    count = 0
    has_sentinel = False
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("__done__"):
            has_sentinel = True
            continue
        if "predicted_code" in obj and "column" in obj:
            count += 1
    return count, has_sentinel


def _build_prompt(task: WorkerTask) -> str:
    template = PROMPT_TEMPLATE.read_text()
    return template.format(
        input_path=str(task.input_path),
        vocabulary_path=str(task.vocabulary_path),
        output_path=str(task.output_path),
    )


@dataclass
class ClaudeAgentWorker:
    """Runs one ``claude -p`` subprocess per table, routed through Bedrock."""

    model_arn: str
    workdir: Path = REPO_ROOT
    max_budget_usd: float = 5.0  # per-table safety cap
    effort: str = "medium"

    async def run_table(self, task: WorkerTask) -> WorkerResult:
        prompt = _build_prompt(task)
        env = _bedrock_env(self.model_arn)

        cmd = [
            "claude",
            "-p",
            "--bare",  # strip hooks/attribution/auto-memory for deterministic runs
            "--model", self.model_arn,
            "--permission-mode", "acceptEdits",
            "--tools", "Read,Write",
            "--output-format", "json",
            "--effort", self.effort,
            "--max-budget-usd", str(self.max_budget_usd),
            "--no-session-persistence",
            "--add-dir", str(self.workdir),
            prompt,
        ]

        t0 = time.time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=str(self.workdir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=task.timeout_s)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = time.time() - t0
                classified, _ = _count_valid_jsonl(task.output_path)
                return WorkerResult(
                    table=task.table, attempt=task.attempt, status="timeout",
                    classified_columns=classified, expected_columns=task.expected_columns,
                    elapsed_s=elapsed, error=f"exceeded {task.timeout_s}s",
                )
        except FileNotFoundError:
            return WorkerResult(
                table=task.table, attempt=task.attempt, status="error",
                classified_columns=0, expected_columns=task.expected_columns,
                elapsed_s=time.time() - t0, error="claude CLI not found on PATH",
            )

        elapsed = time.time() - t0
        stdout_txt = stdout.decode("utf-8", errors="replace") if stdout else ""
        stderr_txt = stderr.decode("utf-8", errors="replace") if stderr else ""
        classified, sentinel = _count_valid_jsonl(task.output_path)

        usage = {}
        # Agent SDK json output includes metadata at end — best-effort parse.
        try:
            last_brace = stdout_txt.rfind("{")
            if last_brace >= 0:
                tail = json.loads(stdout_txt[last_brace:])
                usage = tail.get("usage", {}) or {}
        except Exception:
            pass

        if proc.returncode != 0:
            return WorkerResult(
                table=task.table, attempt=task.attempt, status="error",
                classified_columns=classified, expected_columns=task.expected_columns,
                elapsed_s=elapsed, exit_code=proc.returncode,
                error=f"claude exit {proc.returncode}",
                raw_stdout_tail=stdout_txt[-400:], raw_stderr_tail=stderr_txt[-400:],
                usage=usage,
            )

        if classified < task.expected_columns or not sentinel:
            return WorkerResult(
                table=task.table, attempt=task.attempt, status="truncated",
                classified_columns=classified, expected_columns=task.expected_columns,
                elapsed_s=elapsed, exit_code=proc.returncode,
                error=f"got {classified}/{task.expected_columns} cols, sentinel={sentinel}",
                raw_stdout_tail=stdout_txt[-400:], raw_stderr_tail=stderr_txt[-400:],
                usage=usage,
            )

        return WorkerResult(
            table=task.table, attempt=task.attempt, status="ok",
            classified_columns=classified, expected_columns=task.expected_columns,
            elapsed_s=elapsed, exit_code=proc.returncode,
            raw_stdout_tail=stdout_txt[-400:], usage=usage,
        )
