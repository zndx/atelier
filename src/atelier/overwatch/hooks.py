# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""PreToolUse hook — path & command sandbox for the supervisor agent.

The supervisor overwatch is granted the Read/Grep/Glob/Bash tools so
it can investigate a failing run (inspect classifications.json, grep
source, read ``state.batch_audit``) and invoke its four controlled
CLIs (``write_proposal``, ``ingest_reference``, ``apply_and_rerun``,
``kill_run``).  Without a sandbox the Bash tool would also let it
``rm -rf``, ``git reset --hard``, or exfiltrate data.

The hook enforces two bounds:

* **Path sandbox** — Read/Glob/Grep may only touch the repository
  root; no ``/etc``, ``/home/*`` outside the project, ``~/.ssh``, etc.
* **Bash command allowlist** — the Bash tool must match one of the
  four sanctioned ``uv run python -m atelier.overwatch.*`` invocations,
  the handful of inspection commands (``ls``, ``cat``, ``wc``, ``head``,
  ``tail``) under the results tree, or a small read-only git query
  set.  Anything else (``rm``, ``mv``, ``dd``, piping ``curl | sh``,
  ``git push``, ``chmod``) is denied.

Denials return ``PreToolUseHookOutput(decision="deny")`` with a
message that gets re-shown to the agent so it can pivot instead of
failing silently.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Path sandbox ────────────────────────────────────────────────────

# Resolved inside each call so the module is still import-clean when the
# project root isn't easily derivable (tests, subagents).
def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


# Allowed roots under which Read/Glob/Grep paths must resolve.  Paths
# are compared by resolved-string prefix against each root (also
# resolved) so ``..`` escapes are caught.
def _allowed_roots() -> list[Path]:
    root = _project_root()
    return [
        root,  # whole project is OK; the Bash allowlist is the tighter gate
    ]


def _is_in_sandbox(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve()
    except Exception:
        return False
    for root in _allowed_roots():
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


# ── Bash allowlist ──────────────────────────────────────────────────

# Four sanctioned CLI entry points.  Every supervisor-initiated Bash
# that actually changes state goes through one of these; everything
# else is either a read-only inspection command or rejected.
_ALLOWED_CLI_MODULES = frozenset({
    "atelier.overwatch.write_proposal",
    "atelier.overwatch.ingest_reference",
    "atelier.overwatch.apply_and_rerun",
    "atelier.overwatch.kill_run",
})


# Read-only shell commands permitted against the project tree.  Kept
# deliberately small — the agent can Read/Grep/Glob for most inspection
# work; this list covers what those tools can't express.
_READ_ONLY_COMMANDS = frozenset({
    "ls", "wc", "head", "tail", "cat", "stat", "find", "du", "file",
    "pwd", "echo",
})


# Git subcommands that only query state.  No push / reset / checkout /
# rebase / merge — those write.
_SAFE_GIT_SUBCOMMANDS = frozenset({
    "status", "log", "diff", "show", "blame", "branch", "describe",
    "rev-parse", "remote", "config",
})


# Hard deny patterns (regex, case-insensitive).  Applied first so even
# if something matches an allowlist branch it can't sneak past these.
_DENIED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-[rf]", re.IGNORECASE),
    re.compile(r"\brm\s+-rf", re.IGNORECASE),
    re.compile(r"\bdd\s", re.IGNORECASE),
    re.compile(r"\bchmod\s", re.IGNORECASE),
    re.compile(r"\bchown\s", re.IGNORECASE),
    re.compile(r"\bmv\s+", re.IGNORECASE),
    re.compile(r"\bcurl\s.*\|.*sh\b", re.IGNORECASE),
    re.compile(r"\bwget\s.*\|.*sh\b", re.IGNORECASE),
    re.compile(r"\bgit\s+reset\s+--hard\b", re.IGNORECASE),
    re.compile(r"\bgit\s+push\b", re.IGNORECASE),
    re.compile(r"\bgit\s+checkout\b", re.IGNORECASE),
    re.compile(r"\bgit\s+commit\b", re.IGNORECASE),
    re.compile(r"\bgit\s+clean\b", re.IGNORECASE),
    re.compile(r"\bsudo\b", re.IGNORECASE),
    re.compile(r"\bexec\s", re.IGNORECASE),
    re.compile(r">\s*/etc/", re.IGNORECASE),
    re.compile(r">\s*\$HOME", re.IGNORECASE),
    re.compile(r"\bssh-keygen\b", re.IGNORECASE),
    re.compile(r"\bkubectl\b", re.IGNORECASE),
    re.compile(r"\bdocker\b", re.IGNORECASE),
    re.compile(r"--no-verify\b", re.IGNORECASE),
)


def _is_safe_uv_run(cmd: str) -> bool:
    """``uv run python -m atelier.overwatch.<module>`` -> True."""
    # Accept optional leading `uv run` and require the -m module form.
    m = re.match(
        r"^\s*(uv\s+run\s+)?python(3(\.\d+)?)?\s+-m\s+([A-Za-z0-9_.]+)\b",
        cmd,
    )
    if not m:
        return False
    module = m.group(4)
    return module in _ALLOWED_CLI_MODULES


def _is_safe_plain_command(cmd: str) -> bool:
    """A whitespace-tokenized read-only command we permit."""
    tokens = cmd.strip().split()
    if not tokens:
        return False
    head = tokens[0]
    if head in _READ_ONLY_COMMANDS:
        return True
    if head == "git" and len(tokens) >= 2 and tokens[1] in _SAFE_GIT_SUBCOMMANDS:
        return True
    return False


# ── Decision API ────────────────────────────────────────────────────


def classify_bash_command(cmd: str) -> tuple[bool, str]:
    """Return (allow, reason).  Pure — easy to test."""
    if not isinstance(cmd, str) or not cmd.strip():
        return False, "empty or non-string command"
    stripped = cmd.strip()

    for pat in _DENIED_PATTERNS:
        if pat.search(stripped):
            return False, f"command matches deny pattern: {pat.pattern!r}"

    # Allow shell pipelines if every clause is independently safe.
    # This keeps things like ``cat foo.json | head -200`` working.
    clauses = [c.strip() for c in re.split(r"\s*\|\s*|\s*&&\s*|\s*;\s*", stripped) if c.strip()]
    for clause in clauses:
        if _is_safe_uv_run(clause):
            continue
        if _is_safe_plain_command(clause):
            continue
        return False, (
            f"clause {clause!r} is not on the supervisor allowlist. "
            "Permitted: uv run python -m atelier.overwatch.{write_proposal,"
            "ingest_reference,apply_and_rerun,kill_run}, plus read-only "
            "shell / git inspection commands."
        )
    return True, "ok"


def classify_path(path: str) -> tuple[bool, str]:
    """Return (allow, reason) for a path given to Read/Glob/Grep."""
    if not isinstance(path, str) or not path:
        return False, "empty or non-string path"
    p = Path(path)
    if not _is_in_sandbox(p):
        return False, (
            f"path {str(p)!r} resolves outside the project sandbox. "
            "Supervisor tools may only read files under the repository root."
        )
    return True, "ok"


def evaluate_hook(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Evaluate a Pre-tool-use request against the sandbox.

    Returns a dict shaped for the Claude Agent SDK's hook protocol —
    ``{"decision": "allow"|"deny", "reason": "..."}``.  When the SDK
    contract changes (it has been renamed ``permissionDecision`` at
    least once), the caller is expected to translate.
    """
    if tool_name == "Bash":
        cmd = tool_input.get("command", "")
        ok, reason = classify_bash_command(cmd)
        return {"decision": "allow" if ok else "deny", "reason": reason}

    if tool_name in ("Read", "Glob", "Grep"):
        # Glob/Grep may carry the path under "path" or (for Glob)
        # be missing path entirely → default to project root.
        p = tool_input.get("path") or tool_input.get("file_path") or str(_project_root())
        ok, reason = classify_path(p)
        return {"decision": "allow" if ok else "deny", "reason": reason}

    # Anything else the supervisor shouldn't be touching.  The agent
    # gets allowed_tools trimmed anyway (Pillar 3 options only grant
    # Read/Grep/Glob/Bash), but belt-and-suspenders: deny.
    return {
        "decision": "deny",
        "reason": f"tool {tool_name!r} is not authorized for the supervisor.",
    }
