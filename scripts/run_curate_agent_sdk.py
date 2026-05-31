"""Spawn the Agent SDK against /curate-agent-mediated, run to completion.

Invoked from `just optimize agent` as the middle phase between
`build_agent_mediated.py` (sets up working_set.json) and `apply_review.py`
(persists agent decisions).

The Agent SDK on the CAI pod does NOT auto-discover .claude/commands/
(per the agent-sdk-command-discovery memory), so we pass the skill's
file contents as the prompt explicitly and set ``cwd`` to the project
root so the skill's relative paths resolve.

Model + credential resolution follows the same pattern as the Web
Terminal Agent (``src/atelier/terminal.py``): honor the operator's
active picker selection if present, else fall back to ``cfg.agent_model``,
and let ``_build_sdk_env`` route to Bedrock or direct API based on what
credentials are present.

Exits non-zero on agent failure so the calling `just` recipe surfaces
errors via ``set -euo pipefail``.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path("/home/cdsw")
SKILL_PATH = PROJECT_ROOT / ".claude/commands/curate-agent-mediated.md"

# Repo src is not on PYTHONPATH when invoked via `uv run python scripts/...`
sys.path.insert(0, str(PROJECT_ROOT / "src"))


async def _run() -> int:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        query,
    )

    from atelier.agents.client import _build_sdk_env
    from atelier.config import load_config
    from atelier.model_compat import requires_adaptive_thinking
    from atelier.terminal_selection import active_model_ref

    cfg = load_config()
    model = active_model_ref(cfg) or os.environ.get("ATELIER_AGENT_MODEL", "")
    if not model:
        print("ERROR: no agent model resolved (set ATELIER_AGENT_MODEL or configure agents.model)", file=sys.stderr)
        return 2

    env = _build_sdk_env(cfg, selected_model=model)

    if not SKILL_PATH.exists():
        print(f"ERROR: skill file not found at {SKILL_PATH}", file=sys.stderr)
        return 2

    prompt = (
        SKILL_PATH.read_text()
        + "\n\n---\n\n"
        + "Execute this skill end-to-end against the working set prepared by "
        + "scripts/build_agent_mediated.py.  When the curation pass is complete, "
        + "STOP and return control — the calling recipe will run apply_review.py "
        + "to persist your decisions."
    )

    thinking_kwargs: dict = {}
    if requires_adaptive_thinking(model):
        thinking_kwargs["max_thinking_tokens"] = 0
        thinking_kwargs["effort"] = "high"

    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        permission_mode="bypassPermissions",
        model=model,
        max_turns=None,
        max_budget_usd=50.0,
        cwd=str(PROJECT_ROOT),
        setting_sources=["project"],
        env=env,
        **thinking_kwargs,
    )

    print(f"=== run_curate_agent_sdk ===")
    print(f"  model: {model}")
    print(f"  skill: {SKILL_PATH}")
    print(f"  cwd  : {PROJECT_ROOT}")
    print()

    rc = 0
    async for msg in query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    print(block.text)
        elif isinstance(msg, ResultMessage):
            if getattr(msg, "is_error", False):
                print(f"=== agent ERROR ===", file=sys.stderr)
                rc = 1
            else:
                cost = getattr(msg, "total_cost_usd", None)
                turns = getattr(msg, "num_turns", None)
                print(f"=== agent done (turns={turns}, cost_usd={cost}) ===")
    return rc


def main() -> int:
    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
