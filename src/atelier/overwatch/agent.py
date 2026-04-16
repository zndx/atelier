"""Overwatch agent — monitors classification pipeline runs and writes
recommendations to build/results/{run_id}/overwatch.md.

Requires direct Anthropic API access (``has_overwatch=True``). Uses the
Claude Agent SDK in single-turn mode to analyze pipeline artifacts and
produce actionable insights.

The agent is invoked after each pipeline run reaches CONVERGED. It reads:
- classifications.json (per-column predictions + confidence + evidence)
- evaluation_report.json (accuracy, macro/micro F1, per-category metrics)
- The pipeline summary dict (K, gap, coverage, iterations, token usage)

And writes:
- overwatch.md — structured recommendations for the operator
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def run_overwatch_analysis(
    cfg,
    run_id: str,
    summary: dict[str, Any],
    results_dir: Path | str,
) -> Path | None:
    """Run the overwatch agent on a completed pipeline run.

    Returns the path to the overwatch.md file, or None if overwatch
    is not available or the analysis fails.
    """
    if not cfg.has_overwatch:
        return None

    # Hard validation: overwatch requires a real Anthropic API key.
    # This is not negotiable — Bedrock cannot power overwatch.
    if not cfg.anthropic_api_key:
        log.warning("Overwatch: ANTHROPIC_API_KEY required (not Bedrock). Skipping.")
        return None

    results_dir = Path(results_dir)
    classifications_path = results_dir / "classifications.json"
    eval_path = results_dir / "evaluation_report.json"
    overwatch_path = results_dir / "overwatch.md"

    if not classifications_path.exists():
        log.warning("Overwatch: no classifications.json in %s", results_dir)
        return None

    # Build the analysis prompt from pipeline artifacts
    prompt = _build_analysis_prompt(
        run_id=run_id,
        summary=summary,
        classifications_path=classifications_path,
        eval_path=eval_path,
    )

    try:
        analysis = _query_overwatch(cfg, prompt)
        overwatch_path.write_text(analysis)
        log.info("Overwatch analysis written to %s", overwatch_path)
        return overwatch_path
    except Exception as e:
        log.warning("Overwatch analysis failed (non-fatal): %s", e)
        return None


def _build_analysis_prompt(
    run_id: str,
    summary: dict[str, Any],
    classifications_path: Path,
    eval_path: Path,
) -> str:
    """Build the overwatch analysis prompt from pipeline artifacts."""
    parts = [
        "You are the Atelier Overwatch agent. Analyze this classification "
        "pipeline run and write a structured recommendations report.\n",
        f"## Run: {run_id}\n",
        "## Pipeline Summary\n",
        f"```json\n{json.dumps(summary, indent=2, default=str)}\n```\n",
    ]

    # Include classification results (truncated for large runs)
    try:
        classifications = json.loads(classifications_path.read_text())
        total = len(classifications)
        # Include low-confidence and high-conflict columns in full
        flagged = [
            c for c in classifications
            if float(c.get("confidence", 1.0)) < 0.7
            or float(c.get("conflict", 0.0)) > 0.5
        ]
        parts.append(f"## Classifications: {total} total, {len(flagged)} flagged\n")
        if flagged:
            parts.append("### Flagged columns (low confidence or high conflict)\n")
            parts.append(f"```json\n{json.dumps(flagged[:50], indent=2, default=str)}\n```\n")
        # High-level distribution
        codes = [c.get("predicted_code", "") for c in classifications]
        from collections import Counter
        dist = Counter(codes).most_common(20)
        parts.append("### Classification distribution (top 20)\n")
        for code, count in dist:
            parts.append(f"- {code}: {count}\n")
    except Exception as e:
        parts.append(f"*Could not read classifications: {e}*\n")

    # Include evaluation report if available
    if eval_path.exists():
        try:
            eval_report = json.loads(eval_path.read_text())
            parts.append("\n## Evaluation Report\n")
            parts.append(f"```json\n{json.dumps(eval_report, indent=2, default=str)}\n```\n")
        except Exception:
            pass

    parts.append(
        "\n## Instructions\n\n"
        "Write a markdown report with these sections:\n"
        "1. **Summary** — one paragraph assessment of this run\n"
        "2. **Quality Metrics** — accuracy, coverage, convergence analysis\n"
        "3. **Flagged Columns** — columns needing attention and why\n"
        "4. **Pattern Analysis** — recurring issues or category confusions\n"
        "5. **Recommendations** — specific, actionable next steps\n"
        "6. **Configuration Suggestions** — any parameter tuning needed\n\n"
        "Be specific and concise. Reference column names and codes directly.\n"
    )

    return "".join(parts)


def _query_overwatch(cfg, prompt: str) -> str:
    """Query the overwatch Claude instance via Agent SDK."""
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
    from pathlib import Path

    # Build a clean env for direct Anthropic API — no Bedrock.
    # Overwatch is an independent SDK instance that must never route
    # through Bedrock regardless of the terminal/pipeline config.
    env: dict[str, str] = {
        "ANTHROPIC_API_KEY": cfg.anthropic_api_key,
    }

    project_root = Path(__file__).resolve().parent.parent.parent.parent

    options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="bypassPermissions",
        model=cfg.overwatch_model,
        max_turns=1,  # Single-turn analysis, no tool use
        cwd=str(project_root),
        env=env,
    )

    text_parts: list[str] = []
    import asyncio

    async def _run():
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_parts.append(block.text)

    # Run async query in sync context
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already in async context (gateway) — use a thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _run()).result(timeout=120)
        else:
            loop.run_until_complete(_run())
    except RuntimeError:
        asyncio.run(_run())

    return "\n".join(text_parts) if text_parts else "# Overwatch Analysis\n\n*No analysis generated.*\n"
