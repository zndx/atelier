"""Overwatch agent — monitors classification pipeline runs.

Two entry points live here:

1. :func:`run_overwatch_analysis` — the existing single-turn post-mortem
   that writes ``overwatch.md``.  Kept as-is for operators who only
   want the markdown report.

2. :func:`run_supervisor_overwatch` — the Pillar 3 tool-using
   supervisor.  Given a completed (or errored) run, the supervisor is
   authorized to investigate (Read/Grep/Glob/Bash), propose an overlay,
   and — in autonomous mode — apply the overlay and trigger a rerun.
   All side-effecting operations go through the four controlled CLIs
   (``write_proposal``, ``ingest_ground_truth``, ``apply_and_rerun``,
   ``kill_run``); the agent has no direct ``Write`` tool.

Both entry points require a direct Anthropic API key (``has_overwatch``).
Bedrock-only deployments get neither — the classifier runs on Bedrock,
but the supervisor overwatch itself requires the Anthropic direct API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from atelier.overwatch.hooks import evaluate_hook

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

    # List the keys the operator can actually tune via the Settings
    # page, so the agent emits focus-block keys that will resolve.
    try:
        from atelier.config_overlay import SETTINGS_METADATA
        tunable_keys = sorted(SETTINGS_METADATA.keys())
    except Exception:
        tunable_keys = []

    parts.append(
        "\n## Pipeline regime (read first)\n\n"
        "Atelier operates in a **fit-to-LLM** regime by default: the "
        "LLM labels every column it can reach, then CatBoost is "
        "trained in-memory on those ``(embedding_text, llm_code)`` "
        "pairs and fused into the DST evidence mix.  This means:\n\n"
        "- **LLM and CatBoost agree by construction** on LLM-covered "
        "columns.  The revisit loop exits after one iteration with "
        "zero disagreements — that's expected, NOT a premature stop. "
        "Do not recommend bumping ``classify_bootstrap_max_iterations``.\n"
        "- **K (mean conflict) reflects cosine mass dispersal**, not "
        "classification error.  With Yager fusion (the current "
        "default) dispersed cosine mass goes to Θ and K → 0.  Do "
        "not treat K as a correctness signal under this regime.\n"
        "- **The real health signals are LLM Coverage and LLM "
        "Agreement** (both in the summary).  LLM Coverage < 95% "
        "indicates truncation or batch failures (reduce "
        "``classify_llm_columns_per_call``).  LLM Agreement < 100% "
        "on LLM-covered columns indicates a genuine CatBoost-vs-LLM "
        "divergence worth drilling into.\n"
        "- **Pattern evidence silently drops** on vocabularies where "
        "the ICE-scheme pattern codes don't match — that's fine, "
        "not a problem to solve.  Don't recommend pattern tuning on "
        "Hive / meta-tagging runs.\n\n"
        "## Instructions\n\n"
        "Write a markdown report with these sections:\n"
        "1. **Summary** — one paragraph assessment anchored on LLM "
        "Coverage + LLM Agreement (plus accuracy when ground truth "
        "is available).\n"
        "2. **Coverage gaps** — which tables / columns the LLM "
        "didn't reach, and why CatBoost's generalization may or may "
        "not have carried them correctly.\n"
        "3. **LLM-CatBoost divergence** — columns where LLM and "
        "CatBoost disagreed after fine-tuning (the real diagnostic).\n"
        "4. **Mispredictions vs ground truth** — only when "
        "``columns_with_gt > 0``.  Reference column names and codes "
        "directly; group by likely root cause (truncation, vocab "
        "seam, taxonomy ambiguity).\n"
        "5. **Recommendations** — specific, actionable next steps. "
        "**Do not recommend K-based fixes or iteration count bumps "
        "under the fit-to-LLM regime.**\n"
        "6. **Configuration Suggestions** — only when a tunable knob "
        "genuinely addresses a finding.\n\n"
        "Be specific and concise.  Where the run looks healthy, say "
        "so explicitly rather than inventing concerns.\n"
    )

    # Focus-block addendum: the Settings page surfaces the union of
    # deterministic drift rules and anything the agent names here,
    # so listing 3–7 keys that actually matter for the next run
    # steers the operator's attention efficiently.
    if tunable_keys:
        parts.append(
            "\n## Focus keys\n\n"
            "After the markdown report, append a final code block "
            "tagged ``focus`` containing a JSON object with a "
            "``focus_keys`` array — the 3–7 Settings keys most worth "
            "the operator's attention for the next run. Use the "
            "exact keys from the tunable list below; unknown keys "
            "are silently dropped. Omit the block entirely if "
            "nothing needs attention.\n\n"
            "Example:\n\n"
            "```focus\n"
            '{"focus_keys": ["mc_sample_fraction", "classify_bootstrap_clarity_target"]}\n'
            "```\n\n"
            "Tunable keys:\n```\n" + "\n".join(tunable_keys) + "\n```\n"
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

    # Opus 4.7+ requires thinking.type=adaptive + output_config.effort.
    # SDK v0.1.56 has a bug where passing the adaptive thinking dict
    # still causes the bundled CLI to emit --max-thinking-tokens 32000
    # → API rejects with "thinking.type.enabled not supported".
    # Workaround: pass max_thinking_tokens=0 + effort=<level> so the
    # CLI sends only --effort to the API.  See terminal.py for the
    # matching workaround + upstream-fix note.
    from atelier.model_compat import requires_adaptive_thinking
    thinking_kwargs: dict = {}
    if requires_adaptive_thinking(cfg.overwatch_model):
        thinking_kwargs["max_thinking_tokens"] = 0
        thinking_kwargs["effort"] = "medium"

    options = ClaudeAgentOptions(
        allowed_tools=[],
        permission_mode="bypassPermissions",
        model=cfg.overwatch_model,
        max_turns=1,  # Single-turn analysis, no tool use
        cwd=str(project_root),
        env=env,
        **thinking_kwargs,
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


# ── Supervisor overwatch (Pillar 3) ─────────────────────────────


def _make_pretooluse_hook():
    """Return an async PreToolUse hook enforcing the supervisor sandbox.

    The SDK expects an awaitable returning ``SyncHookJSONOutput`` with
    a ``hookSpecificOutput.permissionDecision`` of ``"allow"`` / ``"deny"``.
    The pure decision logic lives in :mod:`atelier.overwatch.hooks`;
    this shim just adapts to the SDK contract.
    """

    async def _hook(input_data, tool_use_id, context):  # noqa: ANN001 — SDK-typed
        tool_name = getattr(input_data, "tool_name", None) or input_data.get("tool_name", "")
        tool_input = getattr(input_data, "tool_input", None) or input_data.get("tool_input", {})
        decision = evaluate_hook(tool_name, tool_input)
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": decision["decision"],
                "permissionDecisionReason": decision["reason"],
            }
        }

    return _hook


def _supervisor_system_prompt(cfg, autonomy: str) -> str:
    """System prompt that teaches the supervisor its bounds and tools."""
    return (
        "You are the Atelier supervisor overwatch.  Your job is to adapt "
        "the classification pipeline's scaffolding when a run fails or "
        "underperforms, NOT to change the classifier provider.\n\n"
        "## Hard invariant (Bedrock-only)\n"
        "The classification LLM must continue to run on AWS Bedrock.  "
        "You MAY NOT propose changes to `classify_llm_backend`, "
        "`classify_llm_api_key`, `classify_llm_base_url`, "
        "`classify_llm_model`, or any agent/subagent model pin.  These "
        "keys are rejected at the CLI layer.\n\n"
        f"## Autonomy tier: {autonomy}\n"
        + (
            "- **monitor** — write a markdown report only.  Do not invoke "
            "the CLIs.  No side effects.\n"
            if autonomy == "monitor"
            else (
                "- **propose** — investigate freely, then call "
                "`write_proposal` to persist a `proposed_overlay.json`.  "
                "You MAY NOT call `apply_and_rerun` or `kill_run`; the "
                "operator chooses whether to apply.\n"
                if autonomy == "propose"
                else
                "- **autonomous** — you may call any of the four CLIs: "
                "`write_proposal`, `ingest_ground_truth`, `apply_and_rerun`, "
                "`kill_run`.  Bounded by `overwatch.max_retries` per session.\n"
            )
        )
        + "\n## Tools available\n"
        "- `Read`, `Grep`, `Glob` — inspect anything under the repository "
        "root; not /etc, not ~/.ssh.\n"
        "- `Bash` — only the four controlled CLIs above plus read-only "
        "inspection commands (`ls`, `cat`, `head`, `tail`, `git status`, "
        "`git log`, etc).  Destructive commands are denied.\n\n"
        "## Preferred workflow\n"
        "1. Read `build/results/<run_id>/classifications.json`, "
        "`settings_snapshot.json`, and `evaluation_report.json`.\n"
        "2. If nautilus fired an intervention trigger mid-run, read the "
        "gateway `/api/overwatch/nautilus/<run_id>` dump embedded in "
        "your prompt — it explains why you were invoked.\n"
        "3. Identify the specific failure mode and pick ONE targeted "
        "overlay change.  Small, verifiable changes beat broad sweeps.\n"
        "4. Call `write_proposal` with a clear rationale + expected "
        "effect.  If autonomous and the change is safe, also call "
        "`apply_and_rerun`.\n"
        "5. End with a brief summary of what you tried and why.\n\n"
        "Do NOT recommend K-based fixes or iteration-count bumps under "
        "the fit-to-LLM regime (K is diagnostic; CatBoost agrees with "
        "the LLM by construction).\n"
    )


def _build_supervisor_prompt(
    *,
    run_id: str,
    summary: dict[str, Any],
    session_id: str | None,
    intervention_detail: str | None,
    results_dir: Path,
) -> str:
    """Construct the initial user prompt for a supervisor invocation."""
    parts: list[str] = []

    if intervention_detail:
        parts.append(
            "## You were invoked by a mid-run intervention\n\n"
            f"{intervention_detail}\n\n"
            "Decide whether to continue, request cancellation, or "
            "propose an overlay for the next attempt.  Prefer the "
            "lightest intervention that addresses the specific "
            "failure — don't cancel a run that's merely slow.\n\n"
        )
    else:
        parts.append(
            "## Post-mortem investigation\n\n"
            "The pipeline run completed.  Investigate whether a "
            "follow-up attempt with a targeted overlay would help.  "
            "If the run is healthy (LLM coverage 100%, LLM agreement "
            "100%, no failed batches), say so and stop.\n\n"
        )

    parts.append(f"### Run: `{run_id}`\n")
    parts.append(f"### Results directory: `{results_dir}`\n")
    if session_id:
        parts.append(f"### Supervisor session: `{session_id}`\n")
    parts.append("\n### Pipeline summary (as returned by the pipeline)\n")
    parts.append(f"```json\n{json.dumps(summary, indent=2, default=str)}\n```\n\n")

    parts.append(
        "### What to do\n\n"
        "Read the artifacts, decide on one targeted remediation (or "
        "none), and execute the controlled CLIs per the system prompt.  "
        "Respond with a concise summary — the operator already has the "
        "artifacts; you don't need to paste them back.\n"
    )
    return "".join(parts)


def run_supervisor_overwatch(
    cfg,
    *,
    run_id: str,
    summary: dict[str, Any],
    results_dir: Path | str,
    session_id: str | None = None,
    intervention_detail: str | None = None,
    max_turns: int = 30,
    max_budget_usd: float = 15.0,
) -> dict[str, Any]:
    """Run the tool-using supervisor overwatch on a pipeline run.

    Returns a dict with ``{"status", "transcript", "error"}``.
    ``status`` is ``"ok"`` on clean completion, ``"skipped"`` when
    overwatch isn't available, or ``"error"`` on failure.  The
    supervisor's side-effects (proposal writes, reruns) happen via
    the controlled CLIs — read them from the session/results dir.
    """
    autonomy = getattr(cfg, "overwatch_autonomy", "propose")

    if not cfg.has_overwatch:
        return {"status": "skipped", "reason": "overwatch disabled or no Anthropic key"}

    results_dir = Path(results_dir)

    try:
        from claude_agent_sdk import (
            AssistantMessage, ClaudeAgentOptions, HookMatcher, TextBlock, query,
        )
    except ImportError as exc:
        return {"status": "skipped", "reason": f"claude_agent_sdk unavailable: {exc}"}

    project_root = Path(__file__).resolve().parents[3]

    env: dict[str, str] = {"ANTHROPIC_API_KEY": cfg.anthropic_api_key}

    from atelier.model_compat import requires_adaptive_thinking
    thinking_kwargs: dict[str, Any] = {}
    if requires_adaptive_thinking(cfg.overwatch_model):
        # SDK v0.1.56 workaround — see the matching comment in
        # run_overwatch_analysis above.
        thinking_kwargs["max_thinking_tokens"] = 0
        thinking_kwargs["effort"] = "medium"

    hook = _make_pretooluse_hook()
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob", "Bash"],
        disallowed_tools=["Write", "Edit", "NotebookEdit"],
        permission_mode="default",
        model=cfg.overwatch_model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        cwd=str(project_root),
        env=env,
        system_prompt=_supervisor_system_prompt(cfg, autonomy),
        hooks={"PreToolUse": [HookMatcher(hooks=[hook])]},
        **thinking_kwargs,
    )

    user_prompt = _build_supervisor_prompt(
        run_id=run_id,
        summary=summary,
        session_id=session_id,
        intervention_detail=intervention_detail,
        results_dir=results_dir,
    )

    transcript: list[str] = []
    import asyncio

    async def _run():
        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        transcript.append(block.text)

    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pool.submit(asyncio.run, _run()).result(timeout=600)
            else:
                loop.run_until_complete(_run())
        except RuntimeError:
            asyncio.run(_run())
    except Exception as exc:
        log.exception("supervisor overwatch failed")
        return {"status": "error", "error": str(exc), "transcript": transcript}

    return {"status": "ok", "transcript": transcript}


def run_supervisor_intervention(
    cfg,
    *,
    intervention_record: dict[str, Any],
    run_id: str,
    summary: dict[str, Any] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Short-budget supervisor invocation triggered by nautilus.

    Compared to the post-mortem ``run_supervisor_overwatch``, this is:

    - bounded to ``max_turns=10`` and ``max_budget_usd=3`` so a
      stalled run doesn't sink budget into investigation
    - expected to produce one concrete decision (continue / cancel /
      proposal), not a full report
    """
    detail = (
        f"Trigger: **{intervention_record.get('trigger', 'unknown')}**\n"
        f"Detail: {intervention_record.get('trigger_detail', '')}\n"
        f"FSM state: {intervention_record.get('fsm_state', '?')} "
        f"(last update {intervention_record.get('fsm_updated_at', '?')})\n"
        f"Batch audit length: {intervention_record.get('batch_audit_len', 0)}\n"
        f"Failed batches: {intervention_record.get('failed_batch_count', 0)}\n"
    )
    results_dir = Path(__file__).resolve().parents[3] / "build" / "results" / run_id
    return run_supervisor_overwatch(
        cfg,
        run_id=run_id,
        summary=summary or {},
        results_dir=results_dir,
        session_id=session_id,
        intervention_detail=detail,
        max_turns=10,
        max_budget_usd=3.0,
    )
