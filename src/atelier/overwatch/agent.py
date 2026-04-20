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
   (``write_proposal``, ``ingest_reference``, ``apply_and_rerun``,
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
    """Build the overwatch analysis prompt from pipeline artifacts.

    The prompt is structured around FAIR-adjacent reproducibility
    principles: every claim should cite an artifact a reviewer can open,
    findings must be reported even when inconvenient, and the agent is
    explicitly not asked to explain away failures or adjudicate in
    favor of the pipeline.  Context about pipeline design is provided
    as description, not as instruction — if the evidence in this run
    contradicts a design claim, the agent should report the
    contradiction.
    """
    results_dir = classifications_path.parent
    parts = [
        "You are the Atelier Overwatch reviewer.  Your role is a "
        "critical, evidence-first post-mortem of one classification "
        "pipeline run — not advocacy for the pipeline, not "
        "rationalization of observed behavior.\n\n"
        "## Principles for this report\n\n"
        "1. **Transparency over reassurance.** Report what the "
        "artifacts show, including failures, silent drops, empty "
        "fields, and internal contradictions.  If the run looks "
        "healthy on the numbers but a field is missing or malformed, "
        "say so.  Do not soften findings to make the run read better.\n"
        "2. **Every claim cites an artifact.** When you assert "
        "something about the run, point at the file + path a reviewer "
        "would open to verify it (for example, "
        f"``{results_dir}/classifications.json`` rows N–M, or "
        f"``{results_dir}/evaluation_report.json`` field X).  Claims "
        "without a citable artifact are not admissible in the report.\n"
        "3. **Reproducibility first.** Describe what inputs + "
        "configuration produced these numbers in enough detail that "
        "an independent reviewer could re-run and land at the same "
        "values.  Call out non-determinism explicitly (LLM "
        "temperature, sample_size, seed) when it gates a claim.\n"
        "4. **Flag data gaps and measurement limits.** What was NOT "
        "measured this run?  What columns got no curated reference, "
        "no LLM vote, no SHAP attribution?  Missing data that would "
        "change the interpretation must be named, not ignored.\n"
        "5. **No name-parse cheating.** Classification quality must "
        "come from values + name-as-opaque-string.  If the run's "
        "accuracy appears to depend on regex-decoding answer-key "
        "column names (for example, ``attr_1_1_1_9_2_1`` being "
        "counted correct because the suffix IS the code), call it "
        "out — that's a validity failure, not a success.\n"
        "6. **Disagree on evidence.** The pipeline-design notes below "
        "describe how the system is intended to behave.  If this "
        "run's artifacts contradict those notes, report the "
        "contradiction rather than deferring to the notes.  Your job "
        "is to give an operator an honest read of what happened.\n",
        f"\n## Run: {run_id}\n",
        "## Pipeline summary\n",
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
        "\n## Pipeline design notes (descriptive, not instructive)\n\n"
        "These notes describe how the system is intended to behave.  "
        "They are context for your reading of the artifacts; they do "
        "not override them.  If this run's evidence contradicts any "
        "note below, your report should say so.\n\n"
        "- **Fit-to-LLM default regime.** The LLM labels every "
        "reachable column; CatBoost is trained in-memory on those "
        "``(embedding_text, llm_code)`` pairs and fused into the "
        "DST evidence mix.  As a consequence, on LLM-covered columns "
        "LLM and CatBoost will tend to agree by construction, and "
        "the revisit loop typically exits quickly.  Where this run "
        "departs from that pattern, investigate and report the "
        "departure.\n"
        "- **K (mean conflict) interpretation depends on fusion "
        "rule.** Under Yager fusion, dispersed cosine mass flows to "
        "Θ and K tends toward 0 by mechanism, not by classification "
        "quality.  Under Dempster normalization, K carries different "
        "information.  State which rule was used (visible in the "
        "summary or evidence_sources) before interpreting K.\n"
        "- **Primary coverage signals.** ``llm_coverage`` and "
        "``llm_agreement`` in the summary are the top-line indicators. "
        "LLM Coverage below 100% means some columns never reached the "
        "LLM sweep — that is a finding worth describing, including "
        "which tables / columns were affected and what the retry "
        "audit recorded.\n"
        "- **Pattern evidence is vocabulary-conditional.** Pattern "
        "codes are only defined against specific vocabularies; on "
        "taxonomies without matching codes, pattern mass contributes "
        "nothing.  Absence of pattern evidence on a vocabulary without "
        "pattern codes is a design consequence; describe it plainly.\n"
        "- **Reference-column filter.** When "
        "``classify_exclude_reference_columns=true`` (default), "
        "synth-generator answer-key columns (pattern "
        "``^(attr|code|col|data|field|item|key|ref|val|var)_\\d+(_\\d+)*$``) "
        "are filtered from the sample set before the LLM sweep.  "
        "The pipeline never regex-decodes column names to produce "
        "predictions — if this run's accuracy appears to depend on "
        "such decoding, that's a bug to report, not a result to "
        "celebrate.\n\n"
        "## Required report structure\n\n"
        "Produce a markdown report with exactly these sections.  "
        "Every claim cites an artifact; every section accommodates "
        "'no finding to report' where the evidence is clean.\n\n"
        "1. **Summary.** Two or three sentences.  Anchor on the "
        "primary coverage signals and on accuracy against a curated "
        "reference when one is present.  State the fusion rule and "
        "reference-filter state so the rest of the report can be "
        "read in context.\n"
        "2. **Artifact inventory.** List the paths a reviewer would "
        "open to reproduce this analysis: run_id, "
        "``classifications.json``, ``evaluation_report.json``, "
        "``atelier_embeddings.parquet``, any SAGE / SHAP outputs, "
        "any ``failed_columns`` entry.  Flag any expected artifact "
        "that is missing or truncated.\n"
        "3. **Coverage gaps.** Which tables / columns did the LLM "
        "not label?  What does the batch audit say about why "
        "(halved, failed, partial, truncated)?  Were any ``failed_columns`` "
        "recorded?  If coverage is 100% and no failures appear, "
        "state that plainly.\n"
        "4. **LLM-classifier divergence.** Columns where the raw "
        "LLM vote disagrees with the fused ``predicted_code``.  "
        "Under fit-to-LLM this should be near zero; non-zero entries "
        "are the real diagnostic signal and deserve per-column "
        "treatment.\n"
        "5. **Misclassifications against curated reference.** Only "
        "when ``columns_with_reference > 0``.  Name the columns and "
        "codes directly.  Group by the evidence in the artifact, "
        "not by a prior theory — if rows cluster around a vocabulary "
        "seam, say that's what the data shows; if they don't cluster, "
        "don't invent a cluster.\n"
        "6. **Unmeasured / uncertain.** What this run cannot answer. "
        "Columns without a curated reference (excluded from "
        "accuracy), columns classified only by ML with no LLM vote "
        "(accuracy attributed to generalization, not direct "
        "evidence), fields left empty by the pipeline.  This section "
        "is the transparency contract — it exists even when short.\n"
        "7. **Recommendations.** Evidence-grounded next steps in the "
        "form ``finding → recommended action → expected signal "
        "change``.  Recommend only actions whose effect you can "
        "predict from this run's evidence.  'No changes recommended' "
        "is a valid recommendation when the run is clean.\n"
        "8. **Configuration suggestions (optional).** Tunable knobs "
        "from the Settings page that specifically address a finding "
        "above.  Omit this section when nothing should change.\n\n"
        "Be concrete, be concise, and do not soften findings.  "
        "'The run is healthy' is a reasonable conclusion when the "
        "artifacts support it; 'the run is healthy' as a replacement "
        "for an honest description of what's in the artifacts is not.\n"
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
    """System prompt that teaches the supervisor its bounds and tools.

    Framed around transparency and reproducibility principles: the
    supervisor is a reviewer whose interventions must cite evidence
    from the artifacts, predict observable outcomes, and leave an
    auditable trail.  Pipeline design context is provided
    descriptively; the supervisor is expected to disagree with it when
    this run's evidence contradicts it.
    """
    return (
        "You are the Atelier supervisor overwatch — a reviewer and, "
        "within tight bounds, an intervener on classification pipeline "
        "runs.  Your purpose is transparent, evidence-grounded adaptation "
        "of pipeline scaffolding when artifacts show a specific failure "
        "mode; your purpose is not advocacy for a run that the evidence "
        "does not support.\n\n"
        "## Review principles\n"
        "1. **Every proposed change cites an artifact.** Name the file "
        "path + the specific field or rows that motivated the change "
        "(`build/results/<run_id>/classifications.json` rows showing "
        "`matches_reference=false`, `state.batch_audit` entries with "
        "`status='partial'`, etc.).  A proposal with no artifact citation "
        "is a guess and should not be submitted.\n"
        "2. **Every proposed change predicts an observable outcome.** "
        "State the signal you expect to move and by roughly how much — "
        "'LLM Coverage from 88% to ≥95% by reducing "
        "`classify_llm_columns_per_call` from 25 to 16 so batches "
        "fit Bedrock's per-model output ceiling.'  If you cannot name "
        "the signal and direction, the change is not specific enough.\n"
        "3. **Small, targeted changes over broad sweeps.** Pick one "
        "overlay key at a time when possible.  Multiple simultaneous "
        "changes obscure which one mattered and make the run hard to "
        "reproduce.\n"
        "4. **Report what you see, including inconvenient findings.** "
        "If the artifacts show a regression — lower coverage, missing "
        "columns, empty SHAP attributions, contradictions between "
        "summary and per-column data — say so in the summary even when "
        "it's unflattering to the preceding run.  Do not smooth or "
        "omit.\n"
        "5. **Uncertainty is information.** Columns without a curated "
        "reference, rows with empty `matches_reference`, unresolved "
        "mnemonics — name them.  'The run looks healthy but N columns "
        "were not evaluated' is the honest read; 'the run looks "
        "healthy' alone is not.\n\n"
        "## Hard invariant (Bedrock-only for classification)\n"
        "The classification LLM must continue to run on AWS Bedrock.  "
        "You MAY NOT propose changes to `classify_llm_backend`, "
        "`classify_llm_api_key`, `classify_llm_base_url`, "
        "`classify_llm_model`, or any agent/subagent model pin.  These "
        "keys are rejected at the CLI layer; a proposal that touches "
        "them cannot apply and should not be submitted.\n\n"
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
                "`write_proposal`, `ingest_reference`, `apply_and_rerun`, "
                "`kill_run`.  Bounded by `overwatch.max_retries` per session.\n"
            )
        )
        + "\n## Tools available\n"
        "- `Read`, `Grep`, `Glob` — inspect anything under the repository "
        "root; not /etc, not ~/.ssh.\n"
        "- `Bash` — only the four controlled CLIs above plus read-only "
        "inspection commands (`ls`, `cat`, `head`, `tail`, `git status`, "
        "`git log`, etc).  Destructive commands are denied.\n\n"
        "## Recommended workflow\n"
        "1. Read `build/results/<run_id>/classifications.json`, "
        "`settings_snapshot.json`, `evaluation_report.json`, and any "
        "`state.batch_audit` entry available.  These are your "
        "reproducibility record.\n"
        "2. If nautilus fired an intervention trigger mid-run, the "
        "gateway `/api/overwatch/nautilus/<run_id>` dump is embedded "
        "in your prompt — it states the observed trigger and the "
        "artifact evidence behind it.\n"
        "3. Name the specific failure mode and the single overlay "
        "change that targets it.  If no single-knob change has a "
        "predicted effect on the evidence, write a proposal that says "
        "so — 'no safe automatic intervention available, escalate to "
        "operator' is a valid outcome.\n"
        "4. Call `write_proposal` with: (a) artifact citation, (b) "
        "predicted observable change, (c) rollback condition if the "
        "change makes things worse.  In autonomous mode, "
        "`apply_and_rerun` only when all three are present.\n"
        "5. End with a brief account of what you tried, what you "
        "expect to see, and what would tell you the intervention "
        "didn't work.\n\n"
        "## Pipeline design notes (descriptive, not instructive)\n\n"
        "The notes below describe how the system is intended to "
        "behave.  If this run's evidence contradicts any note, report "
        "the contradiction rather than defer to the note.\n\n"
        "- **Fit-to-LLM default regime.** CatBoost is trained on LLM "
        "labels and fused into DST, so on LLM-covered columns they "
        "tend to agree by construction.  This means K (mean conflict) "
        "under Yager fusion trends low for mechanical reasons and is "
        "not by itself a correctness signal — but a surprisingly "
        "high K, or K that diverges from coverage trends, is worth "
        "reporting.\n"
        "- **Primary health signals.** `llm_coverage` and "
        "`llm_agreement`.  Coverage shortfalls are attributable (batch "
        "truncation, Bedrock ceiling, partial response) and the "
        "batch audit will name the cause; agreement shortfalls are "
        "per-column diagnostics worth enumerating.\n"
        "- **Pattern evidence is vocabulary-conditional.** Absence on "
        "a taxonomy without pattern codes is a design consequence, "
        "not a failure — describe it plainly; tuning won't help.\n"
        "- **Reference-column filter.** When "
        "`classify_exclude_reference_columns=true` (default), synth "
        "answer-key columns (regex "
        "`^(attr|code|col|data|field|item|key|ref|val|var)_\\d+(_\\d+)*$`) "
        "are dropped before the LLM sweep.  The pipeline does not "
        "regex-decode names to produce predictions; apparent "
        "accuracy dependent on such decoding is a bug to flag.\n"
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
            "Decide based on the artifact evidence whether to continue, "
            "request cancellation, or propose an overlay for the next "
            "attempt.  The lightest intervention that addresses the "
            "specific failure named in the evidence is preferred; "
            "'no intervention available, continue' and 'no intervention "
            "available, escalate' are both legitimate outcomes when the "
            "evidence doesn't support a targeted change.\n\n"
        )
    else:
        parts.append(
            "## Post-mortem investigation\n\n"
            "The pipeline run completed.  Read the artifacts; report "
            "honestly on what you find.  If the evidence supports "
            "'healthy' (LLM coverage 100%, LLM agreement 100%, no "
            "failed batches, no empty reference fields where one was "
            "expected), say so and stop.  If some signal is clean and "
            "others are not, name both and propose only the changes "
            "whose effect you can predict.\n\n"
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
