# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Agent-driven convergence loop using Messages API with tool_use.

Replaces the programmatic convergence loop in pipeline.py with a Claude
agent that reasons about DST conflict and decides which columns to revisit.
The agent's tools wrap the same functions as the programmatic loop
(bootstrap._llm_revisit, _run_ml_validation, _identify_disagreements).

Requires ``classify.agent.enabled=true`` in config. Disabled by default —
the programmatic loop runs unchanged when the agent is not enabled.

Tool dispatch is in-process Python (not subprocess/CLI), making it testable
with a mock client that returns pre-built tool_use blocks.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.bootstrap import BootstrapConfig, BootstrapState
    from atelier.classify.llm_backend import LLMBackend
    from atelier.classify.mass_functions import DiscountConfig
    from atelier.classify.sampler import ColumnSample
    from atelier.classify.taxonomy import HierarchicalCategorySet
    from atelier.config import AtelierConfig

logger = logging.getLogger(__name__)

# ── Tool definitions (JSON schemas for Messages API tools=) ──────


TOOLS = [
    {
        "name": "get_conflict_report",
        "description": (
            "Get DST conflict analysis: high-K columns, disagreeing "
            "evidence sources, confusable pairs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "k_threshold": {
                    "type": "number",
                    "description": "Minimum conflict K to include (default 0.2)",
                },
            },
        },
    },
    {
        "name": "revisit_columns",
        "description": (
            "Re-classify specific columns with enriched context "
            "(ML prediction, pattern signals, value descriptions). "
            "Updates DST state via LLM revisit + ML revalidation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to re-classify with enriched LLM context",
                },
            },
            "required": ["column_names"],
        },
    },
    {
        "name": "check_convergence",
        "description": (
            "Check current convergence metrics: coverage, mean K, "
            "K trend, iteration count."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "get_column_detail",
        "description": (
            "Deep-dive into a single column: all evidence source masses, "
            "belief path, pattern signals, sample values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "column_name": {
                    "type": "string",
                    "description": "Column to inspect",
                },
            },
            "required": ["column_name"],
        },
    },
    {
        "name": "declare_converged",
        "description": (
            "Declare the classification converged and exit the loop. "
            "Call when metrics are acceptable or further iteration "
            "won't improve results.  The pipeline targets "
            "``boot_cfg.min_iterations`` revisit cycles; when you "
            "declare convergence early, pipeline-side fallback runs "
            "additional programmatic revisits to honor the directive. "
            "Prefer doing meaningful revisits inside the agent loop "
            "rather than relying on the fallback — your candidate "
            "selection and revisit context will be richer than what "
            "the programmatic path can do."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "convergence_kind": {
                    "type": "string",
                    "enum": [
                        "iterative_convergence",
                        "no_revisit_candidates",
                        "gap_threshold_met",
                        "plateau",
                        "budget_exhausted",
                        "agent_convergence",
                    ],
                    "description": (
                        "Structured tag describing WHY this converged. "
                        "Pick the most accurate from the enum so "
                        "downstream consumers (Status UI, overwatch) can "
                        "categorize the run.  Use 'iterative_convergence' "
                        "when revisits have settled the predictions; "
                        "'no_revisit_candidates' when no columns remain "
                        "above the bel_floor / gap_threshold; "
                        "'gap_threshold_met' when mean(Pl − Bel) < "
                        "boot_cfg.gap_threshold (the primary belief-gap "
                        "convergence criterion — K is diagnostic, not "
                        "load-bearing); 'plateau' when belief gap stopped "
                        "decreasing; 'budget_exhausted' when LLM call "
                        "budget is the constraint; 'agent_convergence' as "
                        "a default when none of the above fit cleanly."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Free-form prose explaining the decision in 1-3 "
                        "sentences with specific metric references.  This "
                        "is the audit trail; surface the actual numbers "
                        "you observed (mean_gap is primary; mean_k, "
                        "disagreements, coverage are diagnostic)."
                    ),
                },
            },
            "required": ["convergence_kind", "reason"],
        },
    },
]


AGENT_SYSTEM_PROMPT = """\
You are the Atelier classification agent. Your goal is to classify data columns
with high confidence using Dempster-Shafer Theory (DST) evidence fusion.

You have already completed an initial classification pass. Now you are in
the convergence loop: examining weakly-supported predictions (high belief
gap, low Bel) and source disagreements, investigating why, and requesting
targeted re-classification with enriched context.

## Convergence signal (read carefully)
The pipeline's primary convergence criterion is the **mean belief gap**
``mean(Pl − Bel)``.  A small gap means evidence tightly supports the
predicted code; a large gap means the prediction is plausible but not
confident.  DST conflict K is a useful **diagnostic** for spotting source
disagreement, but it is not the convergence criterion — under default
Dempster fusion, K stays high because the loop normalizes conflict out,
so gating convergence on K would be vestigial.  Reason about the gap.

## Your Tools
- get_conflict_report: Surface columns where evidence sources disagree
  (high K).  Useful diagnostic for finding investigation targets.
- get_column_detail: Deep-dive into a specific column's evidence breakdown
  (Bel, Pl, gap, per-source masses).
- revisit_columns: Re-classify selected columns with enriched context
- check_convergence: See overall metrics (coverage, mean_gap, mean_k, trend)
- declare_converged: Stop when further iteration won't help

## Strategy
1. Check overall metrics — what is mean_gap?  Is it above gap_threshold?
2. Pull the conflict report and column details to find columns with high
   gap or low Bel — those are the predictions that deserve a revisit.
3. For source-disagreement columns (high K), examine the evidence — do
   sources agree on the category family but disagree on the specific
   leaf?  That's a confusable pair worth re-classifying with context.
4. Prioritize revisiting columns where pattern evidence or name matching
   provides strong signal that the LLM may have missed.
5. After each revisit batch, check convergence — is mean_gap decreasing?
6. Declare converged when: mean_gap < gap_threshold, or the gap has
   plateaued, or remaining disagreements are confusable pairs (expected
   ambiguity).

## Project directive — minimum iterations
The pipeline targets ``boot_cfg.min_iterations`` (default 2) revisit
cycles before declaring convergence — "iteration is part of the
algorithm we publish numbers for."  When the corpus' initial metrics
look settled and you call ``declare_converged`` early, the pipeline-
side fallback runs additional programmatic revisits over the
broader uncertain-columns set to satisfy the directive.  Your job is
to do thorough work within the agent loop; the pipeline guarantees
the directive holds either way.

If conflict / belief metrics warrant additional revisits, do them
inside the loop — that's better than letting the fallback handle it,
because you can pick the candidate set and enrich the revisit context
more thoughtfully than the programmatic fallback can.

## Key Insight
High conflict K and high belief gap are SIGNAL, not error.  When cosine
similarity says "EMAIL" but the LLM says "USERNAME", or when the predicted
code has Bel = 0.4 and Pl = 0.9, those are columns worth investigating —
not discarding.
"""


# ── Tool handlers ────────────────────────────────────────────────


def _handle_get_conflict_report(
    state: BootstrapState,
    column_names: list[str],
    category_set: HierarchicalCategorySet,
    k_threshold: float = 0.2,
) -> dict[str, Any]:
    """Format BootstrapState conflict data for the agent.

    Includes both K (source disagreement) and belief-gap (prediction
    certainty) for each column, so the agent can distinguish "sources
    fought but winner is clear" from "genuinely uncertain."
    """
    high_k = []
    for name in column_names:
        k = state.ml_conflict.get(name, 0.0)
        bel = state.ml_belief.get(name, 0.0)
        pl = state.ml_plausibility.get(name, 1.0)
        gap = pl - bel

        # Report columns with high K OR high gap OR low belief
        if k <= k_threshold and gap <= 0.3 and bel >= 0.5:
            continue

        llm_code = state.labels.get(name, "")
        ml_code = state.ml_prediction.get(name, "")

        llm_cat = category_set.by_code.get(llm_code) or category_set.all_by_code.get(llm_code)
        ml_cat = category_set.by_code.get(ml_code) or category_set.all_by_code.get(ml_code)

        high_k.append({
            "column_name": name,
            "conflict_K": round(k, 4),
            "belief": round(bel, 4),
            "plausibility": round(pl, 4),
            "gap": round(gap, 4),
            "settled": bel >= 0.7 and gap <= 0.15,
            "llm_prediction": llm_cat.label if llm_cat else llm_code,
            "llm_code": llm_code,
            "ml_prediction": ml_cat.label if ml_cat else ml_code,
            "ml_code": ml_code,
            "disagrees": llm_code != ml_code,
        })

    high_k.sort(key=lambda x: -x["gap"])  # sort by uncertainty, not K
    return {
        "total_columns": len(column_names),
        "flagged_count": len(high_k),
        "k_threshold": k_threshold,
        "columns": high_k,
    }


def _handle_revisit_columns(
    state: BootstrapState,
    boot_cfg: BootstrapConfig,
    backend: LLMBackend,
    system_prompt: str,
    column_names_to_revisit: list[str],
    all_column_names: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    has_embeddings: bool,
    discounts: DiscountConfig | None = None,
) -> dict[str, Any]:
    """Revisit selected columns then revalidate ML on all columns."""
    from atelier.classify.bootstrap import (
        _llm_revisit,
        _mean_k,
        _run_ml_validation,
        record_iteration_metrics,
        _identify_disagreements,
    )

    prev_mean_k = _mean_k(state, all_column_names)
    state.iteration += 1

    _llm_revisit(
        state, boot_cfg, backend, system_prompt,
        column_names_to_revisit, samples, column_table, category_set,
    )

    _run_ml_validation(
        state, boot_cfg, all_column_names, samples,
        category_set, frame, has_embeddings, discounts=discounts,
    )

    disagreements = _identify_disagreements(state, all_column_names, boot_cfg)
    new_mean_k = _mean_k(state, all_column_names)
    record_iteration_metrics(
        state, all_column_names, len(disagreements), boot_cfg,
        revisited_this_iter=set(column_names_to_revisit),
    )

    return {
        "revisited": len(column_names_to_revisit),
        "iteration": state.iteration,
        "mean_k_before": round(prev_mean_k, 4),
        "mean_k_after": round(new_mean_k, 4),
        "k_delta": round(new_mean_k - prev_mean_k, 4),
        "remaining_disagreements": len(disagreements),
        "llm_calls_total": state.llm_calls_total,
    }


def _handle_check_convergence(
    state: BootstrapState,
    column_names: list[str],
    boot_cfg: BootstrapConfig,
) -> dict[str, Any]:
    """Return convergence metrics from BootstrapState."""
    from atelier.classify.bootstrap import (
        _coverage,
        _mean_k,
        _max_k,
        _mean_gap,
        _mean_bel,
        _frac_needing_clarification,
        k_convergence_rate,
        gap_convergence_rate,
        _identify_disagreements,
        _identify_uncertain_columns,
    )

    disagreements = _identify_disagreements(state, column_names, boot_cfg)
    uncertain = _identify_uncertain_columns(state, column_names, boot_cfg)

    return {
        "iteration": state.iteration,
        "coverage": round(_coverage(state, column_names), 4),
        # Belief-gap convergence (primary)
        "mean_gap": round(_mean_gap(state, column_names), 4),
        "mean_bel": round(_mean_bel(state, column_names), 4),
        "frac_unclear": round(_frac_needing_clarification(state, column_names), 4),
        "gap_threshold": boot_cfg.gap_threshold,
        "clarity_target": boot_cfg.clarity_target,
        "gap_convergence_rate": round(gap_convergence_rate(state), 6),
        "uncertain_columns": len(uncertain),
        # K (diagnostic)
        "mean_k": round(_mean_k(state, column_names), 4),
        "max_k": round(_max_k(state, column_names), 4),
        "k_convergence_rate": round(k_convergence_rate(state), 6),
        "k_threshold": boot_cfg.k_threshold,
        "coverage_target": boot_cfg.coverage_target,
        "disagreements": len(disagreements),
        "llm_calls_total": state.llm_calls_total,
        "max_llm_calls": boot_cfg.max_total_llm_calls,
        "truncation_count": state.truncation_count,
        "effective_batch_size": state.effective_batch_size,
        "iteration_history": [
            {
                "iteration": m.iteration,
                "mean_k": m.mean_k,
                "mean_gap": m.mean_gap,
                "mean_bel": m.mean_bel,
                "frac_unclear": m.frac_unclear,
                "disagreements": m.disagreements,
                "coverage": m.coverage,
                # Numerical-methods diagnostics (Saad 2003 §4.1):
                # ‖r‖ unified residual norm; ρ contraction factor.
                "residual_norm": m.residual_norm,
                "contraction_rate": m.contraction_rate,
                "indep_tier_disagreement_frac": m.indep_tier_disagreement_frac,
            }
            for m in state.iteration_metrics
        ],
    }


def _handle_get_column_detail(
    state: BootstrapState,
    col_name: str,
    samples: dict[str, ColumnSample],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    boot_cfg: BootstrapConfig,
    discounts: DiscountConfig | None = None,
) -> dict[str, Any]:
    """Deep-dive: re-run _classify_column and return full evidence breakdown."""
    col = samples.get(col_name)
    if not col:
        return {"error": f"Column '{col_name}' not found in samples"}

    from atelier.classify.pipeline import _classify_column

    llm_code = state.labels.get(col_name)
    llm_conf = state.confidence.get(col_name, 0.0)

    result = _classify_column(
        col, category_set, frame,
        llm_code=llm_code,
        llm_confidence=llm_conf,
        llm_discount=boot_cfg.llm_discount,
        use_cosine=True,
        discounts=discounts,
    )

    return {
        "column_name": col_name,
        "table_name": col.table_name,
        "column_type": col.column_type,
        "sample_values": col.values[:10],
        "predicted_code": result.get("predicted_code"),
        "predicted_label": result.get("predicted_label"),
        "confidence": result.get("confidence"),
        "belief": result.get("belief"),
        "plausibility": result.get("plausibility"),
        "conflict": result.get("conflict"),
        "evidence_sources": result.get("evidence_sources", {}),
        "belief_path": result.get("belief_path", []),
        "pattern_signals": result.get("pattern_signals", []),
        "llm_label": state.labels.get(col_name),
        "ml_prediction": state.ml_prediction.get(col_name),
    }


def _handle_declare_converged(
    state: BootstrapState,
    reason: str,
    convergence_kind: str = "agent_convergence",
    *,
    boot_cfg: BootstrapConfig | None = None,
) -> dict[str, Any]:
    """Record convergence reason and signal loop exit.

    The ``min_iterations`` directive is enforced primarily by the
    pipeline-side fallback in
    :func:`atelier.classify.pipeline.run_classification_pipeline`,
    which runs additional programmatic revisits after the agent loop
    returns if ``state.iteration`` is below the floor.  The tool here
    just accepts the agent's declaration and records the structured
    tag + prose; that keeps the agent loop bounded by ``max_turns``
    regardless of how the directive interacts with the corpus's
    natural convergence.

    A tool-side rejection (the original Tier 1A design) created an
    unwinnable loop on small corpora where the predictions naturally
    settle in one sweep — the agent has nothing meaningful to revisit
    yet the gate refuses to let it declare.  Letting the pipeline
    handle the directive avoids that pathology while still
    guaranteeing the directive holds.
    """
    state.agent_reasoning.append(f"CONVERGED [{convergence_kind}]: {reason}")
    state.agent_converged_reason = reason
    state.agent_converged_tag = convergence_kind
    return {
        "converged": True,
        "convergence_kind": convergence_kind,
        "reason": reason,
    }


# ── Client builder ───────────────────────────────────────────────


def _build_client(cfg: AtelierConfig):
    """Build Anthropic or Bedrock client for agent loop."""
    if cfg.has_anthropic:
        from atelier.agents.client import _build_anthropic_client
        return _build_anthropic_client(cfg)
    if cfg.has_bedrock:
        from atelier.agents.client import _build_bedrock_client
        return _build_bedrock_client(cfg)
    raise RuntimeError("No Anthropic or Bedrock credentials configured")


def _agent_model(cfg: AtelierConfig) -> str:
    """Resolve the model ID for the agent. Fallback chain:
    classify.agent.model → agents.model → default Sonnet.
    """
    return (
        cfg.classify_agent_model
        or cfg.agent_model
        or "claude-sonnet-4-5-20250929"
    )


# ── Initial state formatting ─────────────────────────────────────


def _format_initial_state(
    state: BootstrapState,
    column_names: list[str],
    boot_cfg: BootstrapConfig,
) -> str:
    """Summarize current state for the agent's first turn."""
    from atelier.classify.bootstrap import (
        _coverage,
        _mean_gap,
        _mean_k,
        _max_k,
        _identify_disagreements,
    )

    disagreements = _identify_disagreements(state, column_names, boot_cfg)
    coverage = _coverage(state, column_names)
    mean_gap = _mean_gap(state, column_names)
    mean_k = _mean_k(state, column_names)
    max_k = _max_k(state, column_names)

    return (
        f"Initial classification pass complete.\n\n"
        f"## Current State\n"
        f"- Total columns: {len(column_names)}\n"
        f"- Labeled: {len(state.labels)}\n"
        f"- Coverage: {coverage:.1%}\n"
        f"- Mean belief gap (Pl − Bel): {mean_gap:.4f}  ← primary signal\n"
        f"- Mean conflict K: {mean_k:.4f}  (diagnostic)\n"
        f"- Max conflict K: {max_k:.4f}  (diagnostic)\n"
        f"- Disagreements (LLM vs ML, K > {boot_cfg.k_threshold}): "
        f"{len(disagreements)}\n"
        f"- LLM calls so far: {state.llm_calls_total}\n"
        f"- Budget: {boot_cfg.max_total_llm_calls} max calls\n\n"
        f"## Convergence Targets\n"
        f"- Coverage target: {boot_cfg.coverage_target:.0%}\n"
        f"- Gap threshold (primary): {boot_cfg.gap_threshold}\n"
        f"- Bel floor: {boot_cfg.bel_floor}\n\n"
        f"Investigate weakly-supported columns (high gap, low Bel) and "
        f"work toward convergence on the belief-gap criterion."
    )


# ── Tool dispatch ────────────────────────────────────────────────


def _dispatch_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    state: BootstrapState,
    boot_cfg: BootstrapConfig,
    cfg: AtelierConfig,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    backend: LLMBackend | None,
    system_prompt: str,
    has_embeddings: bool,
    discounts: DiscountConfig | None,
) -> dict[str, Any]:
    """Route a tool_use block to the appropriate handler."""
    if tool_name == "get_conflict_report":
        return _handle_get_conflict_report(
            state, column_names, category_set,
            k_threshold=tool_input.get("k_threshold", boot_cfg.k_threshold),
        )

    elif tool_name == "revisit_columns":
        cols = tool_input.get("column_names", [])
        # Filter to columns that actually exist
        valid_cols = [c for c in cols if c in samples]
        if not valid_cols:
            return {"error": "No valid column names provided", "requested": cols}
        if backend is None:
            return {"error": "No LLM backend available for revisit"}
        return _handle_revisit_columns(
            state, boot_cfg, backend, system_prompt,
            valid_cols, column_names, samples, column_table,
            category_set, frame, has_embeddings, discounts,
        )

    elif tool_name == "check_convergence":
        return _handle_check_convergence(state, column_names, boot_cfg)

    elif tool_name == "get_column_detail":
        col_name = tool_input.get("column_name", "")
        return _handle_get_column_detail(
            state, col_name, samples, category_set,
            frame, boot_cfg, discounts,
        )

    elif tool_name == "declare_converged":
        return _handle_declare_converged(
            state,
            reason=tool_input.get("reason", ""),
            convergence_kind=tool_input.get("convergence_kind", "agent_convergence"),
            boot_cfg=boot_cfg,
        )

    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ── Agent conversation loop ──────────────────────────────────────


def run_agent_loop(
    state: BootstrapState,
    cfg: AtelierConfig,
    boot_cfg: BootstrapConfig,
    backend: LLMBackend | None,
    system_prompt: str,
    column_names: list[str],
    samples: dict[str, ColumnSample],
    column_table: dict[str, str],
    category_set: HierarchicalCategorySet,
    frame: FrameOfDiscernment,
    has_embeddings: bool = True,
    discounts: DiscountConfig | None = None,
    *,
    on_event: Callable[[dict], None] | None = None,
    client=None,
) -> bool:
    """Agent-driven convergence loop using Messages API with tool_use.

    Args:
        state: Mutable bootstrap state (labels, conflict, metrics).
        cfg: Application config (credentials, model selection).
        boot_cfg: Bootstrap convergence parameters.
        backend: LLM backend for revisit calls.
        system_prompt: Classification system prompt for revisit.
        column_names: All column names in the dataset.
        samples: Column samples by name.
        column_table: Column name → table name mapping.
        category_set: Hierarchical category vocabulary.
        frame: Frame of discernment for DST.
        has_embeddings: Whether cosine similarity is available.
        discounts: DST discount configuration.
        on_event: Optional callback for WebSocket event streaming.
        client: Optional pre-built Anthropic client (for testing).

    Returns:
        True if the agent declared convergence.
    """
    if client is None:
        client = _build_client(cfg)

    model = _agent_model(cfg)
    max_turns = cfg.classify_agent_max_turns
    messages: list[dict[str, Any]] = []
    converged = False

    # Initial prompt
    initial_state = _format_initial_state(state, column_names, boot_cfg)
    messages.append({"role": "user", "content": initial_state})

    logger.info(
        "Starting agent loop: model=%s, max_turns=%d, columns=%d",
        model, max_turns, len(column_names),
    )

    if on_event:
        on_event({
            "type": "agent_spawned",
            "agent": "classify",
            "model": model,
            "columns": len(column_names),
        })

    for turn in range(max_turns):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=AGENT_SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            logger.error("Agent API call failed on turn %d: %s", turn, e)
            state.agent_reasoning.append(f"API error on turn {turn}: {e}")
            break

        # Track tokens
        state.tokens_input += response.usage.input_tokens
        state.tokens_output += response.usage.output_tokens
        state.agent_turns = turn + 1

        # Process response content blocks
        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Extract reasoning text
        for block in assistant_content:
            if hasattr(block, "type") and block.type == "text":
                state.agent_reasoning.append(block.text)
                if on_event:
                    on_event({
                        "type": "agent_reasoning",
                        "text": block.text,
                        "turn": turn,
                    })

        # Handle tool calls
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in assistant_content:
                if not (hasattr(block, "type") and block.type == "tool_use"):
                    continue

                logger.info("Agent turn %d: tool=%s", turn, block.name)

                result = _dispatch_tool(
                    block.name, block.input,
                    state, boot_cfg, cfg, column_names, samples,
                    column_table, category_set, frame,
                    backend, system_prompt, has_embeddings, discounts,
                )

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                })

                if on_event:
                    on_event({
                        "type": "agent_tool_call",
                        "tool": block.name,
                        "turn": turn,
                    })

                if block.name == "declare_converged":
                    converged = bool(result.get("converged"))

            messages.append({"role": "user", "content": tool_results})

            if converged:
                break
        else:
            # end_turn without tool call — agent is done reasoning
            logger.info("Agent ended reasoning on turn %d (stop_reason=%s)",
                        turn, response.stop_reason)
            break

    # If we exhausted turns without explicit convergence, record why
    if not converged and state.agent_converged_reason is None:
        state.agent_converged_reason = (
            f"Turn limit reached ({max_turns} turns) without explicit convergence"
        )

    logger.info(
        "Agent loop finished: turns=%d, converged=%s, reason=%s",
        state.agent_turns, converged, state.agent_converged_reason,
    )

    if on_event:
        on_event({
            "type": "agent_completed",
            "agent": "classify",
            "turns": state.agent_turns,
            "converged": converged,
            "reason": state.agent_converged_reason,
        })

    return converged
