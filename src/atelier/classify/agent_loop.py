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
            "won't improve results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why convergence is declared",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "retrain_svm",
        "description": (
            "Retrain the SVM classifier on blended synthetic + frontier LLM "
            "labels. Call when you judge enough new frontier labels have "
            "accumulated to improve the SVM's accuracy. The retrained SVM "
            "will be used in subsequent ML validation passes. Returns the "
            "number of frontier labels used and training statistics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


AGENT_SYSTEM_PROMPT = """\
You are the Atelier classification agent. Your goal is to classify data columns
with high confidence using Dempster-Shafer Theory (DST) evidence fusion.

You have already completed an initial classification pass. Now you are in the
convergence loop: examining columns where evidence sources disagree (high
conflict K), investigating why, and requesting targeted re-classification
with enriched context.

## Your Tools
- get_conflict_report: See which columns have high K (source disagreement)
- get_column_detail: Deep-dive into a specific column's evidence breakdown
- revisit_columns: Re-classify selected columns with enriched context
- check_convergence: See overall metrics (coverage, mean K, trend)
- declare_converged: Stop when further iteration won't help
- retrain_svm: Retrain the SVM on accumulated frontier labels (blended with
  synth data). Call after revisiting several batches to let the SVM learn
  from the latest classifications.

## Strategy
1. Start by checking the conflict report — which columns need attention?
2. For high-K columns, examine the evidence detail — do sources agree on the
   category family but disagree on the specific leaf? That's a confusable pair.
3. Prioritize revisiting columns where pattern evidence or name matching
   provides strong signal that the LLM may have missed.
4. After each revisit batch, check convergence — is mean K decreasing?
5. Declare converged when: mean_K < threshold, or K has plateaued, or
   remaining disagreements are confusable pairs (expected ambiguity).

## Key Insight
High conflict K is SIGNAL, not error. When cosine similarity says "EMAIL" but
the LLM says "USERNAME", that's a column worth investigating — not discarding.
"""


# ── Tool handlers ────────────────────────────────────────────────


def _handle_get_conflict_report(
    state: BootstrapState,
    column_names: list[str],
    category_set: HierarchicalCategorySet,
    k_threshold: float = 0.2,
) -> dict[str, Any]:
    """Format BootstrapState conflict data for the agent."""
    high_k = []
    for name in column_names:
        k = state.ml_conflict.get(name, 0.0)
        if k <= k_threshold:
            continue

        llm_code = state.labels.get(name, "")
        ml_code = state.ml_prediction.get(name, "")

        llm_cat = category_set.by_code.get(llm_code) or category_set.all_by_code.get(llm_code)
        ml_cat = category_set.by_code.get(ml_code) or category_set.all_by_code.get(ml_code)

        high_k.append({
            "column_name": name,
            "conflict_K": round(k, 4),
            "llm_prediction": llm_cat.label if llm_cat else llm_code,
            "llm_code": llm_code,
            "ml_prediction": ml_cat.label if ml_cat else ml_code,
            "ml_code": ml_code,
            "disagrees": llm_code != ml_code,
        })

    high_k.sort(key=lambda x: -x["conflict_K"])
    return {
        "total_columns": len(column_names),
        "high_k_count": len(high_k),
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
    record_iteration_metrics(state, all_column_names, len(disagreements))

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
        k_convergence_rate,
        _identify_disagreements,
    )

    disagreements = _identify_disagreements(state, column_names, boot_cfg)

    return {
        "iteration": state.iteration,
        "coverage": round(_coverage(state, column_names), 4),
        "mean_k": round(_mean_k(state, column_names), 4),
        "max_k": round(_max_k(state, column_names), 4),
        "k_convergence_rate": round(k_convergence_rate(state), 6),
        "k_threshold": boot_cfg.k_threshold,
        "coverage_target": boot_cfg.coverage_target,
        "disagreements": len(disagreements),
        "llm_calls_total": state.llm_calls_total,
        "max_llm_calls": boot_cfg.max_total_llm_calls,
        "iteration_history": [
            {
                "iteration": m.iteration,
                "mean_k": m.mean_k,
                "disagreements": m.disagreements,
                "coverage": m.coverage,
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


def _handle_retrain_svm(
    state: BootstrapState,
    samples: dict[str, ColumnSample],
    boot_cfg: BootstrapConfig,
    cfg: AtelierConfig,
) -> dict[str, Any]:
    """Retrain SVM on blended synth + frontier labels, hot-swap."""
    from pathlib import Path
    from atelier.classify import ml_inference
    from atelier.classify.ml_train import train_svm_on_frontier_labels

    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    results_dir = _project_root / "build" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    svm_path = results_dir / "svm_frontier_agent.pkl"

    synth_dir = _project_root / "build" / "data" / "synth"
    result = train_svm_on_frontier_labels(
        state, samples, svm_path,
        synth_dir=synth_dir if synth_dir.exists() else None,
        min_frontier_labels=boot_cfg.frontier_svm_min_labels,
    )

    if result is None:
        frontier_count = sum(
            1 for src in state.label_source.values()
            if src in ("llm", "llm_revisit")
        )
        return {
            "retrained": False,
            "reason": (
                f"Below threshold: {frontier_count} frontier labels "
                f"(min={boot_cfg.frontier_svm_min_labels})"
            ),
        }

    # Hot-swap
    ml_inference.reset()
    ml_inference.configure_paths(
        svm_path=str(svm_path),
        catboost_path=cfg.classify_catboost_model_path,
    )

    frontier_count = sum(
        1 for src in state.label_source.values()
        if src in ("llm", "llm_revisit")
    )
    state.svm_retrain_count = frontier_count
    state.svm_frontier_path = str(svm_path)

    return {
        "retrained": True,
        "frontier_samples": frontier_count,
        "total_labels": len(state.labels),
        "classes": len(set(
            code for name, code in state.labels.items()
            if state.label_source.get(name) in ("llm", "llm_revisit")
        )),
        "model_path": str(svm_path),
    }


def _handle_declare_converged(
    state: BootstrapState,
    reason: str,
) -> dict[str, Any]:
    """Record convergence reason and signal loop exit."""
    state.agent_reasoning.append(f"CONVERGED: {reason}")
    state.agent_converged_reason = reason
    return {"converged": True, "reason": reason}


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
        _mean_k,
        _max_k,
        _identify_disagreements,
    )

    disagreements = _identify_disagreements(state, column_names, boot_cfg)
    coverage = _coverage(state, column_names)
    mean_k = _mean_k(state, column_names)
    max_k = _max_k(state, column_names)

    return (
        f"Initial classification pass complete.\n\n"
        f"## Current State\n"
        f"- Total columns: {len(column_names)}\n"
        f"- Labeled: {len(state.labels)}\n"
        f"- Coverage: {coverage:.1%}\n"
        f"- Mean conflict K: {mean_k:.4f}\n"
        f"- Max conflict K: {max_k:.4f}\n"
        f"- Disagreements (LLM vs ML, K > {boot_cfg.k_threshold}): "
        f"{len(disagreements)}\n"
        f"- LLM calls so far: {state.llm_calls_total}\n"
        f"- Budget: {boot_cfg.max_total_llm_calls} max calls\n\n"
        f"## Convergence Targets\n"
        f"- Coverage target: {boot_cfg.coverage_target:.0%}\n"
        f"- K threshold: {boot_cfg.k_threshold}\n\n"
        f"Please investigate the conflict report and work toward convergence."
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
        return _handle_declare_converged(state, tool_input.get("reason", ""))

    elif tool_name == "retrain_svm":
        return _handle_retrain_svm(state, samples, boot_cfg, cfg)

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
                    converged = True

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
