"""Cautious-Code Review — agent-mediated backoff for over-specified predictions.

When the DST fusion produces a deeper code than the cautious-belief
threshold supports, the over-specified code may be wrong.  Overwatch's
analysis of run ``a0f80287`` named this Seam A: 12 errors clustered on
the curator-provides-parent / pipeline-predicts-shipping-child split
(``city → Shipping City``, ``country → Shipping Country``,
``phone_number → Other Phone Number``, etc.).

This module asks Claude (via the same LLM backend the classify pipeline
uses, or optionally direct Anthropic for reasoning-budget access) to
consider whether to back off to the cautious code given the column's
values + sibling context.  The agent's decision and rationale are
recorded for audit: ``predicted_code`` is updated in place when backoff
is recommended, the original is preserved as
``predicted_code_pre_review``, and ``review_decision`` /
``review_rationale`` are stamped on the classification dict and into
``seam_a_review.json`` beside the run parquet.

Project directive — on by default.  Iteration is part of the algorithm,
and this is one of its iterations.  Toggle off only for ablation
windows (compare-with-and-without).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are reviewing a column-classification decision from an "
    "evidence-fusion pipeline (Atelier).  The pipeline assigns each "
    "column a hierarchical taxonomy code from belief-mass fusion across "
    "six evidence sources (name match, value patterns, cosine similarity, "
    "an LLM sweep, CatBoost, SVM).  Sometimes the fused decision commits "
    "to a deeper code (e.g. 'Shipping City' = 1.1.1.4.2.2.2) when the "
    "evidence really only supports the parent (e.g. 'City' = "
    "1.1.1.4.2.2).  Your job is to decide:\n"
    "  - keep — the deeper code is genuinely supported by value content "
    "and sibling-table context\n"
    "  - backoff — the deeper code over-specifies; use the cautious "
    "(parent-or-shallower) code instead\n"
    "Reply with strict JSON matching the schema "
    '{"decision": "keep" | "backoff", "rationale": "<1-3 sentence cite '
    'of specific evidence>", "confidence": <0.0-1.0>}.  No other prose.'
)


def _format_belief_path(path: list[dict]) -> str:
    """Render a belief path leaf→root for the prompt."""
    if not path:
        return "(no belief path available)"
    lines = []
    for entry in path:
        code = entry.get("code", "")
        label = entry.get("label", "")
        bel = entry.get("bel", 0.0)
        pl = entry.get("pl", 0.0)
        lines.append(f"  {code} ({label}): Bel={bel:.3f}, Pl={pl:.3f}")
    return "\n".join(lines)


def _build_review_prompt(column: dict, sibling_predictions: list[dict]) -> str:
    """Build the per-column review prompt."""
    lines = [
        f"Column: {column.get('table_name','')}.{column.get('column_name','')}",
        f"Type: {column.get('column_type', 'unknown')}",
        "",
        "Current prediction (from DST fusion):",
        f"  predicted_code:  {column.get('predicted_code', '')}",
        f"  predicted_label: {column.get('predicted_label', '')}",
        f"  belief:          {float(column.get('belief', 0)):.3f}",
        f"  plausibility:    {float(column.get('plausibility', 0)):.3f}",
        f"  uncertainty gap: {float(column.get('uncertainty', 0)):.3f}",
        "",
        "Cautious-code alternative (deepest code where Bel >= 0.70):",
        f"  cautious_code: {column.get('cautious_code', '')}",
        "",
        "Belief path (leaf → root; Bel increases ascending the hierarchy):",
        _format_belief_path(column.get("belief_path", [])),
    ]

    sigs = column.get("pattern_signals") or {}
    if sigs:
        lines.append("")
        lines.append(f"Pattern signals: {sigs}")

    embedding_text = column.get("embedding_text", "")
    if embedding_text:
        lines.append("")
        excerpt = embedding_text[:500]
        if len(embedding_text) > 500:
            excerpt += "..."
        lines.append(f"Value evidence (excerpt): {excerpt}")

    if sibling_predictions:
        lines.append("")
        lines.append(
            f"Sibling columns in {column.get('table_name','')} "
            f"(at most 8 shown):"
        )
        for sib in sibling_predictions[:8]:
            lines.append(
                f"  {sib.get('column_name','')} → {sib.get('predicted_code','')} "
                f"({sib.get('predicted_label','')})"
            )

    lines.extend([
        "",
        "Question: Is the predicted_code over-specified for what the "
        "column's value evidence supports?  If the deeper code introduces "
        "context (e.g. 'Shipping' City vs generic 'City') that the values "
        "don't actually evidence, recommend backoff to cautious_code.  If "
        "the deeper code is genuinely supported by the values + sibling "
        "table domain, keep it.",
    ])
    return "\n".join(lines)


def _parse_decision(text: str) -> dict[str, Any]:
    """Extract a structured decision from agent output."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    try:
        decision = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON ({e}) in: {text[:200]!r}")
    if decision.get("decision") not in ("keep", "backoff"):
        raise ValueError(
            f"decision must be 'keep' or 'backoff', got {decision.get('decision')!r}"
        )
    decision.setdefault("rationale", "(none)")
    decision.setdefault("confidence", None)
    return decision


def _resolve_client(cfg, backend_choice: str):
    """Build the Anthropic-compatible client + model for the review.

    Three modes:
      - ``default`` — reuse whichever provider classify is using
        (Bedrock when ``classify_subagent_model`` is an ARN, else
        direct Anthropic).
      - ``anthropic_direct`` — force direct Anthropic API (for
        Opus 4.7 reasoning-budget access; requires
        ``ANTHROPIC_API_KEY``).
      - ``bedrock`` — force Bedrock.
    Returns ``(client, model_id)``.
    """
    import anthropic
    import httpx
    from atelier.config import is_bedrock_model

    timeout = httpx.Timeout(connect=15.0, read=180.0, write=10.0, pool=5.0)

    if backend_choice == "anthropic_direct":
        if not cfg.anthropic_api_key:
            raise ValueError(
                "seam_a_review backend=anthropic_direct requires ANTHROPIC_API_KEY"
            )
        client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=timeout)
        # Direct Anthropic prefers cfg.agent_model (Opus 4.7 family)
        model = cfg.agent_model
        return client, model

    # default or bedrock — figure out which is configured
    classify_model = cfg.classify_subagent_model or cfg.agent_model
    if backend_choice == "bedrock" or (
        backend_choice == "default" and is_bedrock_model(classify_model)
    ):
        from atelier.agents.client import _build_bedrock_client
        client = _build_bedrock_client(cfg, timeout=180.0)
        return client, classify_model

    # Default with non-Bedrock model → Anthropic direct
    if not cfg.anthropic_api_key:
        raise ValueError(
            "seam_a_review default backend chose Anthropic but "
            "ANTHROPIC_API_KEY is not set"
        )
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key, timeout=timeout)
    return client, classify_model


def _invoke(client, model: str, prompt: str) -> dict[str, Any]:
    """Invoke the review agent and parse the JSON decision.

    Note: Opus 4.7+ deprecated the ``temperature`` parameter, so we
    don't pass it.  The model is deterministic enough at default
    settings for the binary keep/backoff decision.
    """
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    blocks = getattr(resp, "content", None) or []
    text = ""
    for b in blocks:
        if getattr(b, "type", "") == "text":
            text += getattr(b, "text", "")
    if not text:
        raise ValueError("agent returned no text content")
    return _parse_decision(text)


def review_classifications(
    classifications: list[dict[str, Any]],
    cfg,
    *,
    category_set=None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Run the Cautious-Code Review pass on a fused classification set.

    Mutates ``classifications`` in place when the agent recommends
    backoff: ``predicted_code`` becomes ``cautious_code``, the original
    is preserved as ``predicted_code_pre_review``, and ``review_decision``
    / ``review_rationale`` are stamped on the column.

    Returns an audit dict suitable for writing as
    ``seam_a_review.json`` beside the run's other artifacts.
    """
    if not getattr(cfg, "classify_seam_a_review_enabled", True):
        return {"enabled": False}

    bel_threshold = float(
        getattr(cfg, "classify_seam_a_review_bel_threshold", 0.85)
    )
    backend_choice = getattr(cfg, "classify_seam_a_review_backend", "default")

    # Identify candidates: cautious differs from predicted AND belief is
    # below threshold (the deep code isn't strongly supported).
    candidates: list[dict] = []
    for c in classifications:
        predicted = c.get("predicted_code") or ""
        cautious = c.get("cautious_code") or ""
        bel = float(c.get("belief", 1.0))
        if predicted and cautious and predicted != cautious and bel < bel_threshold:
            candidates.append(c)

    if not candidates:
        logger.info(
            "Seam A review: 0 candidates (bel_threshold=%.2f) — no review needed",
            bel_threshold,
        )
        return {
            "enabled": True,
            "bel_threshold": bel_threshold,
            "backend_choice": backend_choice,
            "candidates": 0,
            "decisions": [],
        }

    # Build sibling map for context
    by_table: dict[str, list[dict]] = {}
    for c in classifications:
        by_table.setdefault(c.get("table_name", ""), []).append(c)

    try:
        client, model = _resolve_client(cfg, backend_choice)
    except Exception as exc:
        logger.warning(
            "Seam A review skipped — could not resolve LLM client: %s", exc,
        )
        return {
            "enabled": True,
            "bel_threshold": bel_threshold,
            "backend_choice": backend_choice,
            "candidates": len(candidates),
            "skipped": True,
            "skip_reason": str(exc),
            "decisions": [],
        }

    logger.info(
        "Seam A review: %d candidates (bel_threshold=%.2f, model=%s)",
        len(candidates), bel_threshold, model,
    )

    decisions: list[dict] = []
    for i, col in enumerate(candidates):
        siblings = [
            s for s in by_table.get(col.get("table_name", ""), [])
            if s.get("column_name") != col.get("column_name")
        ]
        prompt = _build_review_prompt(col, siblings)

        if progress_callback:
            try:
                progress_callback({
                    "phase": "seam_a_review",
                    "candidate_index": i,
                    "candidates_total": len(candidates),
                    "column": (
                        f"{col.get('table_name','')}."
                        f"{col.get('column_name','')}"
                    ),
                })
            except Exception:
                pass

        try:
            decision = _invoke(client, model, prompt)
        except Exception as exc:
            logger.warning(
                "Seam A review failed for %s.%s: %s — keeping predicted_code",
                col.get("table_name", ""), col.get("column_name", ""), exc,
            )
            col["review_decision"] = "error"
            col["review_rationale"] = f"agent invocation failed: {exc}"[:300]
            decisions.append({
                "column": col.get("column_name", ""),
                "table": col.get("table_name", ""),
                "decision": "error",
                "rationale": col["review_rationale"],
                "predicted_code": col.get("predicted_code", ""),
                "cautious_code": col.get("cautious_code", ""),
            })
            continue

        col["review_decision"] = decision["decision"]
        col["review_rationale"] = decision["rationale"][:500]

        if decision["decision"] == "backoff":
            col["predicted_code_pre_review"] = col.get("predicted_code", "")
            col["predicted_code"] = col.get("cautious_code", "")
            # Refresh predicted_label from the category_set, if available.
            if category_set is not None:
                cautious_cat = (
                    getattr(category_set, "all_by_code", {}).get(col["predicted_code"])
                    or getattr(category_set, "by_code", {}).get(col["predicted_code"])
                )
                if cautious_cat is not None:
                    col["predicted_label"] = getattr(cautious_cat, "label", col["predicted_code"])
                    col["predicted_annotation"] = getattr(cautious_cat, "abbrev", "") or ""
            # Update matches_reference if a reference exists
            ref = col.get("reference_code")
            if ref:
                col["matches_reference"] = (col["predicted_code"] == ref)

        decisions.append({
            "column": col.get("column_name", ""),
            "table": col.get("table_name", ""),
            "decision": decision["decision"],
            "rationale": decision.get("rationale", ""),
            "confidence": decision.get("confidence"),
            "predicted_code": col.get("predicted_code", ""),
            "cautious_code": col.get("cautious_code", ""),
            "predicted_code_pre_review": col.get("predicted_code_pre_review"),
        })

    backed_off = sum(1 for d in decisions if d.get("decision") == "backoff")
    kept = sum(1 for d in decisions if d.get("decision") == "keep")
    errored = sum(1 for d in decisions if d.get("decision") == "error")

    logger.info(
        "Seam A review complete: %d/%d backed off, %d kept, %d errored",
        backed_off, len(decisions), kept, errored,
    )

    return {
        "enabled": True,
        "bel_threshold": bel_threshold,
        "backend_choice": backend_choice,
        "model": model,
        "candidates": len(candidates),
        "backed_off": backed_off,
        "kept": kept,
        "errored": errored,
        "decisions": decisions,
    }
