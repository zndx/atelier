# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""LLM backend abstraction for bootstrap classification.

Provides a unified interface for LLM-based column classification:
- Anthropic Structured (Claude) — default when ANTHROPIC_API_KEY is
  available.  Uses the Messages API ``output_config`` with JSON Schema
  for guaranteed valid structured output (direct API only), plus prompt
  caching for the taxonomy system prompt.
- Anthropic (Claude) — free-text fallback via the anthropic SDK
- OpenAI-compatible (vLLM/GLM-4.7) via the openai SDK
- Cerebras (GLM-4.7 via OpenAI-compatible API)
- AWS Bedrock (Claude via invoke_model + tool-use for structured output)

Each backend converts batch column metadata into structured
classification responses with token tracking for cost estimation.

Ported from signals/src/sigint/llm_backend.py, adapted for atelier's
HOCON config and ColumnSample type.
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Anthropic structured output constants ────────────────────────

# JSON Schema for structured classification output.  The SDK enforces
# this via constrained decoding — no regex parsing needed.
_CLASSIFICATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "column_name": {"type": "string"},
        "category_code": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"type": "number"},
        "evidence": {"type": "string"},
        "alternatives": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["code", "confidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["column_name", "category_code", "confidence", "evidence", "alternatives"],
    "additionalProperties": False,
}

CLASSIFICATION_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "classifications": {
            "type": "array",
            "items": _CLASSIFICATION_ITEM_SCHEMA,
        },
    },
    "required": ["classifications"],
    "additionalProperties": False,
}


# ── Cerebras constants ──────────────────────────────────────────

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1"
CEREBRAS_DEFAULT_MODEL = "zai-glm-4.7"


# ── Response types ───────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnClassification:
    """Single column classification from an LLM."""

    column_name: str
    category_code: str | None
    confidence: float
    evidence: str
    alternatives: list[dict] = field(default_factory=list)  # [{code, confidence}, ...]


@dataclass(frozen=True)
class LLMResponse:
    """Batch response from an LLM backend."""

    classifications: list[ColumnClassification]
    input_tokens: int
    output_tokens: int
    model: str
    finish_reason: str = "stop"
    # Reasoning/thinking-trace text when the backend + model expose one
    # (e.g. GLM-4.7 via Cerebras with ``reasoning_format="parsed"``,
    # which returns ``choices[0].message.reasoning`` separate from
    # ``content``).  Empty string for backends without reasoning
    # capture or for models that don't emit thinking text.  Stored as a
    # single string per batch — one reasoning trace covers the whole
    # batch's decisions, not per-column.
    reasoning_text: str = ""
    reasoning_tokens: int = 0

    # Whether fewer classifications came back than were requested, even
    # when the backend reported a clean stop.  This happens when a
    # model — particularly Bedrock-hosted Claude with its tighter output
    # ceiling — produces an abbreviated response that parses cleanly as
    # JSON but simply omits some requested columns.  The pipeline treats
    # this as truncation so halving retry engages on the missing rows
    # rather than silently dropping them.
    partial: bool = False

    @property
    def truncated(self) -> bool:
        """Whether the response must be retried for coverage.

        Triggers on (a) explicit finish_reason == length/max_tokens, or
        (b) a partial parse where the backend returned a clean stop but
        produced fewer classifications than expected columns.
        """
        return self.finish_reason in ("length", "max_tokens") or self.partial


# ── Configuration ────────────────────────────────────────────────


def _effort_from_budget(budget: int) -> str:
    """Map a legacy thinking-budget token count to the new effort level.

    Opus 4.7's direct API replaced ``thinking.type=enabled`` +
    ``budget_tokens`` with ``thinking.type=adaptive`` +
    ``output_config.effort`` (low / medium / high / max). The mapping
    here preserves the intent of our existing ``reasoning_budget``
    config knob without requiring operators to learn the new axis.
    """
    if budget <= 4096:
        return "low"
    if budget <= 16384:
        return "medium"
    if budget <= 32768:
        return "high"
    return "max"


def _apply_thinking(kwargs: dict, model: str, budget: int) -> None:
    """Mutate *kwargs* to include a thinking config compatible with *model*.

    Opus 4.7+ (and future Opus ≥ 5) reject the legacy
    ``{type:"enabled", budget_tokens:N}`` pair — they require
    ``{type:"adaptive"}`` alongside ``output_config.effort``.  Opus 4.6
    (still on Bedrock) and earlier accept the legacy shape and are
    incompatible with the adaptive route on current deployments.

    Both routes set ``temperature=1`` (the thinking prerequisite).  The
    adaptive route merges ``effort`` into any pre-existing
    ``output_config`` dict so structured-output backends (which already
    use ``output_config.format``) keep their schema setting intact.
    """
    from atelier.model_compat import requires_adaptive_thinking

    if requires_adaptive_thinking(model):
        kwargs["thinking"] = {"type": "adaptive"}
        oc = kwargs.setdefault("output_config", {})
        oc["effort"] = _effort_from_budget(budget)
    else:
        kwargs["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
    kwargs["temperature"] = 1


class ThrottledError(Exception):
    """Backend rate-limited the request after exhausting internal retries.

    Raised by ``BedrockStructuredBackend`` and ``OpenAICompatibleBackend``
    when their inner backoff loop runs out of attempts on a throttling
    code (HTTP 429, AWS ``ThrottlingException`` /
    ``TooManyRequestsException`` / ``ServiceUnavailableException``,
    HTTP 503/529).  The bootstrap halving-retry loop catches this
    distinctly from other recoverable errors and **sleeps with jitter at
    the same batch size** instead of halving — halving under throttling
    multiplies concurrent calls at smaller sizes, making throttling
    worse.

    Carries the underlying exception (``cause``) and the suggested
    retry-after seconds (``retry_after_s``, optional) so the outer
    retry loop can honor the backend's pacing hint when present.
    """

    def __init__(
        self,
        message: str,
        *,
        cause: Exception | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(message)
        self.cause = cause
        self.retry_after_s = retry_after_s


@dataclass
class LLMBackendConfig:
    """Configuration for LLM backend."""

    backend: str = "openai_compatible"  # "anthropic" | "openai_compatible" | "cerebras" | "bedrock"
    api_key: str | None = None
    model: str = "glm-4.7"
    base_url: str | None = None  # e.g. "http://localhost:8000/v1"
    max_tokens: int = 65536
    temperature: float = 0.0
    batch_size: int = 10  # columns per LLM call
    max_retries: int = 3
    retry_delay: float = 2.0
    disable_reasoning: bool = False
    # Chain-of-thought token budget (Anthropic thinking mode, Cerebras
    # reasoning models, reasoning-aware vLLM builds).  0 = don't send the
    # field at all — the default, so stock OpenAI-compatible endpoints
    # (e.g. plain GLM-4.7 on vLLM) won't 422 on the unknown property.
    # Set via ATELIER_LLM_REASONING_BUDGET when the backend supports it.
    reasoning_budget: int = 0
    # AWS Bedrock credentials (used only by "bedrock" backend)
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region: str | None = None
    aws_session_token: str | None = None


# Bedrock's output-token ceiling is model-specific and silently
# enforced by the runtime — a request with maxTokens=65536 against
# claude-3-5-sonnet gets clamped to 4096 with no warning in the
# response, which is how CAI's LLM-sweep was silently dropping ~25%
# of columns per batch.  This table pins a safe upper bound per model
# family so the batch sizer can be honest about how many columns will
# actually fit in one call.  Keys are substring-matched against the
# model ID or inference-profile ARN.  The floor of 4096 is the legacy
# Bedrock default; anything not matched falls back to that.
_BEDROCK_MODEL_OUTPUT_CEILING: tuple[tuple[str, int], ...] = (
    ("claude-opus-4",            32000),
    ("claude-sonnet-4",          64000),
    ("claude-haiku-4",           64000),
    ("claude-3-7-sonnet",         64000),
    ("claude-3-5-haiku",           8192),
    ("claude-3-5-sonnet",          8192),
    ("claude-3-opus",              4096),
    ("claude-3-sonnet",            4096),
    ("claude-3-haiku",             4096),
)


def bedrock_max_output_tokens(model_id: str) -> int:
    """Return the effective output-token ceiling for a Bedrock model.

    Model IDs and inference-profile ARNs are both accepted; the lookup
    scans the known-model table by substring.  Unrecognized models
    fall back to 4096 — conservative, matches Bedrock's legacy default,
    and keeps a downstream adaptive batch sizer honest.
    """
    mid = (model_id or "").lower()
    for token, ceiling in _BEDROCK_MODEL_OUTPUT_CEILING:
        if token in mid:
            return ceiling
    return 4096


def config_from_atelier(cfg) -> LLMBackendConfig:
    """Build LLMBackendConfig from an AtelierConfig."""
    return LLMBackendConfig(
        backend=cfg.classify_llm_backend,
        api_key=cfg.classify_llm_api_key,
        model=cfg.classify_llm_model,
        base_url=cfg.classify_llm_base_url,
        max_tokens=cfg.classify_llm_max_tokens,
        temperature=cfg.classify_llm_temperature,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
        disable_reasoning=cfg.classify_llm_disable_reasoning,
        reasoning_budget=cfg.classify_llm_reasoning_budget,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        aws_region=cfg.aws_region,
        aws_session_token=cfg.aws_session_token,
    )


# ── Prompt building ──────────────────────────────────────────────


def build_category_table(category_set) -> str:
    """Build a flat markdown table of categories for the system prompt.

    Iterates ``category_set.categories``, which under the unified
    taxonomy semantics is the full tagging vocabulary (terminal and
    parent nodes alike).  Retained for callers that want a flat
    rendering; the pipeline default is ``build_category_tree`` which
    additionally encodes the parent/child structure so the LLM can
    reason about hierarchy depth when voting.
    """
    lines = [
        "| Code | Label | Aliases | Description |",
        "|------|-------|---------|-------------|",
    ]
    for cat in category_set.categories:
        aliases = (getattr(cat, "common_names", "") or "")[:40]
        desc = (cat.description or "")[:60]
        lines.append(f"| {cat.code} | {cat.label} | {aliases} | {desc} |")
    return "\n".join(lines)


def _format_sensitivity_dict(sens) -> str:
    """Render a sensitivity dict as ``role=value`` pairs, in a stable order.

    The shape is whatever the customer's annotations table carries —
    typically per-data-subject-role ratings — so we render verbatim
    rather than interpreting numeric scales.  The LLM uses its own
    knowledge of governance rating conventions to read it.
    """
    if not isinstance(sens, dict) or not sens:
        return ""
    preferred = ("non_corp", "emp_contractor", "individual", "corp")
    items: list[tuple[str, str]] = []
    for key in preferred:
        if key in sens and str(sens[key]).strip():
            items.append((key, str(sens[key]).strip()))
    for key, value in sens.items():
        if key in preferred:
            continue
        sval = str(value).strip()
        if sval:
            items.append((key, sval))
    return ", ".join(f"{k}={v}" for k, v in items)


def _depth_for(cat) -> int:
    """Return the indent depth for a category — root nodes at 0."""
    code = cat.code or ""
    if not code:
        return 0
    return code.count(".")


def build_category_tree(category_set) -> str:
    """Render the full taxonomy tree (leaves + parents) for the system prompt.

    Every node — leaf or internal — is a first-class tagging target
    in Atlas-style governance, and the customer's curators tag every
    level (``Financial Data`` parent, ``Salary`` leaf) with their own
    short codes, sensitivity ratings, and aliases.  Rendering only
    leaves discards the parent-level information *and* implicitly
    forbids the LLM from voting at the level its evidence actually
    supports.  This tree-form rendering shows parents alongside
    leaves so the LLM can pick the most-specific defensible level.

    Falls back to ``build_category_table`` when the category_set is
    not hierarchical.
    """
    cats = getattr(category_set, "all_categories", None)
    if not cats:
        return build_category_table(category_set)

    leaf_codes = getattr(category_set, "leaf_codes", None)
    if leaf_codes is None:
        leaf_codes = frozenset(c.code for c in category_set.categories)

    sorted_cats = sorted(cats, key=lambda c: c.code or "")

    lines: list[str] = []
    for cat in sorted_cats:
        depth = _depth_for(cat)
        indent = "  " * depth
        is_leaf = cat.code in leaf_codes

        parts: list[str] = [f"`{cat.code}` **{cat.label}**"]

        abbrev = (getattr(cat, "abbrev", "") or "").strip()
        if abbrev:
            parts.append(f"[{abbrev}]")

        sens_str = _format_sensitivity_dict(getattr(cat, "sensitivity", None))
        if sens_str:
            parts.append(f"sens: {sens_str}")

        aliases = (getattr(cat, "common_names", "") or "").strip()
        if aliases:
            parts.append(f"aliases: {aliases[:60]}")

        desc = (cat.description or "").strip()
        if desc:
            parts.append(desc[:80])

        marker = "-" if is_leaf else "▸"
        lines.append(f"{indent}{marker} " + " · ".join(parts))

        # Specifics live in embedding_text for cosine; surface a short
        # tail to the LLM only when it's clearly an example payload
        # rather than a verbatim duplicate of label/description.
        embedding_text = (getattr(cat, "embedding_text", "") or "").strip()
        if is_leaf and embedding_text:
            tail = embedding_text.split(" | ")[-1].strip()
            if (
                tail
                and tail.lower() != cat.label.lower()
                and tail.lower() != desc.lower()
                and tail.lower() != aliases.lower()
                and len(tail) >= 16
            ):
                lines.append(f"{indent}    e.g., {tail[:120]}")

    return "\n".join(lines)


def _sensitive_subtree_summary(category_set) -> str:
    """Markdown summary of the sensitivity structure, ICE conventions only.

    When the loaded vocabulary uses Atelier's publicly-grounded
    ``ICE.SENSITIVE.*`` / ``ICE.NONSENSITIVE.*`` paths, return a
    compact Markdown block naming the high-sensitivity subtree
    root, the catch-all, and a few publicly-grounded leaf
    abbreviations (per ``fixtures/PROVENANCE.md``).  Return ``""``
    for every other vocabulary shape so the prompt stays silent
    where we can't verify the sensitivity encoding is publicly
    grounded.

    See ``docs/src/architecture/dst-evidence-independence.md``.
    """
    cats = getattr(category_set, "all_categories", None)
    if not cats:
        cats = getattr(category_set, "categories", None)
    if not cats:
        return ""

    has_ice_paths = any(
        c.code.startswith("ICE.SENSITIVE") or c.code.startswith("ICE.NONSENSITIVE")
        for c in cats
    )
    if not has_ice_paths:
        return ""

    by_code = {c.code: c for c in cats}
    sensitive_codes = {c.code for c in cats if c.code.startswith("ICE.SENSITIVE")}
    nonsens_codes = {c.code for c in cats if c.code.startswith("ICE.NONSENSITIVE")}

    lines: list[str] = ["**Vocabulary sensitivity map (computed for this run):**"]
    if sensitive_codes:
        # Pick up to three publicly-grounded leaf exemplars carrying
        # an abbrev (e.g. SSN, PAN, EMAIL) — every retained leaf
        # abbrev in the universal vocabulary traces to a public
        # source per fixtures/PROVENANCE.md.
        members = [by_code[c] for c in sensitive_codes if c in by_code]
        members = [m for m in members if getattr(m, "abbrev", "")]
        members.sort(key=lambda c: (len(c.label or ""), c.code))
        exemplars = [m.abbrev for m in members[:3]]
        ex_str = ", ".join(exemplars) if exemplars else "no abbreviated leaves"
        lines.append(
            f"- High-sensitivity subtree: rooted at `ICE.SENSITIVE` Sensitive Data — "
            f"{len(sensitive_codes)} codes including {ex_str}."
        )
    if nonsens_codes:
        shortest = sorted(nonsens_codes, key=lambda c: (c.count("."), c))[0]
        label_cat = by_code.get(shortest)
        label = label_cat.label if label_cat else shortest
        lines.append(f"- Non-sensitive catch-all: `{shortest}` {label}.")

    if len(lines) > 1:
        return "\n".join(lines)
    return ""


_GOVERNANCE_PERSPECTIVE_TEXT = (
    "## Sensitivity classification perspective\n"
    "\n"
    "You're operating in the same problem-space as the BFO and CCO "
    "information-content-entity ontologies you know well, and the "
    "privacy regimes (GDPR, HIPAA, PCI DSS) those ontologies overlap "
    "with. Treat each of the customer's categories as their own "
    "refinement of those publicly-grounded concepts, and pick "
    "whichever of their codes best matches the canonical sensitivity "
    "concept you'd otherwise assign — Personally Identifiable "
    "Information, Financial Information, Technical Identifier, "
    "Biometric, and so on.\n"
    "\n"
    "A practical asymmetry: in governance, calling sensitive data "
    "non-sensitive is a larger error than the reverse. When concrete "
    "signals — patterns detected on the values, PII-laden sibling "
    "columns, monetary or identifier value formats, an ontology prior "
    "pointing sensitive — converge on a sensitive interpretation, "
    "lean that way even when no leaf is a clean match. Pick the most "
    "specific sensitive parent you can defend rather than dropping "
    "the column into a generic non-sensitive bucket. When signals are "
    "genuinely absent (operational metadata, surrogate keys, "
    "timestamps, status enums), non-sensitive is the correct call — "
    "don't reach for sensitive just because of the asymmetry.\n"
    "\n"
    "Calibrate confidence to what you actually saw, not to this "
    "asymmetry."
)


def _governance_cost_model_block(summary: str) -> str:
    """Compose the sensitivity-classification perspective section.

    Invokes the LLM's existing knowledge of BFO/CCO and the
    privacy-regime conventions (GDPR Art. 25, HIPAA Safe Harbor,
    PCI DSS) and frames the classification task as mapping the
    customer's taxonomy onto those publicly-grounded concepts.
    Cost-sensitive classification (Elkan 2001, *The Foundations
    of Cost-Sensitive Learning*) is invoked as a "practical
    asymmetry" rather than a hard rule — modern LLMs respond
    better to collaborative framing than to prescriptive
    checklists.

    Appends the per-run sensitivity hierarchy when
    ``_sensitive_subtree_summary`` produces one (ICE-conventions
    only); falls back silently otherwise.
    """
    if summary:
        return (
            f"{_GOVERNANCE_PERSPECTIVE_TEXT}\n"
            "\n"
            "For reference, this vocabulary's sensitivity hierarchy:\n"
            "\n"
            f"{summary}"
        )
    return _GOVERNANCE_PERSPECTIVE_TEXT


def build_system_prompt(category_table: str, category_set=None) -> str:
    """Build the bootstrap classification system prompt.

    When *category_set* is provided, the response-format example uses real
    codes from the loaded vocabulary so the LLM doesn't hallucinate codes
    from a different naming convention (e.g. ICE.* vs numeric dot-codes).
    The Governance Cost Model section is also vocabulary-aware: when the
    vocab carries sensitivity ratings or ICE.* path conventions, the
    block names the high-sensitivity subtree and catch-all so the LLM
    can locate the right sensitive parent under uncertainty.
    """
    # Pick a leaf and a parent for the response-format examples so the
    # contract demonstrates that both levels of specificity are valid
    # answers — Atlas-style governance treats every node as a
    # first-class tagging target.
    example_leaf = "ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN"
    example_parent = "ICE.SENSITIVE.PID.IDENTITY.GOVID"
    if category_set is not None:
        leaves = list(getattr(category_set, "categories", []) or [])
        if leaves:
            example_leaf = leaves[0].code
        parents: list = []
        leaf_codes = getattr(category_set, "leaf_codes", None)
        all_cats = getattr(category_set, "all_categories", None)
        if all_cats and leaf_codes is not None:
            parents = [c for c in all_cats if c.code not in leaf_codes]
        if parents:
            example_parent = parents[0].code
        elif len(leaves) >= 2:
            example_parent = leaves[1].code

    sensitivity_summary = _sensitive_subtree_summary(category_set)
    governance_block = _governance_cost_model_block(sensitivity_summary)

    return (
        "You are a data governance classification engine. Your task is to "
        "classify database columns into taxonomy categories based on column "
        "name, data type, sample values, and sibling context.\n"
        "\n"
        "## Taxonomy\n"
        "\n"
        "Every node — parent or leaf — is a valid classification target. "
        "Indentation shows hierarchy; ``▸`` marks a parent (internal node), "
        "``-`` marks a leaf. Each row may carry the customer's own short "
        "code in brackets, sensitivity-by-role ratings (``sens:``), aliases, "
        "and a definition. Treat the metadata as the customer's stated "
        "intent — usually reliable, but cross-check against the column "
        "you're classifying.\n"
        "\n"
        f"{category_table}\n"
        "\n"
        "## Instructions\n"
        "\n"
        "- Classify each column into exactly ONE category from the taxonomy "
        "above — pick the most specific level you can defend. If the "
        "evidence supports a leaf, name the leaf. If the evidence only "
        "supports a parent (e.g. 'this is financial something' but you "
        "can't tell which financial leaf), name the parent. Lower "
        "confidence should track decreasing specificity, not climb to "
        "compensate for it.\n"
        "- Use the exact Code value as it appears in the taxonomy.\n"
        "- Consider column name, data type, sample values, and sibling columns.\n"
        "- If no category fits, set category_code to null.\n"
        "- Provide confidence 0.0–1.0 and brief evidence.\n"
        "- For each column, list up to 3 alternative categories with confidence. "
        "Alternatives may be at any level — leaves, parents, or a mix.\n"
        "- Respond with ONLY a JSON array, no markdown fencing.\n"
        "\n"
        f"{governance_block}\n"
        "\n"
        "## Response Format\n"
        "\n"
        f'[{{"column_name": "ssn", "category_code": "{example_leaf}", "confidence": 0.95, '
        f'"evidence": "SSN pattern", "alternatives": [{{"code": "{example_parent}", "confidence": 0.03}}]}},\n'
        f' {{"column_name": "amount_field", "category_code": "{example_parent}", "confidence": 0.65, '
        f'"evidence": "monetary values, parent-level — no specific financial leaf is a clean match", "alternatives": []}}]'
    )


def _ontology_priors_for_sample(sample) -> list[dict]:
    """Detect patterns + look up canonical ICE.* metadata for one sample.

    Surfaced to the LLM as a publicly-grounded semantic anchor on
    every batch, not just on revisit — when the user vocabulary
    doesn't carry an exact match for a detected pattern, the LLM can
    still translate from the canonical ontology label/description to
    the closest fit in the user's frame (He et al. 2023, *Exploring
    Large Language Models for Ontology Alignment*).  See
    ``mass_functions.lookup_pattern_ontology``.
    """
    values = getattr(sample, "values", None) or []
    if not values:
        return []
    try:
        from atelier.classify.features import detect_patterns
        from atelier.classify.mass_functions import lookup_pattern_ontology
    except Exception:
        return []
    patterns = detect_patterns(values)
    priors: list[dict] = []
    for pattern_name, fraction in (patterns or {}).items():
        prior = lookup_pattern_ontology(pattern_name)
        if prior is None:
            continue
        prior = dict(prior)
        prior["match_fraction"] = float(fraction)
        priors.append(prior)
    return priors


def _render_ontology_priors(priors: list[dict]) -> list[str]:
    """Render ontology priors as compact prompt lines."""
    if not priors:
        return []
    rendered: list[str] = []
    for prior in priors:
        label = prior.get("label", "")
        desc = prior.get("description", "")
        aliases = prior.get("common_names", "")
        path = prior.get("path", []) or []
        path_render = " → ".join(p for p in path if p != "Information Content Entity")
        bits: list[str] = []
        if label:
            bits.append(label)
        if desc:
            bits.append(desc)
        if aliases:
            bits.append(f"aliases: {aliases}")
        if path_render:
            bits.append(f"path: {path_render}")
        if bits:
            rendered.append("- " + "; ".join(bits))
    return rendered


def build_batch_user_prompt(
    samples: list,
    revisit_context: dict[str, dict] | None = None,
    table_name: str | None = None,
) -> str:
    """Build a user prompt for a batch of columns.

    Args:
        samples: ColumnSample objects to classify.
        revisit_context: Optional per-column enrichment for revisit passes.
        table_name: Optional table name for context header.
    """
    parts: list[str] = []

    if table_name:
        parts.append(f"## Table: {table_name}\n")

    for i, sample in enumerate(samples, 1):
        revisit = revisit_context.get(sample.name) if revisit_context else None
        tag = " (REVISIT)" if revisit else ""
        lines = [f"### Column {i}: {sample.name}{tag}"]
        lines.append(f"Type: {sample.column_type or 'UNKNOWN'}")

        if sample.values:
            preview = sample.values[:10]
            lines.append(f"Values: {preview}")
            if hasattr(sample, 'all_values') and sample.all_values and len(sample.all_values) > len(preview):
                lines.append(f"({len(sample.all_values)} total values sampled)")

        if sample.siblings:
            lines.append(f"Siblings: {sample.siblings}")

        # Pattern-detected ontology priors — canonical metadata from
        # Atelier's BFO/IAO-grounded universal vocabulary.  Fed to
        # the LLM on every batch (sweep + revisit) so it has a
        # publicly-grounded translation anchor when the user
        # vocabulary doesn't carry an exact equivalent of the
        # detected pattern.  Choose the closest fit from the user's
        # own taxonomy; the canonical ICE.* code itself is NEVER a
        # valid classification target.
        priors = _ontology_priors_for_sample(sample)
        rendered_priors = _render_ontology_priors(priors)
        if rendered_priors:
            lines.append("Pattern-detected ontology priors (from Atelier's universal taxonomy — translate to the closest fit in the candidate vocabulary):")
            lines.extend(rendered_priors)

        if revisit:
            ml_pred = revisit.get("ml_prediction", "")
            bel = revisit.get("belief", 0.0)
            pl = revisit.get("plausibility", 0.0)
            k = revisit.get("conflict", 0.0)
            lines.append(f"ML prediction: {ml_pred} [Bel={bel:.2f}, Pl={pl:.2f}, K={k:.2f}]")
            if revisit.get("confusable"):
                lines.append(f"Confusable: {revisit['confusable']}")
            if revisit.get("previous"):
                prev = revisit["previous"]
                lines.append(
                    f"Your previous: {prev.get('code', '?')} (conf={prev.get('confidence', 0):.2f})"
                )
            indep = revisit.get("independent_consensus") or {}
            if indep.get("code"):
                lines.append(
                    f"Independent-tier consensus (cosine + pattern + name_match, "
                    f"excluding LLM-derivative ML): {indep.get('label') or indep['code']} "
                    f"(mass={indep.get('mass', 0):.2f})"
                )

        parts.append("\n".join(lines))

    return "\n\n".join(parts)


def _parse_classifications(text: str, expected_names: list[str]) -> list[ColumnClassification]:
    """Parse LLM JSON response into ColumnClassification list.

    Handles markdown fencing, partial JSON, and single-object responses.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1]) if len(lines) > 2 else cleaned

    # Try direct JSON parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = [data]
        if isinstance(data, list):
            return _dicts_to_classifications(data, expected_names)
    except (json.JSONDecodeError, ValueError):
        logger.debug("Direct JSON parse failed, trying regex array extraction")

    # Regex fallback: extract JSON array
    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        try:
            data = json.loads(array_match.group())
            if isinstance(data, list):
                return _dicts_to_classifications(data, expected_names)
        except (json.JSONDecodeError, ValueError):
            logger.debug("Regex array extraction failed, trying individual object extraction")

    # Last resort: extract individual JSON objects
    results = []
    for obj_match in re.finditer(r"\{[^{}]*\}", cleaned):
        try:
            d = json.loads(obj_match.group())
            results.append(d)
        except (json.JSONDecodeError, ValueError):
            continue

    if results:
        return _dicts_to_classifications(results, expected_names)

    logger.warning("Failed to parse LLM response: %s", text[:200])
    return []


def _dicts_to_classifications(
    data: list[dict], expected_names: list[str],
) -> list[ColumnClassification]:
    """Convert parsed dicts to ColumnClassification objects."""
    results = []
    for i, item in enumerate(data):
        name = item.get("column_name", "")
        if not name and i < len(expected_names):
            name = expected_names[i]

        alternatives = []
        for alt in item.get("alternatives", []):
            if isinstance(alt, dict) and "code" in alt:
                alternatives.append({
                    "code": str(alt["code"]),
                    "confidence": float(alt.get("confidence", 0.0)),
                })

        results.append(ColumnClassification(
            column_name=str(name),
            category_code=item.get("category_code"),
            confidence=float(item.get("confidence", 0.0)),
            evidence=str(item.get("evidence", "")),
            alternatives=alternatives,
        ))
    return results


def _parse_structured_response(text: str, expected_names: list[str]) -> list[ColumnClassification]:
    """Parse structured JSON output guaranteed valid by schema.

    Shared by AnthropicStructuredBackend and BedrockStructuredBackend.
    """
    data = json.loads(text)
    items = data.get("classifications", [])
    return _dicts_to_classifications(items, expected_names)


# ── Emission validation + retry ──────────────────────────────────


def validate_emissions(
    response: "LLMResponse",
    category_set,
) -> list[tuple[str, str]]:
    """Return (column_name, invalid_code) for out-of-vocabulary emissions.

    A ``category_code`` is "valid" if it appears in the deployed
    taxonomy — i.e. it matches a ``code`` entry (dot-notation) or an
    ``abbrev`` entry (annotation mnemonic, case-insensitive) on the
    given category set.  ``None`` / empty codes are passed through
    (the LLM saying "no category fits" is a valid response).

    Used by ``classify_batch_with_validation`` to detect hallucinated
    codes that pattern-match the taxonomy's conventions but don't
    actually exist (e.g. the LLM producing ``A_FD`` because it sees
    siblings using the ``A_*`` prefix family, even though the real
    annotation is ``C_FD``).
    """
    if category_set is None:
        return []
    by_code = getattr(category_set, "by_code", {}) or {}
    by_abbrev = getattr(category_set, "by_abbrev", {}) or {}
    # Case-insensitive abbrev index — the LLM occasionally emits
    # lowercased variants and they should still be valid.
    abbrev_upper = {k.upper() for k in by_abbrev if k}

    invalid: list[tuple[str, str]] = []
    for cls in response.classifications:
        code = cls.category_code
        if not code:
            continue
        if code in by_code:
            continue
        if code in by_abbrev:
            continue
        if code.upper() in abbrev_upper:
            continue
        invalid.append((cls.column_name, code))
    return invalid


def _build_validation_callout(invalid: list[tuple[str, str]]) -> str:
    """Construct the augmentation appended to system_prompt on retry.

    Names each invalid emission specifically.  Does NOT include "did
    you mean" lexical suggestions because the system prompt already
    carries the full taxonomy — pointing at the offender is the
    entire signal the LLM needs.
    """
    lines = [
        "",
        "## Re-classification needed",
        "",
        "Your previous response included codes that are not in the taxonomy:",
        "",
    ]
    for col, bad in invalid:
        lines.append(f"  - {col}: {bad!r} not found")
    lines.extend([
        "",
        "Re-classify these columns. Use only the dot-notation Code "
        "values or the bracketed annotation mnemonics from the "
        "taxonomy above. Do not invent codes that look like the "
        "conventions.",
    ])
    return "\n".join(lines)


def classify_batch_with_validation(
    backend: "LLMBackend",
    samples: list,
    system_prompt: str,
    *,
    category_set,
    revisit_context: dict[str, dict] | None = None,
    table_name: str | None = None,
    max_validation_retries: int = 2,
    on_retry: "Callable[[dict], None] | None" = None,
) -> "LLMResponse":
    """``backend.classify_batch`` with out-of-vocabulary retry.

    When ``category_set`` is ``None`` this is a passthrough to
    ``backend.classify_batch`` (no validation, no retry).  When
    provided, every emitted ``category_code`` is checked against the
    taxonomy after the call returns; any code that isn't a known dot-
    code or annotation triggers a targeted retry of just the
    offending columns, with the system prompt augmented to name the
    invalid emissions specifically.

    The retry uses the same backend and same temperature — no model
    change.  The augmentation is appended to ``system_prompt`` so the
    full taxonomy stays cached (relevant for Bedrock prompt-cache
    behavior and Anthropic ephemeral-cache hits).

    After ``max_validation_retries`` exhausted, any residual invalid
    emissions get their ``category_code`` set to ``None`` and
    ``confidence`` to ``0.0`` — this guarantees CatBoost training
    data and bootstrap ``state.labels`` never carry the
    hallucination, which is the core contamination concern.  A 1–2%
    column-drop rate is preferable to learning a non-existent
    category.

    Token counts (input, output, reasoning) are aggregated across
    the initial call and all retries so cost accounting is honest.

    ``on_retry`` callback is invoked once per retry with a dict
    ``{retry_idx, invalid_count, column_names, invalid_codes}`` so
    callers can record trajectory data without coupling this layer
    to bootstrap state.
    """
    import dataclasses

    response = backend.classify_batch(
        samples=samples, system_prompt=system_prompt,
        revisit_context=revisit_context, table_name=table_name,
    )
    if category_set is None:
        return response

    invalid = validate_emissions(response, category_set)
    if not invalid:
        return response

    total_in = response.input_tokens
    total_out = response.output_tokens
    total_reasoning = response.reasoning_tokens
    classifications = list(response.classifications)

    for retry_idx in range(max_validation_retries):
        invalid_names = {n for n, _ in invalid}
        retry_samples = [s for s in samples if s.name in invalid_names]
        if not retry_samples:
            break

        if on_retry is not None:
            try:
                on_retry({
                    "retry_idx": retry_idx,
                    "invalid_count": len(invalid),
                    "column_names": [n for n, _ in invalid],
                    "invalid_codes": [c for _, c in invalid],
                })
            except Exception:
                pass  # observability never breaks the sweep

        bad_preview = ", ".join(f"{n}={c!r}" for n, c in invalid[:5])
        if len(invalid) > 5:
            bad_preview += f" (+{len(invalid)-5} more)"
        logger.info(
            "llm validation: retry %d/%d on %d invalid emission(s): %s",
            retry_idx + 1, max_validation_retries, len(invalid), bad_preview,
        )

        augmented_system = system_prompt + _build_validation_callout(invalid)
        retry_response = backend.classify_batch(
            samples=retry_samples, system_prompt=augmented_system,
            revisit_context=revisit_context, table_name=table_name,
        )
        total_in += retry_response.input_tokens
        total_out += retry_response.output_tokens
        total_reasoning += retry_response.reasoning_tokens

        # Merge retry results back into the response classifications.
        # Columns the retry didn't return for stay with their original
        # (invalid) emission and become eligible for the next retry
        # iteration or, after exhaustion, the blank-out step below.
        retry_by_name = {cls.column_name: cls for cls in retry_response.classifications}
        classifications = [
            retry_by_name.get(cls.column_name, cls)
            for cls in classifications
        ]

        merged = dataclasses.replace(
            response, classifications=classifications,
            input_tokens=total_in, output_tokens=total_out,
            reasoning_tokens=total_reasoning,
        )
        invalid = validate_emissions(merged, category_set)
        if not invalid:
            return merged

    # Exhaustion — clear any residual invalid emissions so downstream
    # consumers (state.labels, CatBoost fit-to-LLM, evaluation) see
    # ``None`` rather than the hallucinated code.  Bootstrap's existing
    # ``if not llm_code: continue`` guard at the training-pair filter
    # naturally drops these columns from CatBoost training, which is
    # the desired terminal behavior.
    if invalid:
        invalid_names = {n for n, _ in invalid}
        bad_preview = ", ".join(f"{n}={c!r}" for n, c in invalid[:5])
        if len(invalid) > 5:
            bad_preview += f" (+{len(invalid)-5} more)"
        logger.warning(
            "llm validation: exhausted %d retries on %d column(s); "
            "blanking their category_code to prevent contamination: %s",
            max_validation_retries, len(invalid), bad_preview,
        )
        classifications = [
            dataclasses.replace(cls, category_code=None, confidence=0.0)
            if cls.column_name in invalid_names else cls
            for cls in classifications
        ]

    return dataclasses.replace(
        response, classifications=classifications,
        input_tokens=total_in, output_tokens=total_out,
        reasoning_tokens=total_reasoning,
    )


# ── Abstract backend ─────────────────────────────────────────────


class LLMBackend(ABC):
    """Abstract LLM backend for batch column classification."""

    def __init__(self, config: LLMBackendConfig) -> None:
        self._config = config

    @abstractmethod
    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        """Classify a batch of columns.

        Args:
            samples: ColumnSample objects (up to batch_size).
            system_prompt: Pre-built system prompt with category table.
            revisit_context: Optional enrichment for revisit passes.
            table_name: Optional table name for prompt context.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Verify the backend is reachable and functional."""


# ── Anthropic backend ────────────────────────────────────────────


class AnthropicBackend(LLMBackend):
    """Backend using the Anthropic Messages API (Claude)."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic
            import httpx
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )

        if not self._config.api_key:
            raise ValueError("Anthropic API key required. Set ATELIER_LLM_API_KEY.")

        self._client = anthropic.Anthropic(
            api_key=self._config.api_key,
            timeout=httpx.Timeout(connect=15.0, read=180.0, write=10.0, pool=5.0),
        )
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self._config.reasoning_budget and not self._config.disable_reasoning:
            _apply_thinking(kwargs, self._config.model, self._config.reasoning_budget)

        response = client.messages.create(**kwargs)

        # With thinking enabled, response may have thinking blocks before text
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            response.content[-1] if response.content else None,
        )
        text = (text_block.text if text_block else "").strip()
        classifications = _parse_classifications(text, expected_names)

        finish_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        returned_names = {c.column_name for c in classifications if c.column_name}
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = bool(missing) and finish_reason not in ("length", "max_tokens")
        if is_partial:
            logger.warning(
                "Anthropic response PARTIAL: got %d/%d items, stop=%s; missing=%s",
                len(classifications), len(expected_names), finish_reason,
                ", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
            partial=is_partial,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(resp.content) > 0
        except Exception:
            return False


# ── Anthropic structured output backend ──────────────────────────


class AnthropicStructuredBackend(LLMBackend):
    """Backend using the Anthropic Messages API with structured output.

    Uses ``output_config`` with JSON Schema for guaranteed valid responses
    (constrained decoding — no regex/JSON parsing fallbacks).  The system
    prompt is sent with ``cache_control`` for prompt caching (90% input
    token discount on subsequent calls in the same session).

    This is the default backend when ``ANTHROPIC_API_KEY`` is present
    and no explicit classify LLM backend is configured.
    """

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import anthropic
            import httpx
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: uv add anthropic"
            )

        if not self._config.api_key:
            raise ValueError("Anthropic API key required.")

        self._client = anthropic.Anthropic(
            api_key=self._config.api_key,
            timeout=httpx.Timeout(connect=15.0, read=180.0, write=10.0, pool=5.0),
        )
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": user_prompt}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": CLASSIFICATION_OUTPUT_SCHEMA,
                },
            },
        }
        if self._config.reasoning_budget and not self._config.disable_reasoning:
            _apply_thinking(kwargs, self._config.model, self._config.reasoning_budget)

        response = client.messages.create(**kwargs)

        # With thinking enabled, text block may not be first
        text_block = next(
            (b for b in response.content if getattr(b, "type", None) == "text"),
            response.content[-1] if response.content else None,
        )
        text = text_block.text if text_block else ""
        classifications = _parse_structured_response(text, expected_names)

        cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        if cache_read:
            logger.debug("Prompt cache hit: %d tokens read from cache", cache_read)
        elif cache_write:
            logger.debug("Prompt cache miss: %d tokens written to cache", cache_write)

        finish_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
        returned_names = {c.column_name for c in classifications if c.column_name}
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = bool(missing) and finish_reason not in ("length", "max_tokens")
        if is_partial:
            logger.warning(
                "Anthropic structured response PARTIAL: got %d/%d items, "
                "stop=%s; missing=%s",
                len(classifications), len(expected_names), finish_reason,
                ", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
            partial=is_partial,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.messages.create(
                model=self._config.model,
                max_tokens=32,
                messages=[{"role": "user", "content": "Respond with {\"ok\":true}"}],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": {
                            "type": "object",
                            "properties": {"ok": {"type": "boolean"}},
                            "required": ["ok"],
                            "additionalProperties": False,
                        },
                    },
                },
            )
            return len(resp.content) > 0
        except Exception:
            return False


# ── OpenAI-compatible backend ────────────────────────────────────


class OpenAICompatibleBackend(LLMBackend):
    """Backend using OpenAI-compatible API (vLLM, GLM-4.7, etc.)."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None
        # Set to True once the endpoint 422s on an extra_body reasoning
        # key; future requests in this session drop the extra_body.
        self._reasoning_unsupported = False

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import openai
        except ImportError:
            raise ImportError(
                "openai package required. Install with: uv add openai"
            )

        kwargs: dict[str, Any] = {}
        if self._config.api_key:
            kwargs["api_key"] = self._config.api_key
        else:
            kwargs["api_key"] = "EMPTY"

        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url

        self._client = openai.OpenAI(**kwargs)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        api_params: dict = {
            "model": self._config.model,
            "max_tokens": self._config.max_tokens,
            "temperature": self._config.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        # Reasoning fields are opt-in via config.  Some OpenAI-compatible
        # endpoints (e.g. stock vLLM) 422 on unknown extra_body keys; we
        # strip the offending field once and remember for the session.
        extra_body: dict[str, Any] = {}
        if self._config.disable_reasoning and not self._reasoning_unsupported:
            extra_body["disable_reasoning"] = True
        if (
            self._config.reasoning_budget
            and not self._config.disable_reasoning
            and not self._reasoning_unsupported
        ):
            extra_body["reasoning_budget"] = self._config.reasoning_budget
        if extra_body:
            api_params["extra_body"] = extra_body

        # Retry with exponential backoff for transient errors
        last_error: Exception | None = None
        last_was_throttle = False
        for attempt in range(self._config.max_retries):
            try:
                response = client.chat.completions.create(**api_params)
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                # 429 = rate limit (throttle); 503 = service unavailable
                # (often capacity-driven, semantically a throttle).
                # 502 / 504 = gateway timeouts (network-side, not
                # throttling — halve-on-retry as today).
                throttle = any(code in err_str for code in ("429", "503"))
                retryable = throttle or any(
                    code in err_str for code in ("502", "504")
                )
                last_was_throttle = throttle
                # Backend doesn't understand extra_body reasoning keys —
                # drop them, mark the session, and retry once without delay.
                unsupported = (
                    "reasoning_budget" in err_str
                    or "disable_reasoning" in err_str
                ) and ("422" in err_str or "unsupported" in err_str.lower())
                if unsupported and not self._reasoning_unsupported:
                    logger.warning(
                        "Backend rejected reasoning extra_body (%s); disabling "
                        "reasoning for the rest of this session.", e,
                    )
                    self._reasoning_unsupported = True
                    api_params.pop("extra_body", None)
                    continue
                if retryable and attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._config.max_retries, delay, e,
                    )
                    time.sleep(delay)
                    continue
                raise
        else:
            if last_was_throttle:
                raise ThrottledError(
                    f"OpenAI-compatible throttling — {self._config.max_retries} "
                    f"backoff attempts exhausted: {last_error}",
                    cause=last_error,
                )
            raise last_error  # type: ignore[misc]

        msg = response.choices[0].message
        text = (msg.content or "").strip()
        # GLM-4.7 on Cerebras with ``reasoning_format="parsed"`` (default)
        # returns the thinking trace in a dedicated ``reasoning`` field
        # separate from the final answer in ``content``.  Capture it when
        # present so downstream pipelines can persist the reasoning as a
        # research artifact.  Other OpenAI-compatible backends simply
        # don't emit this field — getattr keeps us compatible.
        reasoning_text = (getattr(msg, "reasoning", None) or "").strip()

        finish_reason = response.choices[0].finish_reason or "stop"
        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        reasoning_tokens = 0
        if response.usage and response.usage.completion_tokens_details:
            reasoning_tokens = getattr(
                response.usage.completion_tokens_details, "reasoning_tokens", 0
            ) or 0

        if finish_reason == "length":
            logger.warning(
                "LLM response TRUNCATED: %d chars, in=%d, out=%d "
                "(reasoning=%d, max_tokens=%d) — increase max_tokens or reduce batch size",
                len(text), input_tokens, output_tokens,
                reasoning_tokens, self._config.max_tokens,
            )
        else:
            logger.info(
                "LLM response: %d chars, finish=%s, in=%d, out=%d (reasoning=%d, reasoning_text_chars=%d)",
                len(text), finish_reason, input_tokens, output_tokens,
                reasoning_tokens, len(reasoning_text),
            )

        classifications = _parse_classifications(text, expected_names)

        returned_names = {c.column_name for c in classifications if c.column_name}
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = bool(missing) and finish_reason not in ("length", "max_tokens")
        if is_partial:
            logger.warning(
                "LLM response PARTIAL: got %d/%d items, finish=%s; missing=%s",
                len(classifications), len(expected_names), finish_reason,
                ", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
            partial=is_partial,
            reasoning_text=reasoning_text,
            reasoning_tokens=reasoning_tokens,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self._config.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            return len(resp.choices) > 0
        except Exception:
            return False


# ── Bedrock backend ─────────────────────────────────────────────


class _BedrockMixin:
    """Shared max-token awareness for Bedrock backends.

    The configured ``max_tokens`` is what the pipeline *asks* for; the
    actual ceiling Bedrock enforces is model-dependent.  Exposing the
    effective ceiling lets the bootstrap batch sizer scale down front-
    of-pipeline instead of eating a halving round-trip per batch.
    """

    _config: "LLMBackendConfig"

    def effective_max_tokens(self) -> int:
        return min(
            self._config.max_tokens,
            bedrock_max_output_tokens(self._config.model),
        )


class BedrockBackend(_BedrockMixin, LLMBackend):
    """Backend using AWS Bedrock Converse API."""

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package required for Bedrock. Install with: uv add boto3"
            )

        if not self._config.aws_access_key_id or not self._config.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required for Bedrock backend. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        from atelier.config import region_from_arn
        from botocore.config import Config

        arn_region = region_from_arn(self._config.model)
        effective_region = arn_region or self._config.aws_region or "us-east-1"

        # Explicit timeouts because the default boto3 connect-timeout
        # is generous and a cold-boot CAI pod waiting for egress policy
        # propagation can blackhole SYN packets for minutes.  With these
        # values, a hung connect fails fast (15s) and a hung response
        # fails reasonably (180s), both surfacing as exceptions the
        # halving/retry loop can act on instead of an unbounded stall.
        # max_attempts=0 disables boto3's implicit retry so our own
        # halving retry stays in control of the retry budget.
        bcfg = Config(
            connect_timeout=15,
            read_timeout=180,
            retries={"max_attempts": 0},
        )
        session = boto3.Session(
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            aws_session_token=self._config.aws_session_token,
            region_name=effective_region,
        )
        self._client = session.client("bedrock-runtime", config=bcfg)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        # Clamp to the model's actual output ceiling.  Bedrock enforces
        # the model's native limit silently, so asking for 65536 against
        # claude-3-5-sonnet (cap 8192) just truncates with no warning.
        effective_max = min(
            self._config.max_tokens,
            bedrock_max_output_tokens(self._config.model),
        )

        response = client.converse(
            modelId=self._config.model,
            system=[{"text": system_prompt}],
            messages=[{
                "role": "user",
                "content": [{"text": user_prompt}],
            }],
            inferenceConfig={
                "maxTokens": effective_max,
                "temperature": self._config.temperature,
            },
        )

        text = response["output"]["message"]["content"][0]["text"].strip()
        classifications = _parse_classifications(text, expected_names)
        usage = response.get("usage", {})

        finish_reason = response.get("stopReason", "end_turn")
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        # Partial-response detection: clean stop but missing columns.
        returned_names = {c.column_name for c in classifications if c.column_name}
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = bool(missing) and finish_reason not in ("length", "max_tokens")

        if finish_reason == "max_tokens":
            logger.warning(
                "Bedrock response TRUNCATED: %d chars, in=%d, out=%d, max_tokens=%d",
                len(text), input_tokens, output_tokens, effective_max,
            )
        elif is_partial:
            logger.warning(
                "Bedrock response PARTIAL: got %d/%d classifications, "
                "finish=%s, max_tokens=%d; missing=%s",
                len(classifications), len(expected_names),
                finish_reason, effective_max,
                ", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
            )
        else:
            logger.info(
                "Bedrock response: %d chars, finish=%s, in=%d, out=%d",
                len(text), finish_reason, input_tokens, output_tokens,
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=finish_reason,
            partial=is_partial,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            response = client.converse(
                modelId=self._config.model,
                messages=[{
                    "role": "user",
                    "content": [{"text": "ping"}],
                }],
                inferenceConfig={"maxTokens": 10},
            )
            return "output" in response
        except Exception:
            return False


# ── Bedrock structured output backend ────────────────────────────


class BedrockStructuredBackend(_BedrockMixin, LLMBackend):
    """Backend using Bedrock invoke_model with tool-use for structured output.

    Uses ``invoke_model`` with the raw Anthropic Messages format.  Structured
    output is obtained via **forced tool-use** (``tools`` + ``tool_choice``)
    rather than ``output_config`` which is **NOT supported** by Bedrock's
    ``invoke_model`` endpoint.

    The system prompt is sent with ``cache_control`` for prompt caching
    (90% input token discount on cache hits, 5-min default TTL on Bedrock).

    When extended thinking is enabled, ``tool_choice`` must be ``"auto"``
    (Anthropic does not allow forced tool selection with thinking).  A
    text-block fallback parser handles this case.

    This is the auto-default backend when AWS Bedrock credentials are
    present and no explicit classify LLM or ANTHROPIC_API_KEY is configured.
    """

    def __init__(self, config: LLMBackendConfig) -> None:
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 package required for Bedrock. Install with: uv add boto3"
            )

        if not self._config.aws_access_key_id or not self._config.aws_secret_access_key:
            raise ValueError(
                "AWS credentials required for Bedrock backend. "
                "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
            )

        from atelier.config import region_from_arn
        from botocore.config import Config

        arn_region = region_from_arn(self._config.model)
        effective_region = arn_region or self._config.aws_region or "us-east-1"

        # Explicit timeouts because the default boto3 connect-timeout
        # is generous and a cold-boot CAI pod waiting for egress policy
        # propagation can blackhole SYN packets for minutes.  With these
        # values, a hung connect fails fast (15s) and a hung response
        # fails reasonably (180s), both surfacing as exceptions the
        # halving/retry loop can act on instead of an unbounded stall.
        # max_attempts=0 disables boto3's implicit retry so our own
        # halving retry stays in control of the retry budget.
        bcfg = Config(
            connect_timeout=15,
            read_timeout=180,
            retries={"max_attempts": 0},
        )
        session = boto3.Session(
            aws_access_key_id=self._config.aws_access_key_id,
            aws_secret_access_key=self._config.aws_secret_access_key,
            aws_session_token=self._config.aws_session_token,
            region_name=effective_region,
        )
        self._client = session.client("bedrock-runtime", config=bcfg)
        return self._client

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        client = self._get_client()
        user_prompt = build_batch_user_prompt(samples, revisit_context, table_name)
        expected_names = [s.name for s in samples]

        use_thinking = bool(
            self._config.reasoning_budget and not self._config.disable_reasoning
        )

        # Clamp to the model's actual output ceiling — see note on
        # BedrockBackend.classify_batch for why this matters.
        effective_max = min(
            self._config.max_tokens,
            bedrock_max_output_tokens(self._config.model),
        )

        request_body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": effective_max,
            "temperature": self._config.temperature,
            "system": [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": user_prompt}],
            # Tool-use for structured output (output_config is NOT
            # supported on Bedrock invoke_model).
            "tools": [{
                "name": "classify_columns",
                "description": "Submit structured classification results.",
                "input_schema": CLASSIFICATION_OUTPUT_SCHEMA,
            }],
        }

        if use_thinking:
            request_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": self._config.reasoning_budget,
            }
            # Extended thinking requires temperature=1 and does NOT
            # allow forced tool_choice — must use "auto".
            request_body["temperature"] = 1
            request_body["tool_choice"] = {"type": "auto"}
        else:
            request_body["tool_choice"] = {
                "type": "tool",
                "name": "classify_columns",
            }

        last_error: Exception | None = None
        last_was_throttle = False
        for attempt in range(self._config.max_retries):
            try:
                response = client.invoke_model(
                    modelId=self._config.model,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body),
                )
                break
            except Exception as e:
                last_error = e
                err_str = str(e)
                throttle = any(code in err_str for code in (
                    "ThrottlingException", "TooManyRequestsException",
                    "ServiceUnavailableException",
                    "429", "503", "529",
                ))
                # ``ModelTimeoutException`` is a recoverable timeout but
                # not a throttle — keep it in the retryable set without
                # tagging it as throttle (so the outer halving loop can
                # halve as today rather than sleep-at-same-size).
                retryable = throttle or "ModelTimeoutException" in err_str
                last_was_throttle = throttle
                if retryable and attempt < self._config.max_retries - 1:
                    delay = self._config.retry_delay * (2 ** attempt)
                    logger.warning(
                        "Bedrock transient error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self._config.max_retries, delay, e,
                    )
                    time.sleep(delay)
                    continue
                raise
        else:
            # Inner-retry exhaustion.  When the last failure was a
            # throttle, raise ``ThrottledError`` so the bootstrap halving
            # loop sleeps-at-same-size instead of halving and amplifying
            # concurrent pressure on a rate-limited endpoint.
            if last_was_throttle:
                raise ThrottledError(
                    f"Bedrock throttling — {self._config.max_retries} "
                    f"backoff attempts exhausted: {last_error}",
                    cause=last_error,
                )
            raise last_error  # type: ignore[misc]

        body = json.loads(response["body"].read())

        # Extract structured output from the tool_use content block.
        # With thinking enabled (tool_choice=auto), the model may return
        # a text block instead of a tool_use block — fall back to parsing.
        tool_block = next(
            (b for b in body["content"] if b.get("type") == "tool_use"),
            None,
        )
        if tool_block:
            structured = tool_block["input"]
            items = structured.get("classifications", [])
            classifications = _dicts_to_classifications(items, expected_names)
        else:
            # Fallback: parse text block (thinking mode, auto tool_choice)
            text_block = next(
                (b for b in body["content"] if b.get("type") == "text"),
                body["content"][-1] if body["content"] else {"text": ""},
            )
            text = text_block.get("text", "")
            classifications = _parse_classifications(text, expected_names)

        usage = body.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        stop_reason = body.get("stop_reason", "end_turn")

        if cache_read:
            logger.debug("Bedrock prompt cache hit: %d tokens read", cache_read)
        elif cache_write:
            logger.debug("Bedrock prompt cache miss: %d tokens written", cache_write)

        returned_names = {c.column_name for c in classifications if c.column_name}
        missing = [n for n in expected_names if n not in returned_names]
        is_partial = bool(missing) and stop_reason not in ("length", "max_tokens")

        if stop_reason == "max_tokens":
            logger.warning(
                "Bedrock structured response TRUNCATED: %d/%d items, "
                "in=%d, out=%d, max_tokens=%d",
                len(classifications), len(expected_names),
                input_tokens, output_tokens, effective_max,
            )
        elif is_partial:
            logger.warning(
                "Bedrock structured response PARTIAL: got %d/%d items, "
                "stop=%s, max_tokens=%d; missing=%s",
                len(classifications), len(expected_names),
                stop_reason, effective_max,
                ", ".join(missing[:5]) + ("…" if len(missing) > 5 else ""),
            )
        else:
            logger.info(
                "Bedrock structured: %d items, stop=%s, in=%d, out=%d "
                "(cache_read=%d, cache_write=%d)",
                len(classifications), stop_reason, input_tokens, output_tokens,
                cache_read, cache_write,
            )

        return LLMResponse(
            classifications=classifications,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=self._config.model,
            finish_reason=stop_reason,
            partial=is_partial,
        )

    def health_check(self) -> bool:
        try:
            client = self._get_client()
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Respond with ok=true"}],
                "tools": [{
                    "name": "health_check",
                    "description": "Health check response.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"ok": {"type": "boolean"}},
                        "required": ["ok"],
                        "additionalProperties": False,
                    },
                }],
                "tool_choice": {"type": "tool", "name": "health_check"},
            }
            response = client.invoke_model(
                modelId=self._config.model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body),
            )
            body = json.loads(response["body"].read())
            return len(body.get("content", [])) > 0
        except Exception:
            return False


# ── Factory ──────────────────────────────────────────────────────


def create_backend(config: LLMBackendConfig) -> LLMBackend:
    """Create an LLM backend from configuration.

    Raises ValueError on unknown backend type.
    """
    if config.backend == "anthropic_structured":
        return AnthropicStructuredBackend(config)
    if config.backend == "anthropic":
        return AnthropicBackend(config)
    if config.backend == "openai_compatible":
        return OpenAICompatibleBackend(config)
    if config.backend == "cerebras":
        cerebras_config = LLMBackendConfig(
            backend="cerebras",
            api_key=config.api_key,
            model=config.model if config.model != "glm-4.7" else CEREBRAS_DEFAULT_MODEL,
            base_url=config.base_url or CEREBRAS_BASE_URL,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
            disable_reasoning=config.disable_reasoning,
            reasoning_budget=config.reasoning_budget,
        )
        return OpenAICompatibleBackend(cerebras_config)
    if config.backend == "bedrock":
        return BedrockBackend(config)
    if config.backend == "bedrock_structured":
        return BedrockStructuredBackend(config)
    raise ValueError(
        f"Unknown LLM backend: {config.backend!r}. "
        f"Use 'anthropic_structured', 'anthropic', 'openai_compatible', "
        f"'cerebras', 'bedrock', or 'bedrock_structured'."
    )


def create_backend_from_cfg(cfg) -> LLMBackend:
    """Create an LLM backend from an AtelierConfig.

    Resolution order:
    1. Explicit classify LLM config (ATELIER_LLM_API_KEY / ATELIER_LLM_BASE_URL)
    2. ANTHROPIC_SUBAGENT_MODEL — backend inferred from model format
    3. Neither → ValueError (fail fast)
    """
    from atelier.config import is_bedrock_model

    # 1. Explicit classify LLM backend configured
    if cfg.classify_llm_api_key or cfg.classify_llm_base_url:
        return create_backend(config_from_atelier(cfg))

    # 2. Subagent model — infer backend from model identifier format
    if not cfg.classify_subagent_model:
        raise ValueError(
            "No classification LLM configured. "
            "Set ANTHROPIC_SUBAGENT_MODEL or ATELIER_LLM_API_KEY."
        )

    model = cfg.classify_subagent_model
    if is_bedrock_model(model):
        return _build_bedrock_backend(cfg, model)
    return _build_anthropic_backend(cfg, model)


def _build_anthropic_backend(cfg, model: str) -> LLMBackend:
    """Build an AnthropicStructuredBackend from config + model."""
    if not cfg.anthropic_api_key:
        raise ValueError(
            f"Model {model!r} requires ANTHROPIC_API_KEY (direct API)."
        )
    logger.info("Classification LLM: AnthropicStructured %s", model)
    return create_backend(LLMBackendConfig(
        backend="anthropic_structured",
        api_key=cfg.anthropic_api_key,
        model=model,
        max_tokens=cfg.classify_llm_max_tokens,
        temperature=0.0,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
        reasoning_budget=cfg.classify_llm_reasoning_budget,
    ))


def _build_bedrock_backend(cfg, model: str) -> LLMBackend:
    """Build a BedrockStructuredBackend from config + model.

    Uses the region embedded in the model ARN when present, falling back
    to ``cfg.aws_region``.
    """
    if not cfg.has_bedrock:
        raise ValueError(
            f"Model {model!r} requires AWS Bedrock credentials."
        )
    from atelier.config import region_from_arn

    arn_region = region_from_arn(model)
    effective_region = arn_region or cfg.aws_region
    logger.info(
        "Classification LLM: BedrockStructured %s (region=%s)",
        model, effective_region,
    )
    return create_backend(LLMBackendConfig(
        backend="bedrock_structured",
        model=model,
        max_tokens=cfg.classify_llm_max_tokens,
        temperature=0.0,
        batch_size=cfg.classify_llm_columns_per_call,
        max_retries=cfg.classify_llm_max_retries,
        reasoning_budget=cfg.classify_llm_reasoning_budget,
        aws_access_key_id=cfg.aws_access_key_id,
        aws_secret_access_key=cfg.aws_secret_access_key,
        aws_region=effective_region,
        aws_session_token=cfg.aws_session_token,
    ))
