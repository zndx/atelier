"""LLM enrichment generation for annotation profiles.

Pluggable generator interface — the enrichment loop calls
``generate(source_row, ...) -> dict`` without knowing whether the
implementation is a direct Anthropic API call, a Bedrock call through
``llm_backend``, a ``claude-agent-sdk`` harness, or a deterministic
stub for tests.

The Agent-SDK-driven harness with in-situ verification is the long-term
implementation per the memory's harness-vs-model principle (the
qualifying property is non-LLM machinery in the loop, not the model
identity).  The interface here is what makes the swap zero-cost.

The deterministic stub (:class:`DeterministicStubGenerator`) is provided
for tests and for the initial verifier-suite smoke runs before Agent SDK
integration lands.
"""

from __future__ import annotations

import json
import logging
import re as _re
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Generator interface ───────────────────────────────────────────


@dataclass
class GenerationResult:
    """One enrichment generation attempt's output.

    ``enrichment`` is the payload dict — the fields documented in the
    architecture note's Qdrant payload schema, *minus* the
    provenance/verifier/audit fields (those are added by the loop).

    ``attempts`` records how many LLM calls were needed to produce this
    output, useful for cost telemetry and for surfacing flaky
    generation patterns.
    """

    enrichment: dict
    attempts: int = 1
    notes: str = ""


class EnrichmentGenerator(ABC):
    """Abstract base for enrichment generators."""

    @abstractmethod
    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        """Generate enrichment for one annotation.

        ``source_row`` carries label/mnemonic/description/parent_code.
        ``taxonomy_context`` gives global context (full tag list,
        hierarchy, current row's expected parent path, etc.).
        ``prior_attempt`` and ``verifier_feedback`` are non-None when the
        loop is asking for a refinement — implementations may use the
        feedback to focus the next generation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for the generator (recorded in payload ``generated_by``)."""
        raise NotImplementedError


# ── Deterministic stub ────────────────────────────────────────────


class DeterministicStubGenerator(EnrichmentGenerator):
    """A non-LLM stub that produces structurally-valid enrichment.

    Useful for verifier-suite tests, for offline smoke runs, and for
    tests that need a generator without paying for real LLM calls.  The
    output is intentionally minimal — enough to satisfy the verifier
    suite, not enough to be useful at classification time.
    """

    def __init__(self, *, name: str = "stub:deterministic") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        mnemonic = source_row.get("mnemonic", "")
        label = source_row.get("label", mnemonic)
        parent_path = taxonomy_context.get("expected_parent_path") or [label]

        enrichment = {
            "prototype_values": [f"stub-prototype-{i}-{mnemonic}" for i in range(3)],
            "value_patterns": [
                {"kind": "regex", "expr": ".+"},  # vacuously matches the stubs
            ],
            "name_hints": [mnemonic.lower(), label.lower().replace(" ", "_")],
            "anti_examples": [],
            "parent_path": parent_path,
        }
        return GenerationResult(enrichment=enrichment, attempts=1, notes="stub")


# ── Provider-co-located generator ─────────────────────────────────


class ClassifyBackedEnrichmentGenerator(EnrichmentGenerator):
    """Enrichment generator co-located with the classify LLM backend.

    The provider is NOT a separate knob.  Enrichment uses whatever
    backend the classify path is configured to use
    (``cfg.classify_llm_backend``) and selects the strongest reasoning
    model available from that provider via
    :func:`atelier.enrichment.model_resolver.resolve_enrichment_model`.
    This keeps provider+credential management to a single operator-
    facing knob and is load-bearing for the unified-cost-regime
    discipline.

    The generator is single-shot per taxonomy node: builds a system
    prompt (leaf-aware or parent-aware), builds a user prompt with
    source row + taxonomy context (siblings, children, valid codes
    for anti-example targeting), calls the backend's text-generation
    surface via :mod:`atelier.enrichment.backend_client`, parses the
    JSON response into the enrichment dict, and returns a
    :class:`GenerationResult` whose ``name`` records
    ``{backend}:{model}`` for downstream provenance.

    Reasoning depth is enrichment-tuned (default 16384 tokens via
    ``cfg.enrichment_reasoning_budget``) — generous because per-tag
    generation is single-shot and the model is doing structural
    taxonomy judgment work (sibling discrimination, prototype
    selection, pattern induction).
    """

    def __init__(self, cfg) -> None:
        from atelier.enrichment.model_resolver import resolve_enrichment_model
        self._cfg = cfg
        self._backend, self._model = resolve_enrichment_model(cfg)
        self._max_tokens = cfg.enrichment_max_tokens
        self._reasoning_budget = cfg.enrichment_reasoning_budget
        self._temperature = cfg.enrichment_temperature

    @property
    def name(self) -> str:
        return f"{self._backend}:{self._model}"

    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        from atelier.enrichment.backend_client import generate_text
        from atelier.enrichment.prompts import (
            build_system_prompt,
            build_user_prompt,
        )

        is_leaf = bool(taxonomy_context.get("is_leaf", True))
        system_prompt = build_system_prompt(is_leaf=is_leaf)

        prior_json: str | None = None
        if prior_attempt is not None:
            try:
                prior_json = json.dumps(prior_attempt.enrichment, indent=2)
            except (TypeError, ValueError):
                prior_json = str(prior_attempt.enrichment)
        user_prompt = build_user_prompt(
            source_row=source_row,
            taxonomy_context=taxonomy_context,
            is_leaf=is_leaf,
            prior_attempt_json=prior_json,
            verifier_feedback=verifier_feedback,
        )

        # Parse failures are sampling noise (truncated reasoning,
        # malformed JSON), not configuration errors — resample up to
        # twice before surfacing.  Backend/transport errors propagate
        # immediately; only extraction retries here.
        # Retry temperature escalation: at the default temperature of
        # 0.0 a resample is byte-identical, so a deterministic failure
        # (e.g. a local model exhausting its budget mid-reasoning)
        # repeats forever.  Escalate with both the loop-level attempt
        # count (verifier feedback rounds) and the local parse
        # attempts so every retry explores.
        prior_attempts = prior_attempt.attempts if prior_attempt else 0
        parse_error: ValueError | None = None
        enrichment: dict = {}
        parse_notes = ""
        for _parse_attempt in range(3):
            temperature = min(
                self._temperature
                + 0.25 * prior_attempts
                + 0.35 * _parse_attempt,
                0.9,
            )
            text = generate_text(
                backend=self._backend,
                model=self._model,
                cfg=self._cfg,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self._max_tokens,
                reasoning_budget=self._reasoning_budget,
                temperature=temperature,
            )
            try:
                enrichment, parse_notes = _parse_enrichment_json(text)
                parse_error = None
                break
            except ValueError as exc:
                parse_error = exc
                logger.warning(
                    "Enrichment JSON extraction failed for %s (attempt "
                    "%d/3): %s — resampling",
                    source_row.get("code", "?"), _parse_attempt + 1, exc,
                )
        if parse_error is not None:
            raise parse_error
        attempts = 1 + (prior_attempt.attempts if prior_attempt else 0)
        notes = parse_notes if parse_notes else ""
        return GenerationResult(
            enrichment=enrichment,
            attempts=attempts,
            notes=notes,
        )


# ── Backward-compatibility alias ──────────────────────────────────


class AnthropicEnrichmentGenerator(ClassifyBackedEnrichmentGenerator):
    """Deprecated alias preserved for one release cycle.

    Use :class:`ClassifyBackedEnrichmentGenerator` directly.  This
    alias exists so callers that constructed the deprecated class with
    a ``model=...`` kwarg continue to import the symbol, but the new
    class derives its backend + model from ``cfg`` rather than
    accepting them as args.  Constructing this alias with the old
    keyword args will raise :class:`TypeError` — that is intentional;
    silent fallback to the old shape would re-introduce the
    Anthropic-only assumption that the pivot deliberately removes.
    """

    def __init__(self, cfg) -> None:
        import warnings
        warnings.warn(
            "AnthropicEnrichmentGenerator is deprecated; use "
            "ClassifyBackedEnrichmentGenerator(cfg) — enrichment "
            "now co-locates with the classify backend rather than "
            "embedding Anthropic in the class name.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(cfg)


# ── JSON parsing helper ───────────────────────────────────────────


def _parse_enrichment_json(text: str) -> tuple[dict, str]:
    """Extract the enrichment JSON object from a model response.

    Models occasionally wrap JSON in code fences or include trailing
    prose despite system-prompt instructions.  We try strict parse
    first; fall back to extracting the largest balanced ``{...}``
    block.  Returns ``(enrichment_dict, notes)`` — notes is non-empty
    when a fallback path was taken, so the loop can record that fact
    in the payload audit trail.

    Raises :class:`ValueError` when no parseable JSON object can be
    located; the loop converts that into a verifier failure so
    refinement gets a chance.
    """
    import json as _json

    s = text.strip()
    # Strip reasoning blocks first — local reasoning models (Nemotron,
    # DeepSeek-style) may emit <think>…</think> ahead of the answer,
    # and think-content often contains braces that would poison the
    # balanced-brace fallback below.  An unterminated think block
    # (output truncated mid-reasoning) leaves no JSON by definition.
    s = _re.sub(r"<think>.*?</think>", "", s, flags=_re.DOTALL).strip()
    if s.startswith("<think>"):
        raise ValueError(
            "Enrichment response was an unterminated reasoning block "
            "(output truncated before the JSON answer)."
        )
    # Strip common fence patterns
    if s.startswith("```"):
        s = s.split("```", 2)[1] if "```" in s else s
        if s.startswith("json"):
            s = s[4:].lstrip()
        s = s.rsplit("```", 1)[0].strip()

    try:
        obj = _json.loads(s)
        if isinstance(obj, dict):
            return obj, ""
    except _json.JSONDecodeError:
        pass

    # Fallback: find the largest balanced { ... } substring.
    start = s.find("{")
    if start < 0:
        raise ValueError(
            "Enrichment response contained no JSON object."
        )
    depth = 0
    end = -1
    for i, ch in enumerate(s[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError(
            "Enrichment response had unbalanced braces; "
            "could not extract a JSON object."
        )
    candidate = s[start:end]
    try:
        obj = _json.loads(candidate)
        if isinstance(obj, dict):
            return obj, "parsed via balanced-brace fallback"
    except _json.JSONDecodeError as exc:
        raise ValueError(
            f"Enrichment response did not contain valid JSON: {exc}"
        )
    raise ValueError(
        "Enrichment response JSON did not parse to a dict."
    )
