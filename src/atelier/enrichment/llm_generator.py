# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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

import logging
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


# ── Anthropic backend stub ────────────────────────────────────────


class AnthropicEnrichmentGenerator(EnrichmentGenerator):
    """Anthropic-backed enrichment generator.

    Reuses the project's existing Anthropic client patterns from
    ``atelier.classify.llm_backend`` rather than a fresh integration.
    The system prompt instructs the model to produce the structured
    enrichment payload; the verifier suite (run by the loop) decides
    whether the output is acceptable.

    .. note::
        Implementation deferred — the generator interface and the
        deterministic stub above are sufficient for the verifier suite
        and Qdrant-write smoke tests in this phase.  The Anthropic
        wiring lands when the integration step requires real LLM
        output.  The shape of the call site is preserved so the swap
        is local.
    """

    def __init__(self, *, model: str, max_attempts: int = 3) -> None:
        self._model = model
        self._max_attempts = max_attempts

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        raise NotImplementedError(
            "AnthropicEnrichmentGenerator.generate is a deferred stub. "
            "Wire to llm_backend or claude-agent-sdk when integrating "
            "real LLM-driven enrichment."
        )
