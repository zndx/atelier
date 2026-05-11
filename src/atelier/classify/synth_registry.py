# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Extensible generator registry with coverage tracking.

Builds a full registry for any vocabulary by layering:
1. Hand-coded generators from synth_generators.py
2. Template generators from provided value_templates
3. Inferred generators from category metadata (description/common_names)

This enables the onboarding workflow: user provides a custom taxonomy →
synth framework automatically covers most categories → user fills gaps
with template generators from real samples.
"""

from __future__ import annotations

import re
import random
import string
from collections.abc import Callable
from dataclasses import dataclass

from atelier.classify.synth_generators import GENERATORS


@dataclass(frozen=True)
class GeneratorSpec:
    """A registered generator with provenance tracking."""
    code: str
    generator: Callable[[random.Random], str]
    source: str       # "hand-coded" | "template" | "inferred"


class GeneratorRegistry:
    """Extensible registry mapping category codes to value generators.

    Layers: hand-coded (most accurate) > template (from real samples) > inferred (from metadata).
    """

    def __init__(self) -> None:
        self._specs: dict[str, GeneratorSpec] = {}

    def register(self, code: str, gen: Callable[[random.Random], str], source: str = "hand-coded") -> None:
        """Register a generator. Later registrations with lower-priority source don't overwrite."""
        _priority = {"hand-coded": 3, "template": 2, "inferred": 1}
        existing = self._specs.get(code)
        if existing and _priority.get(existing.source, 0) >= _priority.get(source, 0):
            return  # Don't overwrite higher-priority source
        self._specs[code] = GeneratorSpec(code=code, generator=gen, source=source)

    def get(self, code: str) -> GeneratorSpec | None:
        return self._specs.get(code)

    def __contains__(self, code: str) -> bool:
        return code in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def coverage_report(self, category_set) -> dict[str, str]:
        """Report coverage: code → source or "missing" for each leaf."""
        report: dict[str, str] = {}
        for code in category_set.leaf_codes:
            spec = self._specs.get(code)
            report[code] = spec.source if spec else "missing"
        return report

    def coverage_summary(self, category_set) -> dict[str, int]:
        """Summarize coverage by source type."""
        report = self.coverage_report(category_set)
        summary: dict[str, int] = {}
        for source in report.values():
            summary[source] = summary.get(source, 0) + 1
        return summary

    @classmethod
    def from_vocabulary(
        cls,
        category_set,
        value_templates: dict[str, list[str]] | None = None,
    ) -> GeneratorRegistry:
        """Build a full registry for any vocabulary.

        1. Hand-coded generators from GENERATORS
        2. Template generators from provided value_templates
        3. Inferred generators from category metadata
        """
        registry = cls()

        # Layer 1: hand-coded generators
        for code in category_set.leaf_codes:
            if code in GENERATORS:
                registry.register(code, GENERATORS[code], "hand-coded")

        # Layer 2: template generators from real sample values
        if value_templates:
            for code, values in value_templates.items():
                if code in category_set.leaf_codes and len(values) >= 3:
                    registry.register(code, _make_template_generator(values), "template")

        # Layer 3: inferred generators from category metadata
        for cat in category_set.categories:
            if cat.code not in category_set.leaf_codes:
                continue
            if cat.code in registry:
                continue
            gen = _infer_generator(cat)
            if gen:
                registry.register(cat.code, gen, "inferred")

        return registry


# ── Template generator ─────────────────────────────────────────

def _make_template_generator(templates: list[str]) -> Callable[[random.Random], str]:
    """Create a generator that samples from real data values with mild perturbation."""
    def _gen(rng: random.Random) -> str:
        base = rng.choice(templates)
        try:
            val = float(base.replace(",", ""))
            jitter = val * rng.uniform(-0.1, 0.1)
            result = val + jitter
            if "." not in base and "e" not in base.lower():
                return str(int(result))
            return f"{result:.2f}"
        except (ValueError, OverflowError):
            pass
        if len(base) > 3 and rng.random() < 0.3:
            chars = list(base)
            idx = rng.randint(0, len(chars) - 1)
            if chars[idx].isalpha():
                chars[idx] = rng.choice(string.ascii_letters)
            elif chars[idx].isdigit():
                chars[idx] = str(rng.randint(0, 9))
            return "".join(chars)
        return base
    return _gen


# ── Inferred generators ───────────────────────────────────────

# Pattern: keyword regex → generator function
_INFERENCE_PATTERNS: list[tuple[re.Pattern, Callable[[random.Random], str]]] = [
    (re.compile(r"email", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.EMAIL", lambda rng: "test@example.com")),
    (re.compile(r"phone|telephone|mobile", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.PHONE", lambda rng: "+1-555-0100")),
    (re.compile(r"date|timestamp", re.I), GENERATORS.get("ICE.METADATA.TIMESTAMP", lambda rng: "2024-01-01")),
    (re.compile(r"identifier|uuid|key|token", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID", lambda rng: "id-1")),
    (re.compile(r"url|uri|link", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.REF.URL", lambda rng: "https://example.com")),
    (re.compile(r"amount|price|monetary|cost", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT", lambda rng: "$100.00")),
    (re.compile(r"boolean|flag|indicator", re.I), lambda rng: rng.choice(["true", "false"])),
    (re.compile(r"percentage|ratio|rate", re.I), lambda rng: f"{round(rng.uniform(0, 100), 2)}%"),
    (re.compile(r"name|person|human", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.NAME.PERSON", lambda rng: "John Doe")),
    (re.compile(r"address|street|location", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.ADDRESS", lambda rng: "123 Main St")),
    (re.compile(r"country|nation", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY", lambda rng: "USA")),
    (re.compile(r"city|town", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY", lambda rng: "New York")),
    (re.compile(r"status|state", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.STATUS", lambda rng: "active")),
    (re.compile(r"count|quantity|number", re.I), lambda rng: str(rng.randint(0, 100000))),
    (re.compile(r"score|rating", re.I), lambda rng: f"{round(rng.uniform(0, 10), 2)}"),
    (re.compile(r"description|text|note|comment", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT", lambda rng: "No comment.")),
    (re.compile(r"version", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.SEMANTIC_VERSION", lambda rng: "1.0.0")),
    (re.compile(r"hash|checksum|digest", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID", lambda rng: "abcdef")),
    (re.compile(r"code|identifier|id$", re.I), lambda rng: f"CODE-{rng.randint(100, 999)}"),
]


def _infer_generator(cat) -> Callable[[random.Random], str] | None:
    """Infer a generator from category description and common_names."""
    search_text = f"{cat.label} {getattr(cat, 'description', '') or ''} {getattr(cat, 'common_names', '') or ''}"
    for pattern, gen in _INFERENCE_PATTERNS:
        if pattern.search(search_text):
            return gen
    return None
