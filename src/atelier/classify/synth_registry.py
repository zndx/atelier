"""Extensible generator registry with coverage tracking.

Builds a registry for the user's taxonomy by layering:
1. Template generators from enrichment prototype_values (real values)
2. Inferred generators from category metadata (description/common_names)

Enrichment data (from Qdrant or JSON export) is always required —
without it, the SVM cannot emit user-provided terminology.  Real
prototype values are the primary source; keyword-inferred format-shaped
values only fill gaps.  (The hand-coded ICE generator library was
retired 2026-08-13 — offline synthetic corpora are Aegir's domain,
consumed via ``external/sdg-corpora``.)
"""

from __future__ import annotations

import re
import random
import string
from collections.abc import Callable
from dataclasses import dataclass

@dataclass(frozen=True)
class GeneratorSpec:
    """A registered generator with provenance tracking."""
    code: str
    generator: Callable[[random.Random], str]
    source: str       # "template" | "inferred"


class GeneratorRegistry:
    """Extensible registry mapping category codes to value generators.

    Layers: template (from real sample values — enrichment
    ``prototype_values`` or operator-supplied ``value_templates``) >
    inferred (keyword match on category metadata).  The historical
    hand-coded ICE generator library was retired 2026-08-13 — synthetic
    corpus generation is Aegir's responsibility (consumed via
    ``external/sdg-corpora``); this registry only backstops the
    per-vocabulary training path with real-value templates.
    """

    def __init__(self) -> None:
        self._specs: dict[str, GeneratorSpec] = {}

    def register(self, code: str, gen: Callable[[random.Random], str], source: str = "template") -> None:
        """Register a generator. Later registrations with lower-priority source don't overwrite."""
        _priority = {"template": 2, "inferred": 1}
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
        """Report coverage: code → source or "missing" for every taggable node."""
        all_cats = getattr(category_set, "all_categories", category_set.categories)
        report: dict[str, str] = {}
        for cat in all_cats:
            spec = self._specs.get(cat.code)
            report[cat.code] = spec.source if spec else "missing"
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

        1. Template generators from provided value_templates
        2. Inferred generators from category metadata
        """
        registry = cls()
        all_cats = getattr(category_set, "all_categories", category_set.categories)
        all_codes = frozenset(c.code for c in all_cats)

        # Layer 1: template generators from real sample values
        if value_templates:
            for code, values in value_templates.items():
                if code in all_codes and len(values) >= 3:
                    registry.register(code, _make_template_generator(values), "template")

        # Layer 2: inferred generators from category metadata
        for cat in all_cats:
            if cat.code in registry:
                continue
            gen = _infer_generator(cat)
            if gen:
                registry.register(cat.code, gen, "inferred")

        return registry

    @classmethod
    def from_enrichment_payloads(
        cls,
        payloads: dict[str, dict],
        category_set,
    ) -> GeneratorRegistry:
        """Build a registry for user-taxonomy codes from enrichment payloads.

        Two-layer architecture:

          1. **Template** (highest priority): ``prototype_values`` with
             ≥3 items produce a template generator via
             ``_make_template_generator`` — real values sampled from
             the enrichment pass, mildly perturbed.
          2. **Inferred** (fallback): ``_infer_generator`` from category
             metadata (description, common_names) — keyword-matched
             format-shaped values for codes without prototypes.

        Enrichment data is the primary contract: real prototype values
        drive generation.  (The historical layer that mapped ICE
        hand-coded generators onto user codes was retired with the
        ICE generator library — corpus generation is Aegir's domain.)
        """
        registry = cls()
        all_cats = getattr(category_set, "all_categories", category_set.categories)

        # Payloads are keyed by mnemonic (e.g. "EMAIL"); category codes
        # are dot-codes (e.g. "1.1.1.9.3.1").  Bridge via category abbrev.
        abbrev_for: dict[str, str] = {}
        for cat in all_cats:
            if cat.abbrev:
                abbrev_for[cat.code] = cat.abbrev

        for cat in all_cats:
            code = cat.code
            mnemonic = abbrev_for.get(code, "")
            payload = payloads.get(mnemonic, {})

            # Layer 1: template generator from prototype_values
            prototypes = payload.get("prototype_values", [])
            if len(prototypes) >= 3:
                registry.register(code, _make_template_generator(prototypes), "template")

            # Layer 2: inferred from category metadata
            if code not in registry:
                gen = _infer_generator(cat)
                if gen:
                    registry.register(code, gen, "inferred")

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

# Pattern: keyword regex → format-shaped generator.  Ordered from most
# specific to least — first match wins.  These are *gap fillers*: they
# emit values with the right shape (rng-varied so training columns
# aren't constant) but carry no curated semantics.  Real value
# distributions come from enrichment ``prototype_values`` (layer 1) or,
# for offline corpora, from Aegir's real-world-sourced releases.
_INFERENCE_PATTERNS: list[tuple[re.Pattern, Callable[[random.Random], str]]] = [
    # Sensitive PII — specific before general
    (re.compile(r"ssn|social.security", re.I), lambda rng: f"{rng.randint(100, 899):03d}-{rng.randint(10, 99):02d}-{rng.randint(1000, 9999):04d}"),
    (re.compile(r"cvv|cv2|cvc\d?|security.code|card.verification", re.I), lambda rng: f"{rng.randint(100, 999)}"),
    (re.compile(r"credit.card|pan\b|card.number|payment.card", re.I), lambda rng: "4" + "".join(str(rng.randint(0, 9)) for _ in range(15))),
    (re.compile(r"\biban\b|bank.account", re.I), lambda rng: f"GB{rng.randint(10, 99)}NWBK{rng.randint(10**13, 10**14 - 1)}"),
    (re.compile(r"salary|income|compensation|wage", re.I), lambda rng: f"{rng.randint(30, 250) * 1000 + rng.choice([0, 500]):.2f}"),
    (re.compile(r"passport", re.I), lambda rng: f"{chr(rng.randint(65, 90))}{chr(rng.randint(65, 90))}{rng.randint(1000000, 9999999)}"),
    (re.compile(r"driver.?s?.license|drv.?lic", re.I), lambda rng: f"{chr(rng.randint(65, 90))}{rng.randint(10000000, 99999999)}"),
    (re.compile(r"password|secret|credential", re.I), lambda rng: f"vault://secret/data/key-{rng.randint(100, 999)}"),
    (re.compile(r"\bimei\b", re.I), lambda rng: "35" + "".join(str(rng.randint(0, 9)) for _ in range(13))),
    (re.compile(r"\bmac.address|mac.addr\b", re.I), lambda rng: ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))),
    (re.compile(r"\bip.address|ipv[46]|ip.addr\b", re.I), lambda rng: f"10.{rng.randint(0, 255)}.{rng.randint(0, 255)}.{rng.randint(1, 254)}"),
    # Contact
    (re.compile(r"email|e.mail", re.I), lambda rng: f"user{rng.randint(1, 9999)}@example.{rng.choice(['com', 'org', 'net'])}"),
    (re.compile(r"phone|telephone|mobile", re.I), lambda rng: f"+1-555-{rng.randint(100, 999):03d}-{rng.randint(1000, 9999):04d}"),
    (re.compile(r"address|street|location", re.I), lambda rng: f"{rng.randint(1, 9999)} {rng.choice(['Main', 'Oak', 'Cedar', 'Park', 'Lake'])} {rng.choice(['St', 'Ave', 'Blvd', 'Rd'])}"),
    # Temporal
    (re.compile(r"date|timestamp|datetime", re.I), lambda rng: f"202{rng.randint(0, 6)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"),
    # Identifiers
    (re.compile(r"\buuid\b|unique.id", re.I), lambda rng: f"{rng.getrandbits(32):08x}-{rng.getrandbits(16):04x}-4{rng.getrandbits(12):03x}-{rng.getrandbits(16):04x}-{rng.getrandbits(48):012x}"),
    (re.compile(r"url|uri|link|href", re.I), lambda rng: f"https://example.{rng.choice(['com', 'org'])}/{rng.choice(['docs', 'api', 'items'])}/{rng.randint(1, 9999)}"),
    # Financial
    (re.compile(r"amount|price|monetary|cost|payment", re.I), lambda rng: f"${rng.uniform(1, 5000):.2f}"),
    # Descriptive
    (re.compile(r"boolean|flag|indicator", re.I), lambda rng: rng.choice(["true", "false"])),
    (re.compile(r"percentage|ratio|rate", re.I), lambda rng: f"{round(rng.uniform(0, 100), 2)}%"),
    (re.compile(r"name|person|human", re.I), lambda rng: f"{rng.choice(['Alex', 'Sam', 'Jordan', 'Casey', 'Morgan', 'Riley'])} {rng.choice(['Smith', 'Chen', 'Garcia', 'Okafor', 'Patel', 'Kim'])}"),
    (re.compile(r"country|nation", re.I), lambda rng: rng.choice(["USA", "Canada", "Germany", "Japan", "Brazil", "India"])),
    (re.compile(r"city|town", re.I), lambda rng: rng.choice(["New York", "Toronto", "Berlin", "Osaka", "Recife", "Pune"])),
    (re.compile(r"region|province", re.I), lambda rng: rng.choice(["California", "Ontario", "Bavaria", "Kansai", "Pernambuco"])),
    (re.compile(r"status|state", re.I), lambda rng: rng.choice(["active", "inactive", "pending", "archived"])),
    (re.compile(r"count|quantity|number", re.I), lambda rng: str(rng.randint(0, 100000))),
    (re.compile(r"score|rating", re.I), lambda rng: f"{round(rng.uniform(0, 10), 2)}"),
    (re.compile(r"description|text|note|comment", re.I), lambda rng: rng.choice(["Reviewed and approved.", "Pending follow-up.", "No issues found.", "Escalated to owner."])),
    (re.compile(r"version", re.I), lambda rng: f"{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 40)}"),
    (re.compile(r"hash|checksum|digest", re.I), lambda rng: f"{rng.getrandbits(160):040x}"),
    (re.compile(r"code|identifier|id$", re.I), lambda rng: f"CODE-{rng.randint(100, 999)}"),
]


def _infer_generator(cat) -> Callable[[random.Random], str] | None:
    """Infer a generator from category description and common_names."""
    search_text = f"{cat.label} {getattr(cat, 'description', '') or ''} {getattr(cat, 'common_names', '') or ''}"
    for pattern, gen in _INFERENCE_PATTERNS:
        if pattern.search(search_text):
            return gen
    return None
