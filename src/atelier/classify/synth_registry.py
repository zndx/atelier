"""Extensible generator registry with coverage tracking.

Builds a registry for the user's taxonomy by layering:
1. ICE hand-coded generators matched via enrichment metadata
2. Template generators from enrichment prototype_values
3. Inferred generators from category metadata (description/common_names)

Enrichment data (from Qdrant or JSON export) is always required —
without it, the SVM cannot emit user-provided terminology.  ICE
generators from ``synth_generators.GENERATORS`` are the gold-standard
value library; enrichment metadata (label, description, name_hints)
is used to match each user code to the best ICE generator.
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

        1. Hand-coded generators from GENERATORS
        2. Template generators from provided value_templates
        3. Inferred generators from category metadata
        """
        registry = cls()
        all_cats = getattr(category_set, "all_categories", category_set.categories)
        all_codes = frozenset(c.code for c in all_cats)

        # Layer 1: hand-coded generators
        for code in all_codes:
            if code in GENERATORS:
                registry.register(code, GENERATORS[code], "hand-coded")

        # Layer 2: template generators from real sample values
        if value_templates:
            for code, values in value_templates.items():
                if code in all_codes and len(values) >= 3:
                    registry.register(code, _make_template_generator(values), "template")

        # Layer 3: inferred generators from category metadata
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

        Three-layer architecture:

          1. **ICE-matched** (highest priority): enrichment metadata
             (label, description, name_hints) is searched against
             ``_INFERENCE_PATTERNS`` to find the best ICE hand-coded
             generator.  This gives user codes access to the gold-
             standard generator library without an alignment pipeline.
          2. **Template** (medium): ``prototype_values`` with ≥3 items
             produce a template generator via ``_make_template_generator``.
          3. **Inferred** (lowest): ``_infer_generator`` from category
             metadata (description, common_names) — keyword fallback.

        Enrichment data is always required.  Codes without a payload
        entry still get template/inferred coverage if the category_set
        carries metadata, but the primary contract is: enrichment maps
        ICE generators to user codes.
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

            # Layer 1: match ICE hand-coded generator via enrichment metadata
            ice_gen = _match_ice_generator(payload)
            if ice_gen is not None:
                registry.register(code, ice_gen, "hand-coded")

            # Layer 2: template generator from prototype_values
            prototypes = payload.get("prototype_values", [])
            if len(prototypes) >= 3:
                registry.register(code, _make_template_generator(prototypes), "template")

            # Layer 3: inferred from category metadata
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

# Pattern: keyword regex → generator function.  Ordered from most
# specific to least — first match wins.  ICE generators are the
# gold-standard value library; patterns match enrichment metadata
# (label, description, name_hints) to the best ICE generator.
_INFERENCE_PATTERNS: list[tuple[re.Pattern, Callable[[random.Random], str]]] = [
    # Sensitive PII — specific before general
    (re.compile(r"ssn|social.security", re.I), GENERATORS.get("ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN", lambda rng: "123-45-6789")),
    (re.compile(r"cvv|cv2|cvc\d?|security.code|card.verification", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.CVV", lambda rng: "123")),
    (re.compile(r"credit.card|pan\b|card.number|payment.card", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN", lambda rng: "4111111111111111")),
    (re.compile(r"\biban\b|bank.account", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN", lambda rng: "GB29NWBK60161331926819")),
    (re.compile(r"salary|income|compensation|wage", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.INCOME.SALARY", lambda rng: "75000.00")),
    (re.compile(r"passport", re.I), GENERATORS.get("ICE.SENSITIVE.PID.IDENTITY.GOVID.PASSPORT", lambda rng: "AB1234567")),
    (re.compile(r"driver.?s?.license|drv.?lic", re.I), GENERATORS.get("ICE.SENSITIVE.PID.IDENTITY.GOVID.DLN", lambda rng: "D12345678")),
    (re.compile(r"password|secret|credential", re.I), GENERATORS.get("ICE.SENSITIVE.TECHNICAL.SECRET", lambda rng: "vault://secret/data/key")),
    (re.compile(r"\bimei\b", re.I), GENERATORS.get("ICE.SENSITIVE.TECHNICAL.DEVID.IMEI", lambda rng: "353456789012345")),
    (re.compile(r"\bmac.address|mac.addr\b", re.I), GENERATORS.get("ICE.SENSITIVE.TECHNICAL.DEVID.MAC", lambda rng: "aa:bb:cc:dd:ee:ff")),
    (re.compile(r"\bip.address|ipv[46]|ip.addr\b", re.I), GENERATORS.get("ICE.SENSITIVE.TECHNICAL.IPADDR", lambda rng: "192.168.1.1")),
    # Contact
    (re.compile(r"email|e.mail", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.EMAIL", lambda rng: "test@example.com")),
    (re.compile(r"phone|telephone|mobile", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.PHONE", lambda rng: "+1-555-0100")),
    (re.compile(r"address|street|location", re.I), GENERATORS.get("ICE.SENSITIVE.PID.CONTACT.ADDRESS", lambda rng: "123 Main St")),
    # Temporal
    (re.compile(r"date|timestamp|datetime", re.I), GENERATORS.get("ICE.METADATA.TIMESTAMP", lambda rng: "2024-01-01")),
    # Identifiers
    (re.compile(r"\buuid\b|unique.id", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.ID", lambda rng: "id-1")),
    (re.compile(r"url|uri|link|href", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.REF.URL", lambda rng: "https://example.com")),
    # Financial
    (re.compile(r"amount|price|monetary|cost|payment", re.I), GENERATORS.get("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.TXNAMT", lambda rng: "$100.00")),
    # Descriptive
    (re.compile(r"boolean|flag|indicator", re.I), lambda rng: rng.choice(["true", "false"])),
    (re.compile(r"percentage|ratio|rate", re.I), lambda rng: f"{round(rng.uniform(0, 100), 2)}%"),
    (re.compile(r"name|person|human", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.NAME.PERSON", lambda rng: "John Doe")),
    (re.compile(r"country|nation", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.GEO.COUNTRY", lambda rng: "USA")),
    (re.compile(r"city|town", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.GEO.CITY", lambda rng: "New York")),
    (re.compile(r"region|province", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.GEO.REGION", lambda rng: "California")),
    (re.compile(r"status|state", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESCRIPTIVE.CATEGORICAL.STATUS", lambda rng: "active")),
    (re.compile(r"count|quantity|number", re.I), lambda rng: str(rng.randint(0, 100000))),
    (re.compile(r"score|rating", re.I), lambda rng: f"{round(rng.uniform(0, 10), 2)}"),
    (re.compile(r"description|text|note|comment", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESCRIPTIVE.TEXT.COMMENT", lambda rng: "No comment.")),
    (re.compile(r"version", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.SEMANTIC_VERSION", lambda rng: "1.0.0")),
    (re.compile(r"hash|checksum|digest", re.I), GENERATORS.get("ICE.NONSENSITIVE.DESIGNATIVE.CODE.HASH_ID", lambda rng: "abcdef")),
    (re.compile(r"code|identifier|id$", re.I), lambda rng: f"CODE-{rng.randint(100, 999)}"),
]


def _match_ice_generator(payload: dict) -> Callable[[random.Random], str] | None:
    """Match a user code's enrichment payload against ICE generators.

    Searches the payload's label, description, and name_hints against
    ``_INFERENCE_PATTERNS``.  Returns the first matching ICE generator
    or None.
    """
    label = payload.get("label", "")
    description = payload.get("description", "")
    name_hints = payload.get("name_hints", [])
    if isinstance(name_hints, list):
        hints_text = " ".join(str(h) for h in name_hints)
    else:
        hints_text = ""
    search_text = f"{label} {description} {hints_text}"
    if not search_text.strip():
        return None
    for pattern, gen in _INFERENCE_PATTERNS:
        if pattern.search(search_text):
            return gen
    return None


def _infer_generator(cat) -> Callable[[random.Random], str] | None:
    """Infer a generator from category description and common_names."""
    search_text = f"{cat.label} {getattr(cat, 'description', '') or ''} {getattr(cat, 'common_names', '') or ''}"
    for pattern, gen in _INFERENCE_PATTERNS:
        if pattern.search(search_text):
            return gen
    return None
