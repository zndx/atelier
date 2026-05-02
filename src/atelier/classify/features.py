"""Feature extraction for column classification.

Extracts 12 discrete, ablatable features from column metadata to support
SAGE feature importance analysis and multi-method classification.

Ported from signals/src/sigint/features.py — adapted to accept raw metadata
dicts instead of ColumnSample objects (agent-mediated sampling provides dicts).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


# ── Pattern detectors ────────────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern] = {
    "email_pattern": re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    ),
    "phone_pattern": re.compile(
        # Require phone-like formatting: a + prefix, parenthesized area
        # code, or digit groups separated by hyphens/spaces/dots.
        # Bare digit strings like "75000" or "1234.56" must NOT match.
        r"^(?:"
        r"\+[\d\s\-\(\)\.]{7,19}"                        # +1-555-123-4567
        r"|\([\d]{2,4}\)[\s\-\.]?[\d\s\-\.]{4,15}"       # (555) 987-6543
        r"|[\d]{1,4}[\s\-]+[\d][\d\s\-\.]{4,15}"         # 555-123-4567, 1 800 555 0199
        r"|[\d]{2,4}\.[\d]{2,4}\.[\d]{4,}"                # 555.123.4567
        r")$"
    ),
    "ssn_pattern": re.compile(
        r"^\d{3}-\d{2}-\d{4}$"
    ),
    "ipv4_pattern": re.compile(
        r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$"
    ),
    "uuid_pattern": re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
    ),
    "date_iso_pattern": re.compile(
        r"^\d{4}-\d{2}-\d{2}"
    ),
    "datetime_iso_pattern": re.compile(
        r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"
    ),
    "url_pattern": re.compile(
        r"^https?://"
    ),
    "credit_card_pattern": re.compile(
        r"^\d{13,19}$"
    ),
    # ── Expanded detectors (Phase 1 — heuristics library) ─────────
    "mac_address_pattern": re.compile(
        r"^([0-9A-Fa-f]{2}[:\-]){5}[0-9A-Fa-f]{2}$"
    ),
    "iban_pattern": re.compile(
        r"^[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7,}$"
    ),
    "postal_code_pattern": re.compile(
        r"^\d{5}(-\d{4})?$"
    ),
    "monetary_pattern": re.compile(
        r"^[\$\€\£]\s?[\d,]+\.\d{2}$"
    ),
    "hex_hash_pattern": re.compile(
        r"^[0-9a-fA-F]{40,128}$"  # min 40 = SHA-1; excludes MD5/UUID-no-dashes (32)
    ),
    "semver_pattern": re.compile(
        r"^\d+\.\d+\.\d+([.\-+].+)?$"
    ),
    "iso_currency_pattern": re.compile(
        r"^[A-Z]{3}$"
    ),
    # ── New patterns from artifact analysis ──────────────────────
    "vin_pattern": re.compile(
        r"^[A-HJ-NPR-Z0-9]{17}$"  # 17 chars, excludes I/O/Q
    ),
    "iata_pattern": re.compile(
        r"^[A-Z]{3}$"  # Same regex as currency; validator distinguishes
    ),
    "bcrypt_pattern": re.compile(
        r"^\$2[aby]\$\d{2}\$"  # bcrypt password hash prefix
    ),
    "mrn_pattern": re.compile(
        r"^MRN-\d{4,}$", re.IGNORECASE  # Medical Record Number
    ),
    "license_plate_pattern": re.compile(
        r"^[A-Z]{2,3}[-\s]?\d{3,4}$"  # e.g. RUS-3462
    ),
    "eth_address_pattern": re.compile(
        r"^0x[0-9a-fA-F]{40}$"  # Ethereum address
    ),
    "ssh_key_pattern": re.compile(
        r"^ssh-(rsa|ed25519|ecdsa)\s"  # SSH public key prefix
    ),
    "certificate_pattern": re.compile(
        r"^-----BEGIN (CERTIFICATE|PUBLIC KEY|RSA PRIVATE KEY)-----"
    ),
    # ── Mobile / device identifiers ─────────────────────────────
    # UDID: Apple device identifier — 32 hex (legacy) or 40 hex (UDID2).
    # Anchored to exactly those two lengths to avoid colliding with the
    # quarantined hex_hash_pattern (40-128 hex range).  Even so, 40-hex
    # UDID and 40-hex SHA-1 are visually identical; per-detector mass
    # is capped low (DEFAULT_PATTERN_MAP) so name-match + sibling
    # context still dominate fusion when the column name disambiguates.
    "udid_pattern": re.compile(
        r"^[0-9a-fA-F]{32}$|^[0-9a-fA-F]{40}$"
    ),
    # ICCID: SIM card identifier — 19 or 20 digits with Luhn checksum.
    "iccid_pattern": re.compile(
        r"^\d{19,20}$"
    ),
    # IMEI: Mobile device identifier — exactly 15 digits with Luhn.
    "imei_pattern": re.compile(
        r"^\d{15}$"
    ),
}

# ── Post-regex validators ────────────────────────────────────────────
#
# Regexes optimize for recall; validators add precision.  A value must
# pass BOTH the regex AND the validator (if one exists) to count.

_ISO_4217_CODES: frozenset[str] = frozenset({
    "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "HKD",
    "NZD", "SEK", "KRW", "SGD", "NOK", "MXN", "INR", "RUB", "ZAR",
    "TRY", "BRL", "TWD", "DKK", "PLN", "THB", "IDR", "HUF", "CZK",
    "ILS", "CLP", "PHP", "AED", "COP", "SAR", "MYR", "RON", "PEN",
    "BHD", "KWD", "QAR",
})


def _luhn_check(digits: str) -> bool:
    """Luhn algorithm — validates credit card check digit."""
    digits = digits.strip()
    if not digits.isdigit():
        return False
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _is_valid_ipv4(v: str) -> bool:
    """Check all four octets are 0-255."""
    parts = v.strip().split(".")
    if len(parts) != 4:
        return False
    return all(p.isdigit() and 0 <= int(p) <= 255 for p in parts)


def _is_plausible_date(v: str) -> bool:
    """Check that the YYYY-MM-DD prefix has month 01-12 and day 01-31."""
    v = v.strip()
    if len(v) < 10:
        return False
    try:
        month = int(v[5:7])
        day = int(v[8:10])
        return 1 <= month <= 12 and 1 <= day <= 31
    except (ValueError, IndexError):
        return False


def _is_iso_currency(v: str) -> bool:
    """Check against ISO 4217 whitelist."""
    return v.strip() in _ISO_4217_CODES


# Common IATA airport codes (~350 busiest worldwide).
_IATA_CODES: frozenset[str] = frozenset({
    "ATL", "PEK", "LAX", "DXB", "HND", "ORD", "LHR", "PVG", "CDG",
    "DFW", "AMS", "FRA", "IST", "CAN", "JFK", "SIN", "DEN", "ICN",
    "BKK", "SFO", "DEL", "CGK", "KUL", "MAD", "CTU", "BCN", "LAS",
    "BOM", "MIA", "YYZ", "MUC", "SYD", "FCO", "NRT", "MNL", "HKG",
    "MEX", "EWR", "ZRH", "GRU", "TPE", "SEA", "CLT", "MCO", "PHX",
    "IAH", "KMG", "MSP", "BOS", "DTW", "SZX", "XIY", "LGW", "MEL",
    "SHA", "CKG", "HAK", "TFU", "WUH", "NKG", "HGH", "CSX", "TAO",
    "SVO", "DME", "LED", "AUH", "DOH", "JED", "RUH", "CAI", "ADD",
    "JNB", "CPT", "NBO", "LOS", "ACC", "DAR", "CMN", "ALG", "TUN",
    "BOG", "LIM", "SCL", "EZE", "GIG", "BSB", "PTY", "SJO", "UIO",
})


def _is_iata_code(v: str) -> bool:
    """Check against IATA airport code whitelist."""
    return v.strip() in _IATA_CODES


def _is_valid_vin(v: str) -> bool:
    """Basic VIN validation: 17 chars, no I/O/Q, transliteration check."""
    v = v.strip().upper()
    if len(v) != 17:
        return False
    invalid = set("IOQ")
    return not any(c in invalid for c in v)


_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "credit_card_pattern": _luhn_check,
    "ipv4_pattern": _is_valid_ipv4,
    "date_iso_pattern": _is_plausible_date,
    "datetime_iso_pattern": _is_plausible_date,
    "iso_currency_pattern": _is_iso_currency,
    "iata_pattern": _is_iata_code,
    "vin_pattern": _is_valid_vin,
    # ICCID and IMEI both terminate in a Luhn check digit; reuse the
    # existing validator.  Without these, plain numeric strings of the
    # right length would false-positive the regex.
    "iccid_pattern": _luhn_check,
    "imei_pattern": _luhn_check,
}


FEATURE_NAMES: list[str] = [
    "column_name",
    "column_type",
    "sample_values",
    "cardinality",
    "null_ratio",
    "value_entropy",
    "pattern_signals",
    "avg_value_length",
    "numeric_ratio",
    "sibling_context",
    "source_table",
    "value_description",
    # Canonical ICE.* metadata (label / description / common_names /
    # ontological path) for each fired pattern.  Surfaced as its own
    # ablatable feature so SAGE/SHAP can attribute classification mass
    # to the publicly-grounded ontology prior independently of the
    # raw embedding text.  See mass_functions.lookup_pattern_ontology
    # and fixtures/PROVENANCE.md.
    "ontology_priors",
]


# ── Generic name detection ──────────────────────────────────────────

_GENERIC_NAME_RE = re.compile(
    r"^(col\d*|column\d*|field\d*|var\d*|unnamed.*|\d+|_|)$",
    re.IGNORECASE,
)


def _is_generic_name(name: str) -> bool:
    """Detect positional/placeholder column names that carry no semantic signal."""
    return bool(_GENERIC_NAME_RE.match(name.strip()))


def _generate_value_description(
    values: list[str],
    col_type: str | None,  # noqa: ARG001 — reserved for future type-aware descriptions
    patterns: dict[str, float] | list[str],
) -> str:
    """Generate a natural-language description based on value shape."""
    if not values:
        return ""

    if "datetime_iso_pattern" in patterns:
        return "column of datetime/timestamp values"
    if "date_iso_pattern" in patterns:
        return "column of date values in YYYY-MM-DD format"
    if "email_pattern" in patterns:
        return "column of email addresses"
    if "url_pattern" in patterns:
        return "column of URLs or web links"
    if "uuid_pattern" in patterns:
        return "column of UUID identifiers"

    num_ratio = _numeric_ratio(values)
    if num_ratio > 0.8:
        try:
            nums = [float(v.strip().replace(",", "")) for v in values if v.strip()]
            if all(n == int(n) for n in nums) and len(nums) >= 3:
                diffs = [nums[i + 1] - nums[i] for i in range(len(nums) - 1)]
                if all(abs(d - diffs[0]) < 0.01 for d in diffs):
                    return "column of sequential integers, likely identifiers or index"
        except (ValueError, OverflowError):
            pass
        if any("." in v for v in values):
            return "column of decimal numeric measurements"
        return "column of integer values"

    avg_len = sum(len(v) for v in values) / len(values) if values else 0
    distinct = len(set(values))
    distinct_ratio = distinct / len(values) if values else 1.0

    if avg_len > 100:
        return "column of long text content, descriptions or comments"
    if avg_len > 40:
        return "column of text phrases or sentences"
    if distinct_ratio < 0.5 and avg_len < 20:
        return "column of categorical labels or codes"
    if avg_len < 20 and num_ratio < 0.2:
        return "column of short text labels or names"

    return "column of text values"


_PHONE_SUPPRESSORS: frozenset[str] = frozenset({
    "ssn_pattern",
    "date_iso_pattern",
    "datetime_iso_pattern",
    "credit_card_pattern",
    "ipv4_pattern",
    "postal_code_pattern",
    "monetary_pattern",
    "iban_pattern",
})
"""Patterns that, when present, suppress phone_pattern.

The phone regex requires phone-like formatting (+ prefix, parens, or
digit-group separators), but can still overlap with other structured
patterns. When a more specific pattern already fired on the same values,
the phone match is almost certainly a false positive.
"""


def detect_patterns(values: list[str]) -> dict[str, float]:
    """Detect which value patterns are present in a sample.

    Returns a dict mapping pattern name → match fraction (0.0–1.0).
    A pattern fires when >= 1/3 of sample values match.  Post-regex
    validators (see :data:`_VALIDATORS`) add precision: a value must pass
    both the regex and the validator to count.

    The match fraction is used by :func:`pattern_to_mass` to scale
    evidence mass — a 95% match produces more mass than a 35% match.

    The ``phone_pattern`` is suppressed when a more specific pattern
    (date, SSN, credit card, etc.) already matched — see
    :data:`_PHONE_SUPPRESSORS`.
    """
    if not values:
        return {}
    n = len(values)
    hits: dict[str, float] = {}
    threshold = max(1, n // 3)
    for name, pat in _PATTERNS.items():
        validator = _VALIDATORS.get(name)
        if validator:
            match_count = sum(
                1 for v in values
                if pat.match(v.strip()) and validator(v.strip())
            )
        else:
            match_count = sum(1 for v in values if pat.match(v.strip()))
        if match_count >= threshold:
            hits[name] = match_count / n

    # Suppress phone_pattern when a more specific digit-heavy pattern fired.
    if "phone_pattern" in hits and _hits_any_suppressor(hits):
        del hits["phone_pattern"]

    return dict(sorted(hits.items()))


def _hits_any_suppressor(hits: dict[str, float]) -> bool:
    """True when *hits* contains any pattern that suppresses phone_pattern."""
    return bool(_PHONE_SUPPRESSORS.intersection(hits))


def _shannon_entropy(values: list[str]) -> float:
    """Shannon entropy of value lengths (bits)."""
    if not values:
        return 0.0
    lengths = [len(v) for v in values]
    counts = Counter(lengths)
    total = len(lengths)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return round(entropy, 4)


def _numeric_ratio(values: list[str]) -> float:
    """Fraction of values parseable as a number."""
    if not values:
        return 0.0
    numeric = 0
    for v in values:
        v = v.strip()
        try:
            float(v.replace(",", ""))
            numeric += 1
        except (ValueError, OverflowError):
            pass
    return round(numeric / len(values), 4)


# ── Categorical enum detection ──────────────────────────────────────
#
# When a column has low cardinality and its values match a known
# enumeration set, we can classify with high confidence even without
# column name signal.

_CATEGORICAL_SETS: dict[str, tuple[frozenset[str], str]] = {
    # name → (value_set, description_hint)
    "blood_type": (frozenset({
        "A", "B", "AB", "O", "A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-",
    }), "blood type values"),
    "device_type": (frozenset({
        "MOBILE", "DESKTOP", "TABLET", "WATCH", "TV", "IOT", "SMARTPHONE",
        "LAPTOP", "PC", "WEARABLE",
    }), "device/platform type"),
    "quarter": (frozenset({
        "Q1", "Q2", "Q3", "Q4",
    }), "fiscal/calendar quarter"),
    "boolean": (frozenset({
        "TRUE", "FALSE", "YES", "NO", "Y", "N", "1", "0",
        "ACTIVE", "INACTIVE", "ENABLED", "DISABLED",
    }), "boolean/status flag"),
    "gender": (frozenset({
        "M", "F", "MALE", "FEMALE", "OTHER", "NON-BINARY", "NB",
        "PREFER NOT TO SAY", "UNKNOWN",
    }), "gender/sex"),
    "priority": (frozenset({
        "LOW", "MEDIUM", "HIGH", "CRITICAL", "URGENT", "P1", "P2", "P3", "P4",
    }), "priority level"),
    "continent": (frozenset({
        "AFRICA", "ANTARCTICA", "ASIA", "EUROPE", "NORTH AMERICA",
        "SOUTH AMERICA", "OCEANIA",
    }), "continent"),
}


def detect_categorical_enum(values: list[str]) -> dict[str, float]:
    """Match column values against known categorical enumeration sets.

    Returns {enum_name: match_fraction} for sets where ≥70% of non-empty
    values are members.
    """
    if not values:
        return {}
    clean = [v.strip().upper() for v in values if v and v.strip()]
    if len(clean) < 3:
        return {}

    hits: dict[str, float] = {}
    value_set = set(clean)
    for enum_name, (enum_values, _desc) in _CATEGORICAL_SETS.items():
        overlap = len(value_set & enum_values)
        if overlap > 0 and overlap / len(value_set) >= 0.7:
            hits[enum_name] = overlap / len(value_set)
    return hits


# ── Sibling domain clustering ──────────────────────────────────────
#
# When sibling columns form a recognizable domain cluster (e.g. health,
# payment, identity), we can infer the domain for opaque-named columns.

_DOMAIN_CLUSTERS: dict[str, frozenset[str]] = {
    "health": frozenset({
        "blood_type", "diagnosis", "prescription", "allergy", "medication",
        "medical_record", "patient", "vital", "lab_result", "immunization",
        "mrn", "icd", "encounter", "clinical", "prognosis", "symptom",
    }),
    "payment": frozenset({
        "card_number", "cvv", "expiry", "cardholder", "billing",
        "payment", "transaction", "pan", "credit_card", "debit",
        "merchant", "authorization", "refund", "invoice",
    }),
    "identity": frozenset({
        "first_name", "last_name", "name", "dob", "ssn", "passport",
        "driver_license", "gender", "nationality", "ethnicity",
        "marital_status", "birth", "citizen",
    }),
    "geo": frozenset({
        "latitude", "longitude", "address", "city", "country",
        "postal_code", "zip", "timezone", "region", "state",
        "province", "county", "district",
    }),
    "technical": frozenset({
        "ip_address", "mac_address", "hostname", "url", "api_key",
        "password", "token", "certificate", "ssh_key", "user_agent",
        "session_id", "device_id",
    }),
}


def detect_sibling_domain(
    column_name: str, sibling_names: list[str]
) -> dict[str, float]:
    """Detect domain clusters from sibling column names.

    Returns {domain: match_score} where score is the fraction of
    siblings matching that domain's keyword set.
    """
    if not sibling_names:
        return {}

    # Normalize sibling names to lowercase word sets
    sibling_words: set[str] = set()
    for sib in sibling_names:
        sibling_words.update(sib.lower().replace("_", " ").split())

    hits: dict[str, float] = {}
    for domain, keywords in _DOMAIN_CLUSTERS.items():
        overlap = len(sibling_words & keywords)
        if overlap >= 2:  # At least 2 keyword matches
            hits[domain] = overlap / max(len(keywords), 1)
    return hits


# ── ColumnFeatures ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnFeatures:
    """Discrete, ablatable features extracted from a column sample.

    Each feature can be independently masked for SAGE analysis (see
    ``FEATURE_NAMES``).
    """

    column_name_humanized: str
    column_type: str | None
    sample_values_text: str | None
    cardinality: int | None
    null_ratio: float | None
    value_entropy: float | None
    pattern_signals: dict[str, float] = field(default_factory=dict)
    avg_value_length: float | None = None
    numeric_ratio: float | None = None
    sibling_names: list[str] = field(default_factory=list)
    source_table: str | None = None
    value_description: str = ""
    is_generic_name: bool = False
    # Canonical ICE.* metadata for each fired pattern, resolved from
    # the universal vocabulary at extraction time.  Populated by
    # ``mass_functions.lookup_pattern_ontology`` — each entry is a
    # dict with keys: pattern, code, label, description,
    # common_names, path (root→leaf).  Threaded through embedding
    # text + LLM prompt + SHAP feature surface; the canonical labels
    # never appear in user-facing classifications (the universal
    # vocab is a substrate, not a tagging layer — see
    # docs/src/architecture/dst-evidence-independence.md).
    ontology_priors: list[dict] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(FEATURE_NAMES)

    def to_embedding_text(self, mask: dict[str, bool] | None = None) -> str:
        """Build embedding text from enabled features only.

        Args:
            mask: feature_name -> enabled. None means all enabled.
                  This is the ablation hook for SAGE.
        """

        def _enabled(name: str) -> bool:
            if mask is None:
                return True
            return mask.get(name, True)

        parts: list[str] = []

        if _enabled("column_name"):
            if self.is_generic_name and self.value_description:
                parts.append(self.value_description)
            elif self.column_name_humanized:
                parts.append(self.column_name_humanized)

        if _enabled("column_type") and self.column_type:
            parts.append(self.column_type)

        if _enabled("sample_values") and self.sample_values_text:
            parts.append(self.sample_values_text)

        if _enabled("cardinality") and self.cardinality is not None:
            parts.append(f"cardinality={self.cardinality}")

        if _enabled("null_ratio") and self.null_ratio is not None and self.null_ratio > 0:
            parts.append(f"null_ratio={self.null_ratio:.2f}")

        if _enabled("value_entropy") and self.value_entropy is not None and self.value_entropy > 0:
            parts.append(f"entropy={self.value_entropy:.2f}")

        if _enabled("pattern_signals") and self.pattern_signals:
            parts.append("patterns: " + ", ".join(self.pattern_signals.keys()))

        # Canonical ontology metadata for fired patterns — embeds the
        # public-grounded label/description so cosine sees terms an
        # embedding model recognizes from training.  Distinct feature
        # so SAGE can ablate it independently of pattern_signals.
        if _enabled("ontology_priors") and self.ontology_priors:
            prior_strs: list[str] = []
            for prior in self.ontology_priors:
                label = prior.get("label", "")
                desc = prior.get("description", "")
                aliases = prior.get("common_names", "")
                path = prior.get("path", []) or []
                # Skip the "Information Content Entity" root from the
                # rendered path — it adds no semantic discriminator.
                path_render = " → ".join(p for p in path if p != "Information Content Entity")
                bits: list[str] = []
                if label:
                    bits.append(label)
                if desc:
                    bits.append(desc)
                if aliases:
                    bits.append(f"aliases: {aliases}")
                if path_render:
                    bits.append(f"ontology: {path_render}")
                if bits:
                    prior_strs.append("; ".join(bits))
            if prior_strs:
                parts.append("ontology priors: " + " | ".join(prior_strs))

        if _enabled("avg_value_length") and self.avg_value_length is not None:
            parts.append(f"avg_len={self.avg_value_length:.1f}")

        if _enabled("numeric_ratio") and self.numeric_ratio is not None and self.numeric_ratio > 0:
            parts.append(f"numeric={self.numeric_ratio:.2f}")

        if _enabled("sibling_context") and self.sibling_names:
            parts.append("siblings: " + ", ".join(self.sibling_names[:5]))

        if _enabled("source_table") and self.source_table:
            parts.append(f"table={self.source_table}")

        if _enabled("value_description") and self.value_description and not self.is_generic_name:
            parts.append(self.value_description)

        return " | ".join(parts) if parts else ""

    def feature_value(self, name: str) -> str:
        """Return the text contribution of a single feature."""
        mask = {n: (n == name) for n in FEATURE_NAMES}
        return self.to_embedding_text(mask)

    def to_embedding_segments(self) -> dict[str, str | float | None]:
        """Per-feature view used to build a structured CatBoost input.

        Returns a dict keyed by ``FEATURE_NAMES`` with each value typed
        according to the feature's natural shape:

        - **Text features** (column_name, column_type, sample_values,
          pattern_signals, sibling_context, value_description,
          source_table) → ``str`` (empty string when missing).  Each is
          encoded by the SentenceTransformer separately so its
          contribution is preserved as a named slice in the model's
          input vector — TreeSHAP attributes per slice → per source
          feature.
        - **Scalar features** (cardinality, null_ratio, value_entropy,
          avg_value_length, numeric_ratio) → ``float`` or ``None``.
          Passed through to CatBoost as native numerical inputs;
          CatBoost handles NaN natively and TreeSHAP attributes
          directly.

        This is the foundation for genuine per-feature TreeSHAP
        attribution: instead of compressing all 12 features into a
        single concatenated string → 384-dim vector (which entangles
        them so TreeSHAP can only report aggregate "embedding"
        contribution), each feature occupies its own input slot.
        """
        # Text features — render the same content the legacy single-
        # text path used, but per-feature so the encoder produces a
        # dedicated 384-dim slice for each.
        if self.is_generic_name and self.value_description:
            # Generic names have no signal; let the value_description
            # stand in (matches legacy to_embedding_text behaviour).
            name_text = self.value_description
        else:
            name_text = self.column_name_humanized or ""
        type_text = self.column_type or ""
        samples_text = self.sample_values_text or ""
        # pattern_signals serialized exactly the way to_embedding_text
        # rendered it for the LLM, so the embedding lands on the same
        # tokens the LLM saw.
        if self.pattern_signals:
            patterns_text = "patterns: " + ", ".join(self.pattern_signals.keys())
        else:
            patterns_text = ""
        if self.sibling_names:
            siblings_text = "siblings: " + ", ".join(self.sibling_names[:5])
        else:
            siblings_text = ""
        # value_description only when not already used as the name stand-in.
        if self.value_description and not self.is_generic_name:
            value_desc_text = self.value_description
        else:
            value_desc_text = ""
        source_table_text = self.source_table or ""

        return {
            "column_name": name_text,
            "column_type": type_text,
            "sample_values": samples_text,
            "cardinality": (
                float(self.cardinality) if self.cardinality is not None else None
            ),
            "null_ratio": (
                float(self.null_ratio) if self.null_ratio is not None else None
            ),
            "value_entropy": (
                float(self.value_entropy) if self.value_entropy is not None else None
            ),
            "pattern_signals": patterns_text,
            "avg_value_length": (
                float(self.avg_value_length)
                if self.avg_value_length is not None else None
            ),
            "numeric_ratio": (
                float(self.numeric_ratio) if self.numeric_ratio is not None else None
            ),
            "sibling_context": siblings_text,
            "source_table": source_table_text,
            "value_description": value_desc_text,
        }


# ── Extraction ───────────────────────────────────────────────────────


def extract_features(
    column_name: str,
    column_type: str | None = None,
    values: list[str] | None = None,
    siblings: list[str] | None = None,
    source_table: str | None = None,
    total_count: int | None = None,
    null_count: int = 0,
    max_values: int = 5,
    distinct_count: int | None = None,
) -> ColumnFeatures:
    """Extract ColumnFeatures from raw column metadata.

    Unlike signals (which requires ColumnSample), this accepts raw metadata
    dicts from agent-mediated hive sampling.

    Args:
        column_name: The column name.
        column_type: SQL type (e.g. "VARCHAR", "INTEGER").
        values: Sample values as strings.
        siblings: Other column names in the same table.
        source_table: Name of the source table.
        total_count: Total number of rows sampled.
        null_count: Number of NULL values in the sample.
        max_values: Max sample values to include in text.
        distinct_count: True COUNT(DISTINCT) from metadata query.
    """
    values = values or []
    siblings = siblings or []

    name_humanized = column_name.replace("_", " ")

    col_type: str | None = None
    if column_type and column_type.upper() not in ("STRING", "VARCHAR"):
        col_type = column_type.lower()

    sample_text: str | None = None
    if values:
        sample_text = ", ".join(v[:80] for v in values[:max_values])

    # Prefer true COUNT(DISTINCT) from metadata; fall back to sample cardinality
    cardinality = distinct_count if distinct_count is not None else (
        len(set(values)) if values else None
    )

    null_ratio: float | None = None
    if total_count and total_count > 0:
        null_ratio = round(null_count / total_count, 4)

    entropy = _shannon_entropy(values) if values else None
    patterns = detect_patterns(values)

    # Resolve every fired pattern to its canonical ICE.* metadata —
    # this becomes a publicly-grounded semantic prior threaded through
    # the embedding text, the LLM prompt, and SAGE/SHAP attribution.
    # See mass_functions.lookup_pattern_ontology.
    from atelier.classify.mass_functions import lookup_pattern_ontology
    ontology_priors: list[dict] = []
    for pattern_name, fraction in (patterns or {}).items():
        prior = lookup_pattern_ontology(pattern_name)
        if prior is None:
            continue
        prior = dict(prior)
        prior["match_fraction"] = float(fraction)
        ontology_priors.append(prior)

    avg_len: float | None = None
    if values:
        avg_len = round(sum(len(v) for v in values) / len(values), 2)

    num_ratio = _numeric_ratio(values) if values else None

    sibling_names = [s.replace("_", " ") for s in siblings if s != column_name]

    generic = _is_generic_name(column_name)
    val_desc = _generate_value_description(values, col_type, patterns)

    return ColumnFeatures(
        column_name_humanized=name_humanized,
        column_type=col_type,
        sample_values_text=sample_text,
        cardinality=cardinality,
        null_ratio=null_ratio,
        value_entropy=entropy,
        pattern_signals=patterns,
        avg_value_length=avg_len,
        numeric_ratio=num_ratio,
        sibling_names=sibling_names,
        source_table=source_table,
        value_description=val_desc,
        is_generic_name=generic,
        ontology_priors=ontology_priors,
    )
