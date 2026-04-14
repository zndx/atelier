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


# ── Pattern detectors ────────────────────────────────────────────────

_PATTERNS: dict[str, re.Pattern] = {
    "email_pattern": re.compile(
        r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    ),
    "phone_pattern": re.compile(
        r"^[\+]?[\d\s\-\(\)\.]{7,20}$"
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
        r"^[0-9a-fA-F]{32,128}$"
    ),
    "semver_pattern": re.compile(
        r"^\d+\.\d+\.\d+([.\-+].+)?$"
    ),
    "iso_currency_pattern": re.compile(
        r"^[A-Z]{3}$"
    ),
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
    col_type: str | None,
    patterns: list[str],
) -> str:
    """Generate a natural-language description based on value shape."""
    if not values:
        return ""

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


def detect_patterns(values: list[str]) -> list[str]:
    """Detect which value patterns are present in a sample."""
    if not values:
        return []
    hits: list[str] = []
    for name, pat in _PATTERNS.items():
        match_count = sum(1 for v in values if pat.match(v.strip()))
        if match_count >= max(1, len(values) // 3):
            hits.append(name)
    return sorted(hits)


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


# ── ColumnFeatures ───────────────────────────────────────────────────


@dataclass(frozen=True)
class ColumnFeatures:
    """Discrete, ablatable features extracted from a column sample.

    Each of the 12 features can be independently masked for SAGE analysis.
    """

    column_name_humanized: str
    column_type: str | None
    sample_values_text: str | None
    cardinality: int | None
    null_ratio: float | None
    value_entropy: float | None
    pattern_signals: list[str] = field(default_factory=list)
    avg_value_length: float | None = None
    numeric_ratio: float | None = None
    sibling_names: list[str] = field(default_factory=list)
    source_table: str | None = None
    value_description: str = ""
    is_generic_name: bool = False

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
            parts.append("patterns: " + ", ".join(self.pattern_signals))

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
    )
