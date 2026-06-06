"""Realistic mock LLM backend for testing bootstrap convergence.

Makes plausible classification mistakes (~80% accuracy) and corrects
~70% of them when revisited with ML context. Deterministic via seeded RNG.
"""

from __future__ import annotations

import random
from typing import Any

from atelier.classify.llm_backend import ColumnClassification, LLMResponse

# Confusable pairs: categories that a real LLM might plausibly mix up
CONFUSABLE_PAIRS = [
    ("ICE.SENSITIVE.PID.CONTACT.EMAIL", "ICE.SENSITIVE.TECHNICAL.URL"),       # Email ↔ URL
    ("ICE.SENSITIVE.PID.CONTACT.PHONE", "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN"),   # Phone ↔ Bank Account
    ("ICE.SENSITIVE.PID.IDENTITY.GOVID.SSN", "ICE.SENSITIVE.PID.IDENTITY.DOB"),   # SSN ↔ DOB
    ("ICE.SENSITIVE.PID.FINANCIAL.PAYMENT.CARD.PAN", "ICE.SENSITIVE.PID.FINANCIAL.ACCOUNT.BAN"),   # Credit Card ↔ Bank Account
    ("ICE.METADATA.TIMESTAMP", "ICE.SENSITIVE.PID.IDENTITY.DOB"),             # Timestamp ↔ DOB
    ("ICE.SENSITIVE.PID.IDENTITY.NAME.FULLNAME", "ICE.SENSITIVE.PID.CONTACT.ADDRESS"),  # Full Name ↔ Address
    ("ICE.METADATA.RECID", "ICE.SENSITIVE.TECHNICAL.DEVID"),                  # Record ID ↔ UUID
]


def _build_confusion_map(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Build bidirectional confusion lookup (multi-valued).

    A code can appear in multiple pairs (e.g., BAN is confusable with
    both PHONE and PAN).  The map stores all targets so the caller can
    pick one at random.
    """
    m: dict[str, list[str]] = {}
    for a, b in pairs:
        m.setdefault(a, []).append(b)
        m.setdefault(b, []).append(a)
    return m


_CONFUSION = _build_confusion_map(CONFUSABLE_PAIRS)


class RealisticMockLLMBackend:
    """Mock LLM that makes plausible classification mistakes.

    - Gets ~base_accuracy right with varying confidence (0.7–0.95)
    - When wrong, confuses siblings from CONFUSABLE_PAIRS
    - On revisit with ML context, corrects ~revisit_correction_rate of mistakes
    - Deterministic via seeded RNG
    """

    def __init__(
        self,
        reference_labels: dict[str, str],
        *,
        base_accuracy: float = 0.80,
        revisit_correction_rate: float = 0.70,
        seed: int = 42,
    ):
        self._ref = reference_labels
        self._base_accuracy = base_accuracy
        self._revisit_correction_rate = revisit_correction_rate
        self._rng = random.Random(seed)
        # Track which columns we've already gotten wrong
        self._mistakes: set[str] = set()

    def classify_batch(
        self,
        samples: list,
        system_prompt: str,
        revisit_context: dict[str, dict] | None = None,
        table_name: str | None = None,
    ) -> LLMResponse:
        classifications = []
        for sample in samples:
            true_code = self._ref.get(sample.name)
            if not true_code:
                classifications.append(ColumnClassification(
                    column_name=sample.name,
                    category_code=None,
                    confidence=0.0,
                    evidence="no reference label available",
                    alternatives=[],
                ))
                continue

            is_revisit = (
                revisit_context is not None
                and sample.name in revisit_context
            )

            if is_revisit and sample.name in self._mistakes:
                # On revisit, correct mistakes with some probability
                if self._rng.random() < self._revisit_correction_rate:
                    code = true_code
                    confidence = round(self._rng.uniform(0.80, 0.95), 2)
                    self._mistakes.discard(sample.name)
                else:
                    # Still wrong
                    confusables = _CONFUSION.get(true_code)
                    code = self._rng.choice(confusables) if confusables else true_code
                    confidence = round(self._rng.uniform(0.60, 0.75), 2)
            elif self._rng.random() < self._base_accuracy:
                # Correct classification
                code = true_code
                confidence = round(self._rng.uniform(0.70, 0.95), 2)
            else:
                # Wrong — confuse with sibling
                confusables = _CONFUSION.get(true_code)
                code = self._rng.choice(confusables) if confusables else true_code
                confidence = round(self._rng.uniform(0.55, 0.80), 2)
                if code != true_code:
                    self._mistakes.add(sample.name)

            alternatives = []
            if code != true_code:
                alternatives.append({
                    "code": true_code,
                    "confidence": round(confidence * 0.5, 2),
                })

            classifications.append(ColumnClassification(
                column_name=sample.name,
                category_code=code,
                confidence=confidence,
                evidence=f"realistic mock ({'revisit' if is_revisit else 'initial'})",
                alternatives=alternatives,
            ))

        return LLMResponse(
            classifications=classifications,
            input_tokens=len(samples) * 150,
            output_tokens=len(samples) * 80,
            model="realistic-mock",
        )

    def health_check(self) -> bool:
        return True
