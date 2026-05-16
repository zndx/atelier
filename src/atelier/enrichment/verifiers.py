# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Deterministic verifiers for LLM-generated annotation enrichment fields.

These run after each LLM generation pass and decide whether the output
is well-formed and self-consistent.  Failures are surfaced to the
enrichment loop, which re-prompts with a refinement message or skips
to the next annotation depending on the policy.

No LLM in this module — pure functions that take an enrichment payload
plus context and return a structured result.  This is the procedural
component of the LLM-mediated-reference-artifact pattern: the verifier
suite is what makes the generated output falsifiable.

The verifier surface mirrors the payload fields specified in
``docs/src/architecture/late-interaction-cosine.md`` § Qdrant payload
schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


# ── Result types ──────────────────────────────────────────────────


@dataclass
class CheckResult:
    """One verifier's verdict on one enrichment payload."""

    name: str
    passed: bool
    detail: str = ""
    offenders: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "offenders": list(self.offenders),
        }


@dataclass
class VerifierReport:
    """Aggregate result of running the full verifier suite on one payload."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def checks_passed(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def checks_total(self) -> int:
        return len(self.checks)

    def failed(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> dict:
        """Shape the payload-side ``verifier_results`` JSON blob expects."""
        out: dict[str, Any] = {
            "checks_passed": self.checks_passed,
            "checks_total": self.checks_total,
        }
        # Per-check pass flag with the documented payload field name
        for c in self.checks:
            out[c.name] = c.passed
        # Diagnostic detail kept under a separate key so the payload
        # surface stays predictable for queries
        out["details"] = [c.to_dict() for c in self.checks if not c.passed]
        return out


# ── Individual verifier checks ────────────────────────────────────


def check_patterns_compile(enrichment: dict) -> CheckResult:
    """Every ``value_patterns[*].expr`` of kind ``regex`` must compile.

    Non-regex pattern kinds (``format``, ``length``, …) are passed
    through — they're advisory text, not executable patterns.
    """
    patterns = enrichment.get("value_patterns", []) or []
    offenders: list[Any] = []
    for p in patterns:
        if not isinstance(p, dict):
            offenders.append({"value": p, "reason": "not a dict"})
            continue
        kind = p.get("kind", "regex")
        expr = p.get("expr")
        if kind != "regex" or expr is None:
            continue
        try:
            re.compile(expr)
        except re.error as exc:
            offenders.append({"expr": expr, "reason": f"regex: {exc}"})
    return CheckResult(
        name="patterns_compile",
        passed=not offenders,
        detail=(
            "all regex patterns compile"
            if not offenders
            else f"{len(offenders)} pattern(s) failed to compile"
        ),
        offenders=offenders,
    )


def check_prototype_values_match_patterns(enrichment: dict) -> CheckResult:
    """Each ``prototype_value`` must match at least one declared regex pattern.

    Skipped (counted as passing) when no regex patterns are declared —
    the generator is allowed to produce free-form prototypes without
    a pattern claim, and the absence of patterns is a separate signal
    (caught by ``check_patterns_non_empty`` if enforced).
    """
    prototypes = enrichment.get("prototype_values", []) or []
    patterns = enrichment.get("value_patterns", []) or []
    regexes: list[re.Pattern[str]] = []
    for p in patterns:
        if not isinstance(p, dict):
            continue
        if p.get("kind", "regex") != "regex":
            continue
        expr = p.get("expr")
        if expr is None:
            continue
        try:
            regexes.append(re.compile(expr))
        except re.error:
            # The compile-check verifier catches this separately.
            continue

    if not regexes:
        return CheckResult(
            name="prototype_values_match_patterns",
            passed=True,
            detail="no regex patterns declared; vacuously passes",
        )

    offenders: list[Any] = []
    for v in prototypes:
        if v is None:
            offenders.append({"value": None, "reason": "null prototype"})
            continue
        s = str(v)
        if not any(r.search(s) for r in regexes):
            offenders.append({"value": s, "reason": "no pattern matched"})
    return CheckResult(
        name="prototype_values_match_patterns",
        passed=not offenders,
        detail=(
            f"{len(prototypes) - len(offenders)}/{len(prototypes)} prototypes "
            f"match at least one pattern"
        ),
        offenders=offenders,
    )


def check_anti_example_targets_exist(
    enrichment: dict,
    valid_tag_codes: set[str],
) -> CheckResult:
    """Each anti-example's ``confusable_tag`` must exist in the taxonomy.

    ``valid_tag_codes`` is the set of codes (or mnemonics, depending on
    the taxonomy convention) the caller considers legitimate targets.
    Empty set means we can't check — degrade gracefully and pass.
    """
    anti = enrichment.get("anti_examples", []) or []
    if not valid_tag_codes:
        return CheckResult(
            name="anti_example_targets_exist",
            passed=True,
            detail="no taxonomy context provided; check skipped",
        )

    offenders: list[Any] = []
    for a in anti:
        if not isinstance(a, dict):
            offenders.append({"value": a, "reason": "not a dict"})
            continue
        target = a.get("confusable_tag")
        if target is None:
            offenders.append({"value": a, "reason": "no confusable_tag"})
            continue
        if target not in valid_tag_codes:
            offenders.append(
                {"value": a, "reason": f"unknown tag: {target!r}"}
            )
    return CheckResult(
        name="anti_example_targets_exist",
        passed=not offenders,
        detail=(
            f"{len(anti) - len(offenders)}/{len(anti)} anti-examples "
            f"reference known tags"
        ),
        offenders=offenders,
    )


def check_parent_path_consistent(
    enrichment: dict,
    expected_parent_path: list[str] | None,
) -> CheckResult:
    """The generated ``parent_path`` must match the taxonomy structure.

    ``expected_parent_path`` is what the taxonomy hierarchy says the
    parent chain should be (a deterministic lookup from the source
    taxonomy table).  None means we don't have a hierarchy to check
    against; degrade gracefully and pass.
    """
    if expected_parent_path is None:
        return CheckResult(
            name="parent_path_consistent",
            passed=True,
            detail="no expected parent path provided; check skipped",
        )

    got = enrichment.get("parent_path") or []
    if not isinstance(got, list):
        return CheckResult(
            name="parent_path_consistent",
            passed=False,
            detail=f"parent_path is not a list (got {type(got).__name__})",
            offenders=[got],
        )
    if got != expected_parent_path:
        return CheckResult(
            name="parent_path_consistent",
            passed=False,
            detail="parent_path does not match taxonomy hierarchy",
            offenders=[{"expected": expected_parent_path, "got": got}],
        )
    return CheckResult(
        name="parent_path_consistent",
        passed=True,
        detail="parent_path matches taxonomy hierarchy",
    )


def check_name_hints_non_empty(enrichment: dict) -> CheckResult:
    """``name_hints`` must contain at least one non-empty string."""
    hints = enrichment.get("name_hints", []) or []
    valid = [h for h in hints if isinstance(h, str) and h.strip()]
    if not valid:
        return CheckResult(
            name="name_hints_non_empty",
            passed=False,
            detail="no usable name_hints (all empty or non-string)",
            offenders=list(hints),
        )
    return CheckResult(
        name="name_hints_non_empty",
        passed=True,
        detail=f"{len(valid)} usable name hint(s)",
    )


def check_no_contradiction_with_anti_examples(enrichment: dict) -> CheckResult:
    """Prototype values must not appear (by exact match) in anti_examples.

    A column with value V cannot simultaneously be a prototype FOR tag X
    and a *negative* example for tag X — that's a contradiction the
    enrichment loop should reject and retry on.
    """
    prototypes = {str(v) for v in (enrichment.get("prototype_values", []) or [])
                  if v is not None}
    anti = enrichment.get("anti_examples", []) or []
    offenders: list[Any] = []
    for a in anti:
        if not isinstance(a, dict):
            continue
        v = a.get("value")
        if v is None:
            continue
        if str(v) in prototypes:
            offenders.append(
                {"value": v, "reason": "appears in both prototype_values and anti_examples"}
            )
    return CheckResult(
        name="no_contradiction_with_anti_examples",
        passed=not offenders,
        detail=(
            "no overlap between prototypes and anti-examples"
            if not offenders
            else f"{len(offenders)} contradiction(s)"
        ),
        offenders=offenders,
    )


# ── Suite runner ──────────────────────────────────────────────────


def run_verifier_suite(
    enrichment: dict,
    *,
    valid_tag_codes: set[str] | None = None,
    expected_parent_path: list[str] | None = None,
) -> VerifierReport:
    """Run the full verifier suite on one enrichment payload.

    Returns a :class:`VerifierReport` that the caller serializes into
    the Qdrant payload's ``verifier_results`` field.
    """
    report = VerifierReport()
    report.checks.append(check_patterns_compile(enrichment))
    report.checks.append(check_prototype_values_match_patterns(enrichment))
    report.checks.append(
        check_anti_example_targets_exist(enrichment, valid_tag_codes or set())
    )
    report.checks.append(
        check_parent_path_consistent(enrichment, expected_parent_path)
    )
    report.checks.append(check_name_hints_non_empty(enrichment))
    report.checks.append(check_no_contradiction_with_anti_examples(enrichment))
    return report
