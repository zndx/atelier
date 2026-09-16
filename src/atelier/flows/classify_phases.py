"""Phase helpers for ClassificationFlow — coverage is the pass gate.

Success: every target relational entity has a predicted_code.
PRECONDITIONING expensive work runs only when the SKOS/vocab signature
(or encoder identity) is stale — the probe is cheap and claims no GPU.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from atelier.flows.lattice import COMPLETE_CAPABILITIES

# Graph the FlowSpec must realize (step → next). publish is terminal.
STEP_GRAPH: tuple[tuple[str, str | None], ...] = (
    ("start", "probe"),
    ("probe", "precondition"),
    ("precondition", "load"),
    ("load", "sweep"),
    ("sweep", "fuse"),
    ("fuse", "evaluate"),
    ("evaluate", "publish"),
    ("publish", None),
)

SWEEP_CAPABILITY = "instruct"
REVISIT_CAPABILITY = "thinking"


def entity_key(row: Mapping[str, Any]) -> str:
    for k in ("qualified_name", "column_name", "entity_id", "name"):
        v = row.get(k)
        if v:
            return str(v)
    return ""


def predicted_code(row: Mapping[str, Any]) -> str:
    v = row.get("predicted_code")
    return "" if v is None else str(v).strip()


def unclassified_targets(
    classifications: Sequence[Mapping[str, Any]],
    targets: Iterable[str],
) -> list[str]:
    """Target keys with no non-empty predicted_code."""
    by_key: dict[str, str] = {}
    for row in classifications:
        key = entity_key(row)
        if key:
            by_key[key] = predicted_code(row)
    missing: list[str] = []
    for t in targets:
        code = by_key.get(t, "")
        if not code:
            missing.append(t)
    return missing


def coverage_complete(
    classifications: Sequence[Mapping[str, Any]],
    targets: Iterable[str],
) -> bool:
    return not unclassified_targets(classifications, targets)


def assert_coverage(
    classifications: Sequence[Mapping[str, Any]],
    targets: Sequence[str],
) -> None:
    missing = unclassified_targets(classifications, targets)
    if targets and missing:
        raise RuntimeError(
            f"classification coverage incomplete: {len(missing)} target(s) "
            f"unclassified (first: {missing[0]!r}). Success is every "
            f"target relational entity classified."
        )


def probe_requires_precondition(status: Mapping[str, Any] | Any) -> bool:
    """Expensive precondition only when artifacts are not signature-final."""
    if hasattr(status, "final"):
        return not bool(status.final)
    return not bool(status.get("final")) if isinstance(status, Mapping) else True


def llm_capability_for_sweep(*, revisit: bool) -> str:
    cap = REVISIT_CAPABILITY if revisit else SWEEP_CAPABILITY
    assert cap in COMPLETE_CAPABILITIES
    return cap
