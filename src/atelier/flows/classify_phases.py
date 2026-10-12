"""Phase helpers for ClassificationFlow — coverage is the pass gate.

Success: every target relational entity has a predicted_code.
PRECONDITIONING expensive work runs only when the SKOS/vocab signature
(or encoder identity) is stale — the probe is cheap and claims no GPU.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from atelier.flows.lattice import COMPLETE_CAPABILITIES

# Graph the FlowSpec must realize (step → next). ``sweep`` is a foreach
# onto ``sweep_shard`` (one shard per d3-flash replica). ``sweep_join``
# is the join. publish is terminal.
STEP_GRAPH: tuple[tuple[str, str | None], ...] = (
    ("start", "probe"),
    ("probe", "precondition"),
    ("precondition", "load"),
    ("load", "sweep"),
    ("sweep", "sweep_shard"),
    ("sweep_shard", "sweep_join"),
    ("sweep_join", "revisit"),
    ("revisit", "fuse"),
    ("fuse", "evaluate"),
    ("evaluate", "end"),
    ("end", None),
)

SWEEP_CAPABILITY = "decisions"
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


def merge_shard_labels(groups: Iterable[Sequence[Mapping[str, Any]]]) -> list[dict]:
    """Join foreach shard labels. Overlap on a qualified name fails closed."""
    merged: dict[str, dict] = {}
    for group in groups:
        for row in group:
            key = str(row.get("qualified_name") or "")
            if not key:
                raise RuntimeError("decisions shard label is missing qualified_name")
            if key in merged:
                raise RuntimeError(f"decisions shards overlapped on {key}")
            merged[key] = dict(row)
    return [merged[key] for key in sorted(merged)]


def llm_capability_for_sweep(*, revisit: bool) -> str:
    """Sweep is ``decisions``. Revisit stays on Complete ``thinking``."""
    if revisit:
        cap = REVISIT_CAPABILITY
        assert cap in COMPLETE_CAPABILITIES
        return cap
    return SWEEP_CAPABILITY
