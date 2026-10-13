"""Genus-first Choice walk for a decisions sweep.

One entity is a sequence of Engine/Decide calls. Each call asks one
Choice, because the next option set depends on the distribution just
returned. The opening Choice is the structural genera of the loaded
taxonomy (a node that has both a parent and children). A taxonomy
with no such node opens on the nodes that branch, then on the codes
themselves. A Choice larger than 255 fails closed.

The probability on the deciding question is the confidence the sweep
hands to ``llm_to_mass``. It is uncalibrated against the 0.15 LLM
discount. NHSVM membership is not a gate on this walk.

Column type annotation labels each column with a neighborhood of
those genera. The opening Choice stays the genera themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

MAX_CHOICE_OPTIONS = 255
MAX_CALLS = 64
DEFAULT_DESCEND_FLOOR = 0.05

Ask = Callable[[Mapping[str, str | None], str], Mapping[str, float]]


class WalkError(RuntimeError):
    """The taxonomy cannot be asked as a legal Choice."""


@dataclass(frozen=True)
class WalkResult:
    category_code: str | None
    confidence: float
    alternatives: list[dict[str, Any]]
    calls: int
    truncated: bool


def classification_from_distribution(
    dist: Mapping[str, float],
) -> tuple[str | None, float, list[dict[str, Any]]]:
    """Top-1 plus the next three options that carry mass."""
    ranked = sorted(
        ((str(code), float(prob)) for code, prob in dist.items() if float(prob) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return None, 0.0, []
    code, confidence = ranked[0]
    alternatives = [
        {"code": other, "confidence": prob} for other, prob in ranked[1:4]
    ]
    return code, confidence, alternatives


def structural_genera(category_set: Any) -> list[str]:
    """Nodes that have a parent and at least one child, in code order."""
    children: Mapping[str, list[str]] = category_set.children
    parent: Mapping[str, str | None] = category_set.parent
    genera = [
        code
        for code, kids in children.items()
        if kids and parent.get(code)
    ]
    genera.sort()
    return genera


# Anchorless catalog bucket. Its children do not share a BFO/CCO kind, so
# gathering them into this parent makes one class out of unrelated columns.
_RESIDUAL_BUCKET = "SDG.GENERIC"


def genus_term(category_set: Any, code: str) -> str | None:
    """Genus-level SKOS term for one code.

    The nearest structural genus on the path, including ``code`` when
    it has a parent and children. A branch whose nodes have children
    but no parent — Process, Role — is that column's genus.

    ``SDG.GENERIC`` is not such a genus. A child of that bucket stays
    itself. The bucket code is not a column label.
    """
    children: Mapping[str, list[str]] = category_set.children
    parent: Mapping[str, str | None] = category_set.parent
    chain = [code]
    current = parent.get(code)
    while current:
        chain.append(current)
        current = parent.get(current)
    for node in chain:
        if node == _RESIDUAL_BUCKET:
            continue
        if children.get(node) and parent.get(node):
            return node
    for node in chain:
        if node == _RESIDUAL_BUCKET:
            continue
        if children.get(node):
            return node
    if code != _RESIDUAL_BUCKET and _RESIDUAL_BUCKET in chain:
        return code
    return None


def genus_neighborhood(category_set: Any, order: int = 1) -> list[str]:
    """Codes within graph distance ``order`` of a structural genus.

    Edges run to the parent and to each child. The result is closed
    under ancestors, so a parent pointer inside the frame stays inside
    it. Order 1 is each genus, its parent, and its children. Order 2
    adds the siblings of those genera. A taxonomy with no structural
    genera yields an empty frame.
    """
    if order < 1:
        raise ValueError(f"genus neighborhood order starts at 1, got {order}")
    children: Mapping[str, list[str]] = category_set.children
    parent: Mapping[str, str | None] = category_set.parent
    known = set(parent)
    seen = {code for code in structural_genera(category_set) if code in known}
    frontier = list(seen)
    for _step in range(order):
        nxt: list[str] = []
        for code in frontier:
            up = parent.get(code)
            if up and up in known and up not in seen:
                seen.add(up)
                nxt.append(up)
            for kid in children.get(code) or ():
                if kid in known and kid not in seen:
                    seen.add(kid)
                    nxt.append(kid)
        frontier = nxt
    for code in list(seen):
        current = parent.get(code)
        while current and current in known:
            if current in seen:
                break
            seen.add(current)
            current = parent.get(current)
    frame = sorted(seen)
    return frame


def neighborhood_term(
    category_set: Any,
    code: str,
    *,
    order: int = 1,
) -> str | None:
    """Label for one code on the genus neighborhood.

    A code inside the frame keeps itself, so the parent, the genus,
    and the child are three labels. A code outside the frame takes
    the nearest ancestor that is in the frame. A taxonomy with no
    structural genera uses ``genus_term``.
    """
    return _label_on_frame(
        category_set, str(code), set(genus_neighborhood(category_set, order)),
    )


def _label_on_frame(
    category_set: Any,
    code: str,
    frame: set[str],
) -> str | None:
    if not frame:
        return genus_term(category_set, code)
    if code in frame:
        return code
    parent: Mapping[str, str | None] = category_set.parent
    current = parent.get(code)
    while current:
        if current in frame:
            return current
        current = parent.get(current)
    return genus_term(category_set, code)


def associate_genera(
    assignments: Mapping[tuple[str, str], str],
    category_set: Any,
    *,
    order: int = 1,
) -> dict[tuple[str, str], str]:
    """Map each assigned column onto the genus neighborhood.

    The default frame is the first-order neighborhood. A column whose
    code is in that frame keeps the code. A column outside it takes
    the nearest in-frame ancestor. A taxonomy with no structural
    genera keeps ``genus_term``. A column with no term is omitted.
    """
    frame = set(genus_neighborhood(category_set, order))
    associated: dict[tuple[str, str], str] = {}
    for key, code in assignments.items():
        term = _label_on_frame(category_set, str(code), frame)
        if term:
            associated[key] = term
    return associated


def opening_codes(category_set: Any) -> list[str]:
    """The first Choice. Structural genera when the taxonomy has them."""
    genera = structural_genera(category_set)
    if genera:
        return genera
    children: Mapping[str, list[str]] = category_set.children
    branched = [code for code, kids in children.items() if kids]
    branched.sort()
    if branched:
        return branched
    codes = [cat.code for cat in category_set.categories if getattr(cat, "code", "")]
    codes.sort()
    return codes


def _require_choice(codes: list[str], label: str) -> None:
    if not codes:
        raise WalkError(f"decisions walk has an empty {label} choice")
    if len(codes) > MAX_CHOICE_OPTIONS:
        raise WalkError(
            f"decisions walk {label} has {len(codes)} options; "
            f"the limit is {MAX_CHOICE_OPTIONS}"
        )


def _criteria(codes: list[str], by_code: Mapping[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for code in codes:
        cat = by_code.get(code)
        text = ""
        if cat is not None:
            text = (getattr(cat, "description", None) or getattr(cat, "label", None) or "").strip()
        out[code] = text or None
    return out


def _dedupe(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def walk_entity(
    category_set: Any,
    ask: Ask,
    *,
    floor: float = DEFAULT_DESCEND_FLOOR,
    max_calls: int = MAX_CALLS,
) -> WalkResult:
    """Ask the opening genera, then descend every genus at or above ``floor``.

    The primary branch is the genus with the highest opening probability.
    Its deciding question supplies ``category_code``, ``confidence``, and
    the alternatives. Other genera at or above the floor are still asked,
    and a winner there replaces the primary only when its opening genus
    carried more mass. The walk stops at ``max_calls`` and keeps the best
    code already chosen.
    """
    by_code = category_set.all_by_code
    children: Mapping[str, list[str]] = category_set.children
    opening = opening_codes(category_set)
    _require_choice(opening, "opening")

    calls = 0
    truncated = False

    def _ask(codes: list[str], instructions: str) -> Mapping[str, float] | None:
        nonlocal calls, truncated
        if calls >= max_calls:
            truncated = True
            return None
        _require_choice(codes, "choice")
        dist = ask(_criteria(codes, by_code), instructions)
        calls += 1
        return dist

    root = _ask(opening, "Choose the genus for this entity.")
    if not root:
        raise WalkError("decisions walk received an empty opening distribution")
    code, confidence, alternatives = classification_from_distribution(root)
    primary_genus = code
    primary_mass = confidence

    pending = sorted(
        (
            (float(prob), genus)
            for genus, prob in root.items()
            if float(prob) >= floor and children.get(str(genus))
        ),
        key=lambda item: (-item[0], item[1]),
    )
    for genus_mass, genus in pending:
        if calls >= max_calls:
            truncated = True
            break
        current = str(genus)
        branch_code = code
        branch_confidence = confidence
        branch_alts = alternatives
        while calls < max_calls and children.get(current):
            options = _dedupe([current, *children[current]])
            dist = _ask(options, "Choose this code or one of its children.")
            if not dist:
                truncated = True
                break
            winner, win_p, win_alts = classification_from_distribution(dist)
            if winner is None:
                break
            branch_code, branch_confidence, branch_alts = winner, win_p, win_alts
            if winner == current or not children.get(winner):
                break
            current = winner
        if genus_mass >= primary_mass:
            primary_mass = genus_mass
            primary_genus = str(genus)
            code, confidence, alternatives = branch_code, branch_confidence, branch_alts
        if calls >= max_calls:
            truncated = True
            break

    if truncated and code is None:
        code = primary_genus
    return WalkResult(
        category_code=code,
        confidence=float(confidence or 0.0),
        alternatives=list(alternatives),
        calls=calls,
        truncated=truncated,
    )
