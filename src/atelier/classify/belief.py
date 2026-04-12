"""Dempster-Shafer Theory (DST) belief functions for uncertainty management.

Provides mass functions, Dempster's rule of combination, and a restricted
frame of discernment built from the hierarchical taxonomy.

Ported from signals/src/sigint/belief.py — adapted for atelier's taxonomy
structure (hive annotations with dot-notation hierarchical codes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property


@dataclass(frozen=True)
class FocalElement:
    """A subset of the frame of discernment.

    Wraps a frozenset of category codes with an optional label for display.
    """

    codes: frozenset[str]
    label: str = ""

    def __hash__(self) -> int:
        return hash(self.codes)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FocalElement):
            return NotImplemented
        return self.codes == other.codes

    def __repr__(self) -> str:
        if self.label:
            return f"FocalElement({self.label})"
        if len(self.codes) <= 3:
            return f"FocalElement({{{', '.join(sorted(self.codes))}}})"
        return f"FocalElement(|{len(self.codes)}|)"


@dataclass
class BeliefAssignment:
    """A mass function m: 2^Θ → [0,1] with Σ m(A) = 1.

    Only focal elements (m(A) > 0) are stored.
    """

    masses: dict[FocalElement, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        """Check that masses sum to ~1 and are non-negative."""
        total = sum(self.masses.values())
        return all(v >= -1e-12 for v in self.masses.values()) and abs(total - 1.0) < 1e-9

    def normalize(self) -> BeliefAssignment:
        """Return a normalized copy (masses sum to exactly 1)."""
        total = sum(self.masses.values())
        if total <= 0:
            return self
        return BeliefAssignment(
            masses={k: v / total for k, v in self.masses.items() if v > 1e-15}
        )

    def belief(self, target: FocalElement) -> float:
        """Bel(A) = Σ m(B) for all B ⊆ A."""
        return sum(
            m for fe, m in self.masses.items()
            if fe.codes.issubset(target.codes)
        )

    def plausibility(self, target: FocalElement) -> float:
        """Pl(A) = Σ m(B) for all B ∩ A ≠ ∅."""
        return sum(
            m for fe, m in self.masses.items()
            if fe.codes & target.codes
        )

    def belief_interval(self, target: FocalElement) -> tuple[float, float]:
        """Return (Bel(A), Pl(A))."""
        return (self.belief(target), self.plausibility(target))

    def uncertainty(self, target: FocalElement) -> float:
        """Pl(A) - Bel(A): the gap between belief and plausibility."""
        bel, pl = self.belief_interval(target)
        return pl - bel

    def pignistic_probability(self, singleton: FocalElement) -> float:
        """BetP for a singleton — the decision-theoretic transform.

        BetP({x}) = Σ_A m(A) · |{x}∩A| / |A|  for A ≠ ∅.
        Since singleton has |{x}|=1, this simplifies to:
        BetP({x}) = Σ_A (m(A) / |A|) for all A containing x.
        """
        if len(singleton.codes) != 1:
            raise ValueError("pignistic_probability requires a singleton focal element")
        code = next(iter(singleton.codes))
        total = 0.0
        for fe, m in self.masses.items():
            if code in fe.codes and len(fe.codes) > 0:
                total += m / len(fe.codes)
        return total


def dempster_combine(
    m1: BeliefAssignment, m2: BeliefAssignment,
) -> tuple[BeliefAssignment, float]:
    """Dempster's rule of combination (conjunctive, normalized).

    Returns ``(combined_assignment, K)`` where **K** is the conflict
    mass (sum of products assigned to the empty set before normalization).

    Raises ValueError on total conflict (K=1).
    """
    combined: dict[FocalElement, float] = {}
    conflict = 0.0

    for fe1, mass1 in m1.masses.items():
        for fe2, mass2 in m2.masses.items():
            intersection = fe1.codes & fe2.codes
            product = mass1 * mass2
            if not intersection:
                conflict += product
            else:
                fe = FocalElement(frozenset(intersection))
                combined[fe] = combined.get(fe, 0.0) + product

    if conflict >= 1.0 - 1e-12:
        raise ValueError(
            f"Total conflict in Dempster combination (K={conflict:.6f})"
        )

    normalization = 1.0 - conflict
    result = {
        fe: m / normalization
        for fe, m in combined.items()
        if m > 1e-15
    }
    return BeliefAssignment(masses=result), conflict


def combine_multiple(
    assignments: list[BeliefAssignment],
) -> tuple[BeliefAssignment, float]:
    """Left-to-right Dempster combination of multiple sources.

    Returns ``(combined_assignment, cumulative_K)`` where cumulative K
    is computed as ``K = 1 - ∏(1 - Kᵢ)`` (Smarandache & Dezert, 2005).
    """
    if not assignments:
        raise ValueError("Cannot combine empty list of assignments")
    result = assignments[0]
    product_of_complements = 1.0
    for other in assignments[1:]:
        result, k_i = dempster_combine(result, other)
        product_of_complements *= (1.0 - k_i)
    cumulative_k = 1.0 - product_of_complements
    return result, cumulative_k


class FrameOfDiscernment:
    """Restricted frame built from a HierarchicalCategorySet.

    Instead of 2^N focal elements, only tracks:
    - Singletons (1 per leaf category)
    - Internal nodes (descendant leaf sets for each parent)
    - Confusable pairs (manually specified)
    - Theta (full frame)
    """

    def __init__(
        self,
        category_set,  # HierarchicalCategorySet
        confusable_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        self._category_set = category_set
        self._confusable_pairs = confusable_pairs or []
        self._build_focal_elements()

    def _build_focal_elements(self) -> None:
        cs = self._category_set

        # Theta: full frame of all leaf codes
        self.theta = FocalElement(cs.leaf_codes, label="Θ")

        # Singletons
        self._singletons: dict[str, FocalElement] = {}
        for code in cs.leaf_codes:
            cat = cs.by_code.get(code) or cs.all_by_code.get(code)
            label = cat.label if cat else code
            self._singletons[code] = FocalElement(frozenset({code}), label=label)

        # Internal nodes
        self._internal: dict[str, FocalElement] = {}
        for cat in cs.all_categories:
            if cat.code in cs.leaf_codes:
                continue
            desc = cs.descendants(cat.code)
            if desc and desc != cs.leaf_codes:
                self._internal[cat.code] = FocalElement(desc, label=cat.label)

        # Confusable pairs
        self._confusables: list[FocalElement] = []
        for code_a, code_b in self._confusable_pairs:
            pair = frozenset({code_a, code_b})
            cat_a = cs.by_code.get(code_a) or cs.all_by_code.get(code_a)
            cat_b = cs.by_code.get(code_b) or cs.all_by_code.get(code_b)
            label_a = cat_a.label if cat_a else code_a
            label_b = cat_b.label if cat_b else code_b
            self._confusables.append(
                FocalElement(pair, label=f"{label_a}|{label_b}")
            )

    @property
    def singletons(self) -> dict[str, FocalElement]:
        return self._singletons

    @property
    def internal_nodes(self) -> dict[str, FocalElement]:
        return self._internal

    @property
    def confusables(self) -> list[FocalElement]:
        return self._confusables

    @cached_property
    def confusable_map(self) -> dict[str, list[FocalElement]]:
        """Map singleton code -> confusable pair FocalElements containing it."""
        result: dict[str, list[FocalElement]] = {}
        for fe in self._confusables:
            for code in fe.codes:
                result.setdefault(code, []).append(fe)
        return result

    @cached_property
    def all_focal_elements(self) -> list[FocalElement]:
        elements = list(self._singletons.values())
        elements.extend(self._internal.values())
        elements.extend(self._confusables)
        elements.append(self.theta)
        return elements

    def singleton(self, code: str) -> FocalElement:
        return self._singletons[code]

    def internal(self, code: str) -> FocalElement:
        return self._internal[code]

    def vacuous(self) -> BeliefAssignment:
        """Return the vacuous mass function (all mass on Theta)."""
        return BeliefAssignment(masses={self.theta: 1.0})
