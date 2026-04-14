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


@dataclass(frozen=True)
class HierarchicalClassification:
    """Classification result with Dempster-Shafer belief intervals.

    Wraps a combined BeliefAssignment with hierarchy navigation methods
    so consumers can query belief at any level of the taxonomy.
    """

    category: object  # ReferenceCategory — kept generic to avoid circular import
    confidence: float  # pignistic probability of the predicted category
    evidence: str  # human-readable evidence summary
    sensitivity_code: str | None = None

    # DST fields
    belief_assignment: BeliefAssignment | None = None
    conflict: float = 0.0
    source_masses: dict[str, BeliefAssignment] = field(default_factory=dict)

    # References for hierarchy navigation (not serialized)
    _frame: FrameOfDiscernment | None = field(default=None, repr=False, compare=False)
    _category_set: object | None = field(default=None, repr=False, compare=False)

    def belief_at(self, code: str) -> float:
        """Bel(code) — belief that the column is this category."""
        if self.belief_assignment is None:
            return 0.0
        if self._frame is not None:
            if code in self._frame.singletons:
                return self.belief_assignment.belief(self._frame.singletons[code])
            if code in self._frame.internal_nodes:
                return self.belief_assignment.belief(self._frame.internal_nodes[code])
        # Fallback: construct a focal element from descendants
        if self._category_set is not None and hasattr(self._category_set, "descendants"):
            desc = self._category_set.descendants(code)
            return self.belief_assignment.belief(FocalElement(desc))
        return self.belief_assignment.belief(FocalElement(frozenset({code})))

    def plausibility_at(self, code: str) -> float:
        """Pl(code) — plausibility that the column is this category."""
        if self.belief_assignment is None:
            return 0.0
        if self._frame is not None:
            if code in self._frame.singletons:
                return self.belief_assignment.plausibility(self._frame.singletons[code])
            if code in self._frame.internal_nodes:
                return self.belief_assignment.plausibility(self._frame.internal_nodes[code])
        if self._category_set is not None and hasattr(self._category_set, "descendants"):
            desc = self._category_set.descendants(code)
            return self.belief_assignment.plausibility(FocalElement(desc))
        return self.belief_assignment.plausibility(FocalElement(frozenset({code})))

    def interval_at(self, code: str) -> tuple[float, float]:
        """Return (Bel, Pl) at code."""
        return (self.belief_at(code), self.plausibility_at(code))

    @property
    def uncertainty_gap(self) -> float:
        """Pl - Bel for the predicted category."""
        if self.belief_assignment is None or self._frame is None:
            return 0.0
        code = self.category.code
        return self.plausibility_at(code) - self.belief_at(code)

    @property
    def needs_clarification(self) -> bool:
        """True when uncertainty gap > 0.3 or conflict > 0.2."""
        return self.uncertainty_gap > 0.3 or self.conflict > 0.2

    def belief_path(self) -> list[dict]:
        """Trace [Bel, Pl] from predicted leaf to root.

        Returns list of dicts from leaf (most specific) to root (least specific):
        [{"code": "ICE...PAN", "label": "...", "bel": 0.45, "pl": 0.90, "depth": 7}, ...]

        Key property: Bel increases (or stays same) ascending — coarser
        categories are always at least as certain as finer ones.
        """
        if self.category is None or self._category_set is None:
            return []
        path = [{
            "code": self.category.code,
            "label": getattr(self.category, "label", self.category.code),
            "bel": round(self.belief_at(self.category.code), 3),
            "pl": round(self.plausibility_at(self.category.code), 3),
            "depth": self.category.code.count("."),
        }]
        ancestors = self._category_set.ancestors(self.category.code)
        for anc_code in ancestors:
            anc_cat = (
                getattr(self._category_set, "all_by_code", {}).get(anc_code)
                or getattr(self._category_set, "by_code", {}).get(anc_code)
            )
            path.append({
                "code": anc_code,
                "label": anc_cat.label if anc_cat else anc_code,
                "bel": round(self.belief_at(anc_code), 3),
                "pl": round(self.plausibility_at(anc_code), 3),
                "depth": anc_code.count("."),
            })
        return path

    def cautious_code(self, bel_threshold: float = 0.7) -> str:
        """Return deepest code where Bel exceeds threshold.

        Answers: 'At what taxonomy level is evidence unambiguous?'
        May return a parent code when leaf-level evidence is ambiguous.
        """
        path = self.belief_path()
        for entry in path:  # leaf-first ordering
            if entry["bel"] >= bel_threshold:
                return entry["code"]
        return path[-1]["code"] if path else ""  # fall back to root

    @classmethod
    def from_combined_evidence(
        cls,
        source_masses: dict[str, BeliefAssignment],
        frame: FrameOfDiscernment,
        category_set,
        sensitivity_code: str | None = None,
    ) -> HierarchicalClassification:
        """Combine source masses via Dempster's rule, find best category.

        Filters out vacuous sources, combines the rest, ranks singletons
        by pignistic probability, and builds a rich evidence string.
        """
        # Filter out vacuous sources
        non_vacuous = [
            ba for name, ba in source_masses.items()
            if len(ba.masses) > 1 or (len(ba.masses) == 1 and frame.theta not in ba.masses)
        ]

        if not non_vacuous:
            combined = frame.vacuous()
            conflict = 0.0
        else:
            try:
                combined, conflict = combine_multiple(non_vacuous)
            except ValueError:
                combined = frame.vacuous()
                conflict = 1.0

        # Find best category via pignistic probability
        best_code = None
        best_betp = -1.0
        for code, singleton in frame.singletons.items():
            betp = combined.pignistic_probability(singleton)
            if betp > best_betp:
                best_betp = betp
                best_code = code

        if best_code is None:
            raise ValueError("No singletons in frame")

        cat = category_set.by_code.get(best_code)
        if cat is None and hasattr(category_set, "all_by_code"):
            cat = category_set.all_by_code.get(best_code)

        # Build evidence string
        source_parts = []
        for name, ba in source_masses.items():
            best_mass = max(
                (m for fe, m in ba.masses.items() if len(fe.codes) == 1),
                default=0.0,
            )
            source_parts.append(f"{name}={best_mass:.3f}")

        bel = combined.belief(frame.singleton(best_code))
        pl = combined.plausibility(frame.singleton(best_code))

        confusable_parts: list[str] = []
        for fe, m in combined.masses.items():
            if len(fe.codes) == 2 and m > 0.05 and fe.label:
                confusable_parts.append(f"{fe.label}={m:.2f}")

        evidence = (
            f"dst({', '.join(source_parts)}) → {cat.label} "
            f"[Bel={bel:.2f}, Pl={pl:.2f}, K={conflict:.2f}]"
        )
        if confusable_parts:
            evidence += f" [confusable: {', '.join(confusable_parts)}]"

        return cls(
            category=cat,
            confidence=round(best_betp, 3),
            evidence=evidence,
            sensitivity_code=sensitivity_code,
            belief_assignment=combined,
            conflict=round(conflict, 4),
            source_masses=source_masses,
            _frame=frame,
            _category_set=category_set,
        )
