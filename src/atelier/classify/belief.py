# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

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

    def most_committed_singleton(self) -> tuple[str, float] | None:
        """Return the (code, mass) of the highest-mass singleton, or None.

        Used by the independent-tier consensus extraction in
        ``_classify_column`` (see Shafer 1976 §11.3 reliability discount
        and Denoeux 2008 on non-distinct evidence): the consensus of
        truly LLM-independent sources is taken as the argmax over the
        fused mass restricted to singleton focal elements.
        """
        best: tuple[str, float] | None = None
        for fe, m in self.masses.items():
            if len(fe.codes) != 1:
                continue
            if best is None or m > best[1]:
                best = (next(iter(fe.codes)), m)
        return best


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


def yager_combine(
    m1: BeliefAssignment,
    m2: BeliefAssignment,
    theta: FocalElement,
) -> tuple[BeliefAssignment, float]:
    """Yager's modified combination rule.

    Redirects all conflict mass to ignorance (Θ) instead of normalizing
    by (1-K).  Preserves epistemic honesty at the cost of higher
    ignorance mass under conflict — the "winner" in a conflicted fusion
    is not artificially amplified.

    When K=0, produces the same result as Dempster's rule.

    Args:
        m1, m2: Mass functions to combine.
        theta: The frame of discernment focal element (Θ), required so
            we don't have to detect it at runtime.

    Returns:
        (combined_assignment, conflict) where conflict is the mass that
        was redirected to Θ (diagnostic, not used for normalization).
    """
    combined: dict[FocalElement, float] = {}
    conflict = 0.0

    for fe1, mass1 in m1.masses.items():
        for fe2, mass2 in m2.masses.items():
            product = mass1 * mass2
            intersection = fe1.codes & fe2.codes
            if intersection:
                fe = FocalElement(frozenset(intersection))
                combined[fe] = combined.get(fe, 0.0) + product
            else:
                conflict += product

    # Redirect conflict mass to Θ (no normalization)
    if conflict > 0:
        combined[theta] = combined.get(theta, 0.0) + conflict

    # Drop near-zero masses for consistency with dempster_combine
    result = {fe: m for fe, m in combined.items() if m > 1e-15}
    return BeliefAssignment(masses=result), conflict


def combine_multiple(
    assignments: list[BeliefAssignment],
    *,
    strategy: str = "dempster",
    theta: FocalElement | None = None,
) -> tuple[BeliefAssignment, float]:
    """Left-to-right combination of multiple sources.

    Args:
        assignments: Mass functions to combine.
        strategy: "dempster" (default, normalizing) or "yager"
            (redirect conflict to Θ).
        theta: Required when strategy="yager" — the frame of discernment.

    Returns:
        (combined_assignment, conflict) where:
        - For Dempster: conflict is cumulative K via Smarandache's formula
          ``K = 1 - ∏(1 - Kᵢ)``
        - For Yager: conflict is the total mass redirected to Θ across
          all combination steps (diagnostic, additive)
    """
    if not assignments:
        raise ValueError("Cannot combine empty list of assignments")

    if strategy == "yager":
        if theta is None:
            raise ValueError("Yager combination requires theta parameter")
        result = assignments[0]
        total_conflict = 0.0
        for other in assignments[1:]:
            result, k_i = yager_combine(result, other, theta)
            total_conflict += k_i
        return result, total_conflict

    if strategy != "dempster":
        raise ValueError(f"Unknown fusion strategy: {strategy!r}")

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
        """True when the prediction's support is weak.

        Uses belief-gap criteria (belief mass on the winner + spread of
        the belief interval) rather than Dempster's conflict K.  Under
        Dempster's rule K is normalized out, so ``conflict > 0.2``
        evaluates true for essentially every column — it's a bias-
        locked flag that carries no information.  The gap-based
        criterion is fusion-strategy-invariant: a wide [Bel, Pl]
        interval or a low-belief winner deserves clarification
        regardless of whether the fusion was Dempster or Yager.

        Thresholds (0.80 bel floor, 0.20 gap ceiling) match the
        bootstrap ``bel_floor`` / ``gap_threshold`` defaults so this
        flag and the revisit-candidate criterion tell the same story.
        """
        if self.belief_assignment is None or self._frame is None:
            return False
        code = self.category.code if self.category else None
        if not code:
            return True  # no prediction at all — trivially needs clarification
        bel = self.belief_at(code)
        pl = self.plausibility_at(code)
        return bel < 0.80 or (pl - bel) > 0.20

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

    def cross_subtree_belief(
        self,
        bel_threshold: float = 0.20,
        *,
        always_include_top_per_subtree: bool = True,
        min_bel: float = 0.05,
    ) -> list[dict]:
        """Return belief for codes across the full hierarchy, surfacing
        cross-subtree alternatives that ``belief_path`` cannot show.

        Walks every singleton AND every internal-node focal element in
        the frame and returns the structured belief view.  Three
        inclusion rules cooperate:

        1. **Threshold inclusion** — any code (leaf or internal) where
           ``Bel ≥ bel_threshold`` is included.  Default 0.20 surfaces
           non-trivial competing belief without flooding the result;
           0.5 (the prior default) was too strict to surface
           alternatives when one source dominates Dempster fusion.

        2. **Top-per-subtree inclusion** — when
           ``always_include_top_per_subtree`` is True, the highest-
           belief leaf AND highest-belief internal node from each
           top-level subtree (grouped by the first code component) is
           included regardless of threshold, provided ``Bel ≥
           min_bel``.  This guarantees that the operator-facing view
           always shows the "competing subtree" candidate even when
           Dempster fusion has compressed its mass below the headline
           threshold — the loan-amount-as-non-sensitive failure mode
           was structurally invisible without this rule.

        3. **De-duplication** — each code appears at most once.

        Sorted most-specific first (deepest code by dot count); ties
        broken by descending belief.

        Used by :meth:`cautious_code` and surfaced to operators in the
        result dict (``cross_subtree_belief``) so cross-subtree
        disagreement between LLM (leaf vote) and cosine / pattern /
        name_match (subtree localization) is always visible.
        """
        if self.belief_assignment is None or self._frame is None:
            return []
        if self._category_set is None:
            return []

        seen: set[str] = set()
        rows: list[dict] = []

        def _resolve_label(code: str) -> str:
            cat = (
                getattr(self._category_set, "all_by_code", {}).get(code)
                or getattr(self._category_set, "by_code", {}).get(code)
            )
            return cat.label if cat else code

        def _add(code: str, fe, kind: str) -> None:
            if code in seen:
                return
            bel = self.belief_assignment.belief(fe)
            seen.add(code)
            rows.append({
                "code": code,
                "label": _resolve_label(code),
                "bel": round(bel, 3),
                "pl": round(self.belief_assignment.plausibility(fe), 3),
                "depth": code.count("."),
                "kind": kind,
            })

        # Pass 1: threshold inclusion.
        for code, fe in self._frame.singletons.items():
            if self.belief_assignment.belief(fe) >= bel_threshold:
                _add(code, fe, "leaf")
        for code, fe in self._frame.internal_nodes.items():
            if self.belief_assignment.belief(fe) >= bel_threshold:
                _add(code, fe, "internal")

        # Pass 2: top-per-subtree inclusion.  Group by first code
        # component (the top-level subtree root); for each group,
        # add the highest-belief leaf and the highest-belief
        # internal node above ``min_bel``.
        if always_include_top_per_subtree:
            def _root(code: str) -> str:
                return code.split(".", 1)[0] if code else ""

            roots: dict[str, dict[str, tuple[str, object, float]]] = {}
            for code, fe in self._frame.singletons.items():
                bel = self.belief_assignment.belief(fe)
                if bel < min_bel:
                    continue
                r = _root(code)
                slot = roots.setdefault(r, {})
                cur = slot.get("leaf")
                if cur is None or bel > cur[2]:
                    slot["leaf"] = (code, fe, bel)
            for code, fe in self._frame.internal_nodes.items():
                bel = self.belief_assignment.belief(fe)
                if bel < min_bel:
                    continue
                r = _root(code)
                slot = roots.setdefault(r, {})
                cur = slot.get("internal")
                if cur is None or bel > cur[2]:
                    slot["internal"] = (code, fe, bel)
            for slot in roots.values():
                if "leaf" in slot:
                    code, fe, _ = slot["leaf"]
                    _add(code, fe, "leaf")
                if "internal" in slot:
                    code, fe, _ = slot["internal"]
                    _add(code, fe, "internal")

        rows.sort(key=lambda r: (-r["depth"], -r["bel"]))
        return rows

    def cautious_promoted_code(
        self,
        commit_threshold: float = 0.55,
        *,
        require_clarification: bool = True,
    ) -> dict:
        """Apply Smets' least-commitment principle to surface a more
        honest prediction than the leaf-argmax when evidence is
        conflicted.

        Returns a dict ``{"code", "label", "depth", "bel",
        "promoted_from", "rationale"}`` describing either:

        * the predicted leaf (no promotion) — ``promoted_from`` is
          None and ``rationale`` explains why no promotion fired; OR
        * a more-general code (an internal node, possibly in a
          *different* subtree from the predicted leaf) whose belief
          is above ``commit_threshold`` and which is the most-
          specific such code in the frame.

        Promotion fires only when ``require_clarification`` is True
        AND ``self.needs_clarification`` is True — operators get the
        leaf prediction by default; the cautious promotion is the
        epistemically-honest *fallback* when the system itself flags
        the prediction as uncertain.

        Smets 1993 (*Belief Functions: The Disjunctive Rule of
        Combination and the Generalized Bayesian Theorem*) and
        related work on least-commitment: when a fine-grained
        decision is unsupported by evidence, the principled response
        is to commit only at the level of granularity where evidence
        IS unambiguous.  Our leaf prediction stays the user-facing
        ``predicted_code`` for backward compatibility; the promoted
        code lives in a separate field operators can consult when
        ``needs_clarification`` is True.
        """
        if self.category is None:
            return {
                "code": "",
                "label": "",
                "depth": 0,
                "bel": 0.0,
                "promoted_from": None,
                "rationale": "no predicted category",
            }

        predicted_code = self.category.code
        predicted_bel = self.belief_at(predicted_code)
        predicted_label = getattr(self.category, "label", predicted_code)
        predicted_depth = predicted_code.count(".")

        no_promote = {
            "code": predicted_code,
            "label": predicted_label,
            "depth": predicted_depth,
            "bel": round(predicted_bel, 3),
            "promoted_from": None,
            "rationale": "predicted leaf retained",
        }

        if require_clarification and not self.needs_clarification:
            no_promote["rationale"] = "prediction is settled (needs_clarification=False)"
            return no_promote

        # If the predicted leaf already has Bel ≥ commit_threshold,
        # no promotion needed — the leaf is unambiguous enough.
        if predicted_bel >= commit_threshold:
            no_promote["rationale"] = (
                f"predicted leaf Bel={predicted_bel:.2f} ≥ commit_threshold "
                f"{commit_threshold:.2f}"
            )
            return no_promote

        # Find the most-specific code anywhere in the frame whose
        # belief meets the commit threshold.
        rows = self.cross_subtree_belief(bel_threshold=commit_threshold)
        if not rows:
            no_promote["rationale"] = (
                f"no code anywhere reaches commit_threshold "
                f"{commit_threshold:.2f}; retained predicted leaf"
            )
            return no_promote

        # Most-specific first; if the top row IS the predicted code,
        # no promotion (we'd be promoting to ourselves).
        top = rows[0]
        if top["code"] == predicted_code:
            no_promote["rationale"] = (
                f"predicted leaf is already the most-specific code with "
                f"Bel ≥ {commit_threshold:.2f}"
            )
            return no_promote

        return {
            "code": top["code"],
            "label": top["label"],
            "depth": top["depth"],
            "bel": top["bel"],
            "promoted_from": predicted_code,
            "rationale": (
                f"predicted leaf {predicted_code} (Bel={predicted_bel:.2f}) "
                f"below commit_threshold {commit_threshold:.2f}; promoted to "
                f"most-specific code with Bel≥{commit_threshold:.2f} "
                f"(needs_clarification=True; Smets least-commitment)"
            ),
        }

    def cautious_code(self, bel_threshold: float = 0.7) -> str:
        """Return the most-specific code anywhere in the hierarchy
        whose belief meets the threshold.

        Answers: 'At what taxonomy level is evidence unambiguous —
        considering the full frame, not just the predicted leaf's
        ancestor chain?'  Walks every singleton AND every internal-
        node focal element so a subtree-localized signal that
        conflicts with the LLM's leaf vote can still surface (the
        loan-amount-as-non-sensitive failure mode: cosine localizes
        to ``Financial Data`` but the LLM picks ``Internal Non-
        Sensitive``; cautious_code should be able to return
        ``Financial Data`` despite it sitting in a different
        subtree from the predicted leaf).

        Tie-break is most-specific first, then highest belief.  Falls
        back to the predicted leaf or root when no code meets the
        threshold.

        Important: ``cross_subtree_belief`` returns two kinds of rows
        — codes that actually clear ``bel_threshold`` (the cautious
        candidates) and an "always-include-top-per-subtree" fallback
        intended for operator-facing display.  This method
        deliberately filters to the former: a fallback row at low
        belief is informative for the UI but is NOT a code at which
        evidence is unambiguous, so it must not be returned as the
        cautious choice.
        """
        rows = self.cross_subtree_belief(bel_threshold=bel_threshold)
        eligible = [r for r in rows if r["bel"] >= bel_threshold]
        if eligible:
            # cross_subtree_belief sorts most-specific first; preserve.
            return eligible[0]["code"]
        # No code meets the threshold — fall back to walking the
        # predicted code's ancestor chain (legacy semantics) so the
        # caller still gets a non-empty answer.
        path = self.belief_path()
        for entry in path:
            if entry["bel"] >= bel_threshold:
                return entry["code"]
        return path[-1]["code"] if path else ""

    @classmethod
    def from_combined_evidence(
        cls,
        source_masses: dict[str, BeliefAssignment],
        frame: FrameOfDiscernment,
        category_set,
        sensitivity_code: str | None = None,
        *,
        fusion_strategy: str = "dempster",
    ) -> HierarchicalClassification:
        """Combine source masses, find best category.

        Filters out vacuous sources, combines the rest, ranks singletons
        by pignistic probability, and builds a rich evidence string.

        Args:
            fusion_strategy: "dempster" (default, normalizing) or "yager"
                (redirect conflict to Θ).  Yager preserves epistemic
                honesty under high-conflict fusion at the cost of higher
                ignorance mass.
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
                combined, conflict = combine_multiple(
                    non_vacuous,
                    strategy=fusion_strategy,
                    theta=frame.theta if fusion_strategy == "yager" else None,
                )
            except ValueError:
                combined = frame.vacuous()
                conflict = 1.0

        # Pick the headline focal element — Atlas-style "every node is
        # a first-class tag" + Smets least-commitment.  Iterate both
        # singletons and internal-node focal elements; rank by *direct*
        # mass m(A) (not Bel(A), which sums subset masses and would
        # systematically promote ancestors over their committed
        # descendants).  Tie-break: most-specific first (deeper code
        # wins), then deterministic by code.  Theta and depth-0 root
        # are excluded — total ignorance is never a useful tag.
        # Confusable pairs are excluded — "A or B" is not a single tag.
        #
        # Fallback: when no focal element has direct mass on a depth≥1
        # code (every source vacuous, all mass on Theta), revert to the
        # leaf-pignistic argmax so we always return *some* code.
        internal_codes = {fe: code for code, fe in frame.internal_nodes.items()}

        candidates: list[tuple[str, float, int]] = []
        for fe, m in combined.masses.items():
            if m < 1e-9 or fe == frame.theta:
                continue
            if len(fe.codes) == 1:
                code = next(iter(fe.codes))
            elif fe in internal_codes:
                code = internal_codes[fe]
            else:
                continue
            depth = code.count(".")
            if depth == 0 or not code:
                continue
            candidates.append((code, m, depth))

        best_code: str | None = None
        if candidates:
            candidates.sort(key=lambda c: (-c[1], -c[2], c[0]))
            best_code = candidates[0][0]

        if best_code is None:
            best_betp = -1.0
            for code, singleton in frame.singletons.items():
                betp = combined.pignistic_probability(singleton)
                if betp > best_betp:
                    best_betp = betp
                    best_code = code

        if best_code is None:
            raise ValueError("No singletons in frame")

        # Resolve the headline focal element for downstream Bel/Pl
        # queries.  Singleton vs internal-node dispatch — the prior
        # leaf-only picker only ever needed ``frame.singleton``.
        if best_code in frame.singletons:
            best_fe = frame.singletons[best_code]
            best_betp = combined.pignistic_probability(best_fe)
        else:
            best_fe = frame.internal_nodes[best_code]
            # For an internal-node headline, BetP is undefined as a
            # singleton transform; report Bel(A) as the analog
            # operator-facing "confidence in this tag".
            best_betp = combined.belief(best_fe)

        cat = category_set.by_code.get(best_code)
        if cat is None and hasattr(category_set, "all_by_code"):
            cat = category_set.all_by_code.get(best_code)

        # Build evidence string with per-source top-1 *code* (not just
        # mass) so operators can see at a glance which source voted
        # which code — disagreement is then visible without consulting
        # the evidence_sources field separately.  Singletons and
        # internal-node focal elements both qualify; the latter
        # surfaces parent-level votes (Atlas-style "every node is a
        # tag") which would otherwise read as ``name=0.000`` despite
        # carrying real evidence.
        source_parts = []
        for name, ba in source_masses.items():
            top: tuple[float, str | None] = (0.0, None)
            for fe, m in ba.masses.items():
                if fe == frame.theta:
                    continue
                if len(fe.codes) == 1:
                    code = next(iter(fe.codes))
                elif fe in internal_codes:
                    code = internal_codes[fe]
                else:
                    continue
                if m > top[0]:
                    top = (m, code)
            if top[1]:
                source_parts.append(f"{name}→{top[1]}({top[0]:.2f})")
            else:
                source_parts.append(f"{name}=0.000")

        bel = combined.belief(best_fe)
        pl = combined.plausibility(best_fe)

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

        # Cross-subtree alternative summary — when a non-predicted
        # subtree's internal node carries non-trivial belief,
        # surface it inline so the operator sees the competing
        # honest answer alongside the predicted leaf.  The strict
        # 0.5 threshold on cross_subtree_belief is too high to
        # surface alternatives when the LLM dominates fusion (Bel
        # in competing subtrees gets normalized down by Dempster
        # against a confident LLM); use a lower 0.15 absolute
        # threshold here to expose the conflict where it matters.
        predicted_subtree_root = best_code.split(".")[0] if best_code else ""
        competing_internals: list[tuple[str, str, float]] = []
        for code, fe in frame.internal_nodes.items():
            if not code:
                continue
            if code.split(".")[0] == predicted_subtree_root:
                continue  # same subtree as the prediction — skip
            ib = combined.belief(fe)
            if ib >= 0.15:
                anc = (
                    getattr(category_set, "all_by_code", {}).get(code)
                    or getattr(category_set, "by_code", {}).get(code)
                )
                label = anc.label if anc else code
                competing_internals.append((code, label, ib))
        if competing_internals:
            competing_internals.sort(key=lambda c: -c[2])
            top_competing = competing_internals[0]
            evidence += (
                f" [competing: {top_competing[1]} "
                f"({top_competing[0]}) Bel={top_competing[2]:.2f}]"
            )

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
