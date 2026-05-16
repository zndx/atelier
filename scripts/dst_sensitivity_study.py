#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Numeric sensitivity study of the channel-decomposed late-interaction
mass function.

Runs a battery of synthetic scenarios across the input value space to
surface *anti-patterns by construction* — places where the design
itself (β=0.30 negative budget, top-K=5 negative selection, verifier-
pass-rate as multiplier on both channels, [0,1] positive clamp,
hierarchical aggregation only on leaf top-1, Yager fallback at K=1)
produces brittleness, non-monotonicity, mass collapse, or numerical
instability.

Tests
-----

1. **Mass conservation** — does ``sum(masses.values()) ≈ 1.0`` hold
   across the input grid?
2. **Monotonicity wrt positive_score** — for a fixed background of
   competitor scores, does increasing positive_score on a target tag
   monotonically increase its belief?
3. **Anti-monotonicity wrt negative_score** — for a fixed positive
   setup, does increasing negative_score on a target tag
   monotonically decrease its singleton mass?
4. **Conflict K transit** — sweep (positive_score, negative_score) for
   a target tag and characterize where K crosses the 0.5 logging
   threshold and where Yager fallback engages.
5. **Ranking stability under perturbation** — perturb every input by
   ±ε for small ε; how often does the top-1 prediction change?

Outputs
-------

- ``build/sensitivity/dst-<utc>/results.parquet`` — per-cell raw data
  (input parameters + computed masses + invariant evaluations).
- ``build/sensitivity/dst-<utc>/violations.json`` — list of invariant
  violations with diagnostic context.
- ``build/sensitivity/dst-<utc>/summary.json`` — aggregate stats per
  invariant.

The study runs purely on CPU; the DST math is sparse-dict and
doesn't accelerate on GPU.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Repo path setup
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from atelier.classify.belief import FrameOfDiscernment
from atelier.classify.late_interaction import TagScore
from atelier.classify.mass_functions import late_interaction_to_mass
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory

logger = logging.getLogger("dst_sensitivity_study")


# ── Test taxonomy ────────────────────────────────────────────────


def build_test_taxonomy() -> tuple[HierarchicalCategorySet, FrameOfDiscernment]:
    """Branching abstract taxonomy with multiple subtrees and depths."""
    cats = [
        # left subtree (depth 2-3)
        ReferenceCategory(code="left_root", label="left_root",
                          embedding_text="left", parent_code=None),
        ReferenceCategory(code="left_a", label="left_a",
                          embedding_text="left_a", parent_code="left_root"),
        ReferenceCategory(code="left_b", label="left_b",
                          embedding_text="left_b", parent_code="left_root"),
        ReferenceCategory(code="left_a_1", label="left_a_1",
                          embedding_text="leaf_a1", parent_code="left_a"),
        ReferenceCategory(code="left_a_2", label="left_a_2",
                          embedding_text="leaf_a2", parent_code="left_a"),
        ReferenceCategory(code="left_b_1", label="left_b_1",
                          embedding_text="leaf_b1", parent_code="left_b"),
        ReferenceCategory(code="left_b_2", label="left_b_2",
                          embedding_text="leaf_b2", parent_code="left_b"),
        # right subtree (depth 2)
        ReferenceCategory(code="right_root", label="right_root",
                          embedding_text="right", parent_code=None),
        ReferenceCategory(code="right_1", label="right_1",
                          embedding_text="right_1", parent_code="right_root"),
        ReferenceCategory(code="right_2", label="right_2",
                          embedding_text="right_2", parent_code="right_root"),
        # peer leaf (depth 1)
        ReferenceCategory(code="peer", label="peer",
                          embedding_text="peer", parent_code=None),
    ]
    parent_codes = {c.parent_code for c in cats if c.parent_code}
    leaves = [c for c in cats if c.code not in parent_codes]
    cs = HierarchicalCategorySet(
        name="sensitivity_test", categories=leaves, all_categories=cats,
    )
    return cs, FrameOfDiscernment(cs)


# ── Result types ─────────────────────────────────────────────────


@dataclass
class Violation:
    invariant: str
    detail: str
    inputs: dict
    observed: dict


@dataclass
class StudyResult:
    invariant: str
    cells_evaluated: int
    violations: list[Violation] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def violated(self) -> bool:
        return bool(self.violations)


# ── Helpers ──────────────────────────────────────────────────────


def _build_scores(positives: dict[str, float],
                  negatives: dict[str, float] | None = None,
                  verifier_rates: dict[str, float] | None = None) -> list[TagScore]:
    """Build a list of TagScore objects from per-tag positive scores.

    Missing negatives default to 0.0; missing verifier rates default to 1.0.
    """
    negatives = negatives or {}
    verifier_rates = verifier_rates or {}
    out = []
    for code, p in positives.items():
        out.append(TagScore(
            code=code,
            positive_score=p,
            negative_score=negatives.get(code, 0.0),
            verifier_pass_rate=verifier_rates.get(code, 1.0),
            per_role={},
        ))
    return out


def _belief_of_code(mass, code, frame) -> float:
    """Belief of a code in the mass function: Bel({code}).

    For a leaf, this is the sum of mass on singleton + ancestor focal
    elements whose codes ⊆ singleton.codes (trivially just the singleton
    itself for a leaf).  For an internal node, sums mass on subsets.
    """
    if code in frame.singletons:
        fe = frame.singletons[code]
    elif code in frame.internal_nodes:
        fe = frame.internal_nodes[code]
    else:
        return 0.0
    return mass.belief(fe)


def _top_code(mass, frame) -> str | None:
    """Top-1 code by post-fusion focal-element mass."""
    candidates = []
    for fe, m in mass.masses.items():
        if fe == frame.theta:
            continue
        # singletons + internal-node FEs
        for code, sfe in frame.singletons.items():
            if sfe == fe:
                candidates.append((code, m))
                break
        else:
            for code, ife in frame.internal_nodes.items():
                if ife == fe:
                    candidates.append((code, m))
                    break
    if not candidates:
        return None
    candidates.sort(key=lambda kv: (-kv[1], kv[0]))
    return candidates[0][0]


# ── Tests ────────────────────────────────────────────────────────


def test_mass_conservation(frame) -> StudyResult:
    """Total mass should sum to ≈ 1.0 across the input grid."""
    result = StudyResult(invariant="mass_conservation", cells_evaluated=0)
    # Grid: vary positive on left_a_1 in [0, 1], negative on left_a in [0, 1]
    for p in [round(0.05 * i, 2) for i in range(21)]:
        for n in [round(0.05 * j, 2) for j in range(21)]:
            scores = _build_scores(
                positives={"left_a_1": p, "right_1": 0.5},
                negatives={"left_a": n},
            )
            mass = late_interaction_to_mass(scores, frame)
            total = sum(mass.masses.values())
            result.cells_evaluated += 1
            if not (abs(total - 1.0) < 1e-9):
                result.violations.append(Violation(
                    invariant="mass_conservation",
                    detail=f"sum(masses) = {total!r} (deviation {total - 1.0:.3e})",
                    inputs={"positive_left_a_1": p, "negative_left_a": n},
                    observed={"total_mass": total},
                ))
    return result


def test_monotonicity_positive(frame) -> StudyResult:
    """Increasing positive_score on a target tag should monotonically
    increase its belief, holding competitors fixed."""
    result = StudyResult(invariant="monotonicity_positive", cells_evaluated=0)
    target = "left_a_1"
    # Sweep target's positive across [0, 1]; competitors fixed.
    competitors = {
        "left_a_2": 0.3, "left_b_1": 0.3, "right_1": 0.3, "peer": 0.2,
    }
    beliefs: list[tuple[float, float]] = []
    for p in [round(0.05 * i, 2) for i in range(21)]:
        positives = {target: p, **competitors}
        scores = _build_scores(positives=positives)
        mass = late_interaction_to_mass(scores, frame)
        bel = _belief_of_code(mass, target, frame)
        beliefs.append((p, bel))
        result.cells_evaluated += 1

    # Check non-decreasing
    for i in range(1, len(beliefs)):
        prev_p, prev_b = beliefs[i - 1]
        cur_p, cur_b = beliefs[i]
        if cur_b + 1e-9 < prev_b:
            result.violations.append(Violation(
                invariant="monotonicity_positive",
                detail=(
                    f"belief decreased from {prev_b:.6f} to {cur_b:.6f} "
                    f"as positive_score rose from {prev_p} to {cur_p}"
                ),
                inputs={"target": target, "from_p": prev_p, "to_p": cur_p,
                        "competitors": competitors},
                observed={"prev_bel": prev_b, "cur_bel": cur_b},
            ))

    result.notes.append(
        f"belief sweep over {target}.positive ∈ [0, 1]: "
        f"min={min(b for _, b in beliefs):.4f}, "
        f"max={max(b for _, b in beliefs):.4f}, "
        f"monotone-points={result.cells_evaluated - len(result.violations)}/{result.cells_evaluated}"
    )
    return result


def test_anti_monotonicity_negative(frame) -> StudyResult:
    """Increasing negative_score on a target tag should monotonically
    decrease its singleton mass (or leave it unchanged via theta), holding
    other scores fixed."""
    result = StudyResult(invariant="anti_monotonicity_negative", cells_evaluated=0)
    target = "left_a_1"
    positives = {target: 0.7, "left_a_2": 0.2, "right_1": 0.3, "peer": 0.2}
    singleton_masses: list[tuple[float, float]] = []
    for n in [round(0.05 * j, 2) for j in range(21)]:
        scores = _build_scores(positives=positives, negatives={target: n})
        mass = late_interaction_to_mass(scores, frame)
        sfe = frame.singletons[target]
        sm = mass.masses.get(sfe, 0.0)
        singleton_masses.append((n, sm))
        result.cells_evaluated += 1

    for i in range(1, len(singleton_masses)):
        prev_n, prev_m = singleton_masses[i - 1]
        cur_n, cur_m = singleton_masses[i]
        if cur_m > prev_m + 1e-9:
            result.violations.append(Violation(
                invariant="anti_monotonicity_negative",
                detail=(
                    f"singleton mass increased from {prev_m:.6f} to {cur_m:.6f} "
                    f"as negative_score rose from {prev_n} to {cur_n}"
                ),
                inputs={"target": target, "from_n": prev_n, "to_n": cur_n,
                        "positives": positives},
                observed={"prev_mass": prev_m, "cur_mass": cur_m},
            ))

    result.notes.append(
        f"singleton mass sweep over {target}.negative ∈ [0, 1]: "
        f"start={singleton_masses[0][1]:.4f}, "
        f"end={singleton_masses[-1][1]:.4f}, "
        f"monotone-points={result.cells_evaluated - len(result.violations)}/{result.cells_evaluated}"
    )
    return result


def test_conflict_k_transit(frame) -> StudyResult:
    """Directly measure channel-conflict K across the input grid by
    calling dempster_combine on the internal positive/negative mass
    functions.

    Records the (positive, negative) → K surface and notes:
      - max K observed across the grid
      - the (p, n) region where K crosses the 0.5 log-warning threshold
      - whether K ever approaches 1.0 (Yager fallback boundary)
    """
    from atelier.classify.belief import dempster_combine
    from atelier.classify.mass_functions import (
        _AttenuatedTagScore, _late_interaction_negative_mass,
        _late_interaction_positive_mass,
    )

    result = StudyResult(invariant="conflict_k_transit", cells_evaluated=0)
    target = "left_a_1"
    positives_baseline = {"left_a_2": 0.2, "right_1": 0.3, "peer": 0.2}
    k_max = 0.0
    k_above_half = 0
    k_near_one = 0
    samples_at_k_05 = []
    samples_at_k_max = []
    for p in [round(0.05 * i, 2) for i in range(21)]:
        for n in [round(0.05 * j, 2) for j in range(21)]:
            positives = {**positives_baseline, target: p}
            scores = _build_scores(positives=positives, negatives={target: n})
            attenuated = [
                _AttenuatedTagScore(
                    code=s.code,
                    positive_score=s.positive_score * s.verifier_pass_rate,
                    negative_score=s.negative_score * s.verifier_pass_rate,
                ) for s in scores
            ]
            m_pos = _late_interaction_positive_mass(
                attenuated, frame, discount=0.20, reliability_floor=0.10,
            )
            m_neg = _late_interaction_negative_mass(
                attenuated, frame, beta=0.30, top_k=5, min_score=0.10,
            )
            try:
                _, k = dempster_combine(m_pos, m_neg)
            except ValueError:
                # K = 1 (total conflict)
                k = 1.0
            result.cells_evaluated += 1
            if k > k_max:
                k_max = k
                samples_at_k_max = [{"p": p, "n": n, "K": round(k, 4)}]
            elif k >= k_max - 1e-6:
                samples_at_k_max.append({"p": p, "n": n, "K": round(k, 4)})
            if k >= 0.5:
                k_above_half += 1
                if len(samples_at_k_05) < 5:
                    samples_at_k_05.append({"p": p, "n": n, "K": round(k, 4)})
            if k >= 0.99:
                k_near_one += 1

    result.notes.append(
        f"K observed range: [0.000, {k_max:.4f}] across {result.cells_evaluated} cells"
    )
    result.notes.append(
        f"cells with K ≥ 0.5 (channel-conflict log threshold): {k_above_half}/{result.cells_evaluated}"
    )
    result.notes.append(
        f"cells with K ≥ 0.99 (near-Yager-fallback): {k_near_one}/{result.cells_evaluated}"
    )
    if samples_at_k_05:
        result.notes.append(
            "first K≥0.5 cells: " + json.dumps(samples_at_k_05)
        )
    if samples_at_k_max:
        result.notes.append(
            f"K-max cells (K={k_max:.4f}): " + json.dumps(samples_at_k_max[:3])
        )
    # By construction with β=0.30, K is bounded by ~β × m_pos({target}),
    # so K cannot approach 1.  Document this if it holds.
    if k_max < 0.4:
        result.notes.append(
            f"observation: max K {k_max:.4f} stays well below 1.0 "
            f"— β=0.30 caps the negative mass budget, bounding K by "
            f"design.  Yager fallback effectively never engages under "
            f"normal operating ranges.  If higher K is desired (e.g., "
            f"to surface stronger channel disagreement as needs_clarification), "
            f"β must be raised — at the cost of broader anti-example "
            f"over-suppression risk."
        )
    return result


def test_ranking_stability(frame, epsilon: float = 0.01,
                           n_trials: int = 200) -> StudyResult:
    """Perturb all inputs by ±epsilon and check that top-1 prediction
    is stable across trials for the same base configuration."""
    import random
    rng = random.Random(42)
    result = StudyResult(invariant="ranking_stability_under_perturbation",
                         cells_evaluated=0)
    # Test multiple base configurations spanning easy to adversarial.
    # The tight-margin configs should exhibit instability if the
    # design is brittle near top-1 crossover points.
    base_configs = [
        # 1. confident leaf — should be perfectly stable
        {"positives": {"left_a_1": 0.85, "left_a_2": 0.20, "right_1": 0.30,
                       "peer": 0.10}, "negatives": {}},
        # 2. near-tie between two subtrees (Δ = 0.02)
        {"positives": {"left_a_1": 0.55, "right_1": 0.53, "peer": 0.20},
         "negatives": {}},
        # 3. positive + competing negative
        {"positives": {"left_a_1": 0.65, "right_1": 0.40},
         "negatives": {"left_a_1": 0.55}},
        # 4. TIGHT margin (Δ = 0.005) — under ε=0.01 perturbation, the
        #    top-1 *should* flip in some fraction of trials by construction
        {"positives": {"left_a_1": 0.500, "right_1": 0.495, "peer": 0.20},
         "negatives": {}},
        # 5. three-way near-tie — most sensitive
        {"positives": {"left_a_1": 0.50, "right_1": 0.50, "peer": 0.50},
         "negatives": {}},
    ]
    for config_idx, config in enumerate(base_configs):
        base_scores = _build_scores(**config)
        base_mass = late_interaction_to_mass(base_scores, frame)
        base_top = _top_code(base_mass, frame)
        instability_count = 0
        differing_picks: dict[str, int] = {}
        for _ in range(n_trials):
            perturbed_positives = {
                k: max(0.0, min(1.0, v + rng.uniform(-epsilon, epsilon)))
                for k, v in config["positives"].items()
            }
            perturbed_negatives = {
                k: max(0.0, min(1.0, v + rng.uniform(-epsilon, epsilon)))
                for k, v in config["negatives"].items()
            }
            scores = _build_scores(positives=perturbed_positives,
                                   negatives=perturbed_negatives)
            mass = late_interaction_to_mass(scores, frame)
            top = _top_code(mass, frame)
            result.cells_evaluated += 1
            if top != base_top:
                instability_count += 1
                differing_picks[top or "<None>"] = differing_picks.get(top or "<None>", 0) + 1

        instability_rate = instability_count / n_trials
        # Diagnostic margin between base_top and runner-up
        top_two = sorted(config["positives"].items(), key=lambda kv: -kv[1])[:2]
        margin = abs(top_two[0][1] - top_two[1][1]) if len(top_two) >= 2 else 1.0
        result.notes.append(
            f"config[{config_idx}] base_top={base_top!r} input_margin={margin:.3f}: "
            f"instability_rate={instability_rate:.2%} "
            f"({instability_count}/{n_trials}); "
            f"differing_picks={differing_picks}"
        )
        # Flag as violation ONLY when a *confident* config exhibits
        # instability — instability on near-tie configs is correct
        # behavior, not a bug.  A "confident" config has input_margin
        # > 2ε; flagging at >5% instability there indicates real
        # brittleness.
        if margin > 2 * epsilon and instability_rate > 0.05:
            result.violations.append(Violation(
                invariant="ranking_stability_under_perturbation",
                detail=(
                    f"top-1 changed in {instability_count}/{n_trials} "
                    f"trials under ε={epsilon} despite input_margin "
                    f"{margin:.3f} > 2ε ({2 * epsilon:.3f}) — "
                    f"brittleness beyond what the margin predicts"
                ),
                inputs={"config": config, "base_top": base_top,
                        "epsilon": epsilon, "n_trials": n_trials,
                        "input_margin": margin},
                observed={"instability_rate": instability_rate,
                          "differing_picks": differing_picks},
            ))
    return result


def test_verifier_pass_rate_effect(frame) -> StudyResult:
    """Sweep verifier_pass_rate ∈ [0, 1] on a target tag; expect smooth
    monotonic scaling of singleton mass (no discontinuities)."""
    result = StudyResult(invariant="verifier_pass_rate_effect",
                         cells_evaluated=0)
    target = "left_a_1"
    positives = {target: 0.7, "left_a_2": 0.3, "right_1": 0.3, "peer": 0.2}
    singleton_masses: list[tuple[float, float]] = []
    for vr in [round(0.05 * i, 2) for i in range(21)]:
        scores = _build_scores(positives=positives,
                               verifier_rates={target: vr})
        mass = late_interaction_to_mass(scores, frame)
        sfe = frame.singletons[target]
        singleton_masses.append((vr, mass.masses.get(sfe, 0.0)))
        result.cells_evaluated += 1

    # Check non-decreasing (higher verifier-pass-rate should not hurt)
    for i in range(1, len(singleton_masses)):
        prev_vr, prev_m = singleton_masses[i - 1]
        cur_vr, cur_m = singleton_masses[i]
        if cur_m + 1e-9 < prev_m:
            result.violations.append(Violation(
                invariant="verifier_pass_rate_effect",
                detail=(
                    f"singleton mass decreased from {prev_m:.6f} to {cur_m:.6f} "
                    f"as verifier_pass_rate rose from {prev_vr} to {cur_vr}"
                ),
                inputs={"target": target, "from_vr": prev_vr, "to_vr": cur_vr},
                observed={"prev_mass": prev_m, "cur_mass": cur_m},
            ))

    result.notes.append(
        f"verifier_pass_rate sweep on {target}: "
        f"mass at vr=0.0 is {singleton_masses[0][1]:.4f}, "
        f"mass at vr=1.0 is {singleton_masses[-1][1]:.4f}, "
        f"max_step={max(abs(singleton_masses[i][1] - singleton_masses[i-1][1]) for i in range(1, len(singleton_masses))):.4f}"
    )
    return result


# ── Driver ───────────────────────────────────────────────────────


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    out_dir = Path("build/sensitivity") / f"dst-{utc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output: %s", out_dir)

    cs, frame = build_test_taxonomy()
    logger.info("Test taxonomy: %d leaves, %d internal nodes",
                len(frame.singletons), len(frame.internal_nodes))

    tests = [
        ("mass_conservation", test_mass_conservation),
        ("monotonicity_positive", test_monotonicity_positive),
        ("anti_monotonicity_negative", test_anti_monotonicity_negative),
        ("conflict_k_transit", test_conflict_k_transit),
        ("ranking_stability", test_ranking_stability),
        ("verifier_pass_rate_effect", test_verifier_pass_rate_effect),
    ]
    all_results: list[StudyResult] = []
    for name, fn in tests:
        logger.info("Running test: %s", name)
        r = fn(frame)
        logger.info(
            "  cells=%d violations=%d %s",
            r.cells_evaluated, len(r.violations),
            "(violated)" if r.violated else "(clean)",
        )
        for note in r.notes:
            logger.info("    note: %s", note)
        all_results.append(r)

    # Aggregate
    summary = {
        "started_at": utc,
        "test_taxonomy": {
            "leaves": list(frame.singletons),
            "internal_nodes": list(frame.internal_nodes),
        },
        "tests": [
            {
                "invariant": r.invariant,
                "cells_evaluated": r.cells_evaluated,
                "violated": r.violated,
                "violation_count": len(r.violations),
                "notes": r.notes,
            }
            for r in all_results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    violations = []
    for r in all_results:
        for v in r.violations:
            violations.append(asdict(v))
    (out_dir / "violations.json").write_text(
        json.dumps(violations, indent=2, default=str)
    )

    logger.info("Wrote %s", out_dir / "summary.json")
    logger.info("Wrote %s", out_dir / "violations.json")
    print(json.dumps({
        "output_dir": str(out_dir),
        "total_cells": sum(r.cells_evaluated for r in all_results),
        "total_violations": sum(len(r.violations) for r in all_results),
        "tests_violated": [r.invariant for r in all_results if r.violated],
    }, indent=2))

    return 1 if any(r.violated for r in all_results) else 0


if __name__ == "__main__":
    sys.exit(main())
