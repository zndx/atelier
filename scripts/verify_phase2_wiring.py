#!/usr/bin/env python
"""Verify Phase 2 wiring: production code path under defaults and calibrated config.

Uses the ACTUAL production mass functions (late_interaction_to_mass,
combine_multiple from atelier.classify) on 5ef4868c's evidence.

Defaults must reproduce baseline; calibrated config must achieve
Phase 1's offline projection (~87% accuracy).
"""
from __future__ import annotations

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atelier.classify.mass_functions import late_interaction_to_mass
from atelier.classify.belief import (
    BeliefAssignment, FocalElement, FrameOfDiscernment, combine_multiple,
)
from atelier.optimize.svm.reflect import build_category_set


def alpha_scaled_singleton_assignment(
    masses_dict: dict[str, float], frame, alpha: float,
) -> BeliefAssignment:
    """Build BeliefAssignment from a {code: mass} dict, scaled by α.

    Mirrors what mass_functions._apply_alpha_scaling would do post-hoc
    on a channel's output. Used for SVM/CatBoost/LLM where we only
    have the post-mass dict from classifications.json (not the raw
    proba/scores), so we apply scaling at the dict level.
    """
    out: dict[FocalElement, float] = {}
    total = 0.0
    for raw, m in masses_dict.items():
        scaled = m * alpha
        code = raw.rstrip("*")
        is_sub = raw.endswith("*")
        if is_sub and code in frame.internal_nodes:
            fe = frame.internal_nodes[code]
        elif code in frame.singletons:
            fe = frame.singletons[code]
        else:
            continue
        out[fe] = out.get(fe, 0.0) + scaled
        total += scaled
    if total > 1.0:
        shrink = 1.0 / total
        for fe in out:
            out[fe] *= shrink
        total = 1.0
    out[frame.theta] = 1.0 - total
    return BeliefAssignment(masses=out)


def fuse(assignments):
    valid = [a for a in assignments if a and a.masses]
    if not valid:
        return None
    try:
        combined, _ = combine_multiple(valid, strategy="dempster")
        return combined
    except ValueError:
        return None


def pignistic_top1(assignment, frame):
    best = (None, -1.0)
    for code, sing in frame.singletons.items():
        p = assignment.pignistic_probability(sing)
        if p > best[1]:
            best = (code, p)
    return best[0]


def fallback_top1(es, excl):
    best = (None, -1.0)
    for chan, masses in es.items():
        if chan in excl:
            continue
        for code, m in (masses or {}).items():
            if m > best[1]:
                best = (code.rstrip("*"), m)
    return best[0]


def score(records, topk_data, frame, *,
          a_cos, a_svm, a_cb, a_llm,
          union_focal_k, union_focal_alpha):
    correct = 0
    fb = 0
    n = 0
    for c in records:
        ref = (c.get("reference_code") or "").strip()
        if not ref:
            continue
        es = c.get("evidence_sources", {})
        key = f"{c['table_name']}.{c['column_name']}"
        tk = topk_data.get(key, {}).get("top_k", [])
        scored_tags = [(t["code"], t["maxsim_score"]) for t in tk]
        assignments = []
        # Cosine: use production late_interaction_to_mass with the
        # wired calibration params.
        if scored_tags:
            cos_mass = late_interaction_to_mass(
                scored_tags, frame,
                alpha=a_cos,
                union_focal_k=union_focal_k,
                union_focal_alpha=union_focal_alpha,
            )
            assignments.append(cos_mass)
        # Other channels: scale post-mass evidence by α (we don't
        # have raw proba in classifications.json)
        for chan, alpha in [("svm", a_svm), ("catboost", a_cb), ("llm", a_llm)]:
            if alpha <= 0:
                continue
            if es.get(chan):
                assignments.append(alpha_scaled_singleton_assignment(
                    es[chan], frame, alpha))
        combined = fuse(assignments)
        if combined is None:
            pred = fallback_top1(es, {"cosine"} if a_llm > 0 else {"cosine", "llm"})
            fb += 1
        else:
            pred = pignistic_top1(combined, frame)
        if pred == ref:
            correct += 1
        n += 1
    return correct, n, fb


def main():
    cs = build_category_set()
    frame = FrameOfDiscernment(cs)
    records = json.loads(Path("build/results/5ef4868c/classifications.json").read_text())
    topk_data = json.loads(Path("build/runs/calibration/cosine_topk_5ef4868c.json").read_text())

    print(f"frame: {len(frame.singletons)} singletons, "
          f"{len(frame.internal_nodes)} internal nodes")
    print(f"records: {len(records)}  topk-rows: {len(topk_data)}\n")

    scenarios = [
        ("DEFAULT (α=1, K=0, all channels)",
         dict(a_cos=1.0, a_svm=1.0, a_cb=1.0, a_llm=1.0,
              union_focal_k=0, union_focal_alpha=0.45)),
        ("Phase 1 operating point (union K=3, no LLM)",
         dict(a_cos=1.0, a_svm=15.0, a_cb=0.7, a_llm=0.0,
              union_focal_k=3, union_focal_alpha=0.45)),
        ("Phase 1 operating point (union K=3, LLM α=0.1)",
         dict(a_cos=1.0, a_svm=15.0, a_cb=0.7, a_llm=0.1,
              union_focal_k=3, union_focal_alpha=0.45)),
        ("Pure no-LLM, no union (calibration only)",
         dict(a_cos=0.5, a_svm=15.0, a_cb=0.7, a_llm=0.0,
              union_focal_k=0, union_focal_alpha=0.45)),
    ]
    print(f"{'scenario':<55}  {'correct':>12}  {'acc':>7}  fallback")
    print("-" * 90)
    for label, params in scenarios:
        c, n, fb = score(records, topk_data, frame, **params)
        print(f"  {label:<53}  {c:>5d}/{n}  {100*c/n:5.2f}%  {fb} ({100*fb/n:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
