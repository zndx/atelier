#!/usr/bin/env python
"""Sweep cosine union-focal element across K ∈ {3, 5, 7, 10}.

Combines fresh top-10 Qdrant results (cosine_topk_5ef4868c.json) with
the original classifications.json evidence_sources for SVM+CatBoost+LLM,
re-fuses via Dempster, scores against reference. Tells us the right K
value to wire into Phase 2's union-focal mass function.
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from atelier.classify.belief import (
    BeliefAssignment, FocalElement, FrameOfDiscernment, combine_multiple,
)
from atelier.optimize.svm.reflect import build_category_set


def main() -> int:
    cs = build_category_set()
    frame = FrameOfDiscernment(cs)
    print(f"frame: {len(frame.singletons)} singletons, "
          f"{len(frame.internal_nodes)} internal nodes")

    cls = json.loads(
        Path("build/results/5ef4868c/classifications.json").read_text()
    )
    topk = json.loads(
        Path("build/runs/calibration/cosine_topk_5ef4868c.json").read_text()
    )
    by_key = {f"{c['table_name']}.{c['column_name']}": c
              for c in cls if c.get("reference_code")}
    print(f"classifications: {len(by_key)}  topk: {len(topk)}")

    def union_assignment(top_k_list, k, alpha):
        codes = [t["code"] for t in top_k_list[:k]
                 if t["code"] in frame.singletons
                 or t["code"] in frame.internal_nodes]
        if not codes or alpha <= 0:
            return None
        codes_fs = frozenset(codes)
        fe = None
        for node_fe in frame.internal_nodes.values():
            if node_fe.codes == codes_fs:
                fe = node_fe
                break
        if fe is None and len(codes_fs) == 1:
            fe = frame.singletons[next(iter(codes_fs))]
        if fe is None:
            fe = FocalElement(codes=codes_fs, label=f"|top{len(codes_fs)}|")
        return BeliefAssignment(masses={
            fe: alpha, frame.theta: 1.0 - alpha,
        })

    def singleton_assignment(masses_dict, alpha):
        out = {}
        total = 0.0
        for raw, m in masses_dict.items():
            m_s = m * alpha
            code = raw.rstrip("*")
            is_sub = raw.endswith("*")
            if is_sub and code in frame.internal_nodes:
                fe = frame.internal_nodes[code]
            elif code in frame.singletons:
                fe = frame.singletons[code]
            else:
                continue
            out[fe] = out.get(fe, 0.0) + m_s
            total += m_s
        total = min(total, 1.0)
        if 1.0 - total > 0:
            out[frame.theta] = out.get(frame.theta, 0.0) + (1.0 - total)
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

    def pignistic_top1(assignment):
        best = (None, -1.0)
        for code, sing in frame.singletons.items():
            p = assignment.pignistic_probability(sing)
            if p > best[1]:
                best = (code, p)
        return best[0]

    def fallback_top1(es, excl, topk_list):
        # Fall back to highest-mass non-excluded singleton, OR cosine top-1
        # if no other channel offers anything
        best = (None, -1.0)
        for chan, masses in es.items():
            if chan in excl:
                continue
            for code, m in (masses or {}).items():
                if m > best[1]:
                    best = (code.rstrip("*"), m)
        if best[0] is None and topk_list:
            return topk_list[0]["code"]
        return best[0]

    def score(*, k, alpha_cos_union, alpha_svm=15, alpha_cb=0.7,
              alpha_llm=None):
        exclude = {"maxsim"}
        if alpha_llm is None:
            exclude.add("llm")
        correct = 0
        n = 0
        fb = 0
        for key, c in by_key.items():
            ref = (c.get("reference_code") or "").strip()
            es = c.get("evidence_sources", {})
            tk_record = topk.get(key)
            top_k_list = tk_record["top_k"] if tk_record else []
            assignments = []
            for chan, alpha in [("svm", alpha_svm), ("catboost", alpha_cb)]:
                if es.get(chan):
                    assignments.append(singleton_assignment(es[chan], alpha))
            if alpha_llm is not None and es.get("llm"):
                assignments.append(singleton_assignment(es["llm"], alpha_llm))
            ua = union_assignment(top_k_list, k, alpha_cos_union)
            if ua is not None:
                assignments.append(ua)
            combined = fuse(assignments)
            if combined is None:
                pred = fallback_top1(es, exclude, top_k_list)
                fb += 1
            else:
                pred = pignistic_top1(combined)
            if pred == ref:
                correct += 1
            n += 1
        return correct, n, fb

    print(f"\n=== K sweep (no-LLM, α_svm=15, α_cb=0.7) ===")
    print(f"{'K':>3}  {'α_cos_u':>8}  {'correct':>12}  {'acc':>7}  fallback")
    for k in (3, 5, 7, 10):
        for au in (0.35, 0.45, 0.55, 0.65):
            c, n, fb = score(k=k, alpha_cos_union=au)
            star = "  ←" if c >= 1000 else ""
            print(f"  {k:>3}  {au:>8.2f}  {c:>5d}/{n}  {100*c/n:5.2f}%  "
                  f"{fb} ({100*fb/n:.1f}%){star}")

    # Also test with light LLM
    print(f"\n=== K sweep (α_llm=0.1, α_svm=15, α_cb=0.7) ===")
    print(f"{'K':>3}  {'α_cos_u':>8}  {'correct':>12}  {'acc':>7}  fallback")
    for k in (3, 5, 7, 10):
        for au in (0.35, 0.45, 0.55):
            c, n, fb = score(k=k, alpha_cos_union=au, alpha_llm=0.1)
            star = "  ←" if c >= 1000 else ""
            print(f"  {k:>3}  {au:>8.2f}  {c:>5d}/{n}  {100*c/n:5.2f}%  "
                  f"{fb} ({100*fb/n:.1f}%){star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
