"""Phase 1 — Offline calibration sweep.

Reads existing classifications.json, applies per-channel mass scaling,
re-fuses via Dempster's rule, scores against the curated reference.
No pipeline runs, no LLM calls, no model retraining.

The α values output here translate directly into runtime config knobs
(``classify.mass_calibration.{maxsim_alpha, svm_alpha, catboost_alpha}``)
applied inside ``mass_functions.py``.

Usage:
  python -m atelier.optimize.calibration.sweep \\
    build/results/5ef4868c/classifications.json
  python -m atelier.optimize.calibration.sweep \\
    build/results/5ef4868c/classifications.json --no-llm
  python -m atelier.optimize.calibration.sweep \\
    build/results/5ef4868c/classifications.json \\
    --alpha-svm 1,5,15,30,60 --no-llm
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

from atelier.classify.belief import (
    BeliefAssignment, FocalElement, FrameOfDiscernment, combine_multiple,
)


def scale_channel_mass(
    masses_dict: dict[str, float], alpha: float,
) -> dict[str, float]:
    """Scale a channel's non-Θ masses by ``alpha``.

    α < 1.0 → reduce confidence (more mass redistributed to Θ).
    α > 1.0 → expand confidence (less mass on Θ).  Total is clamped
    to 1.0 in the assignment builder to keep Θ non-negative.
    """
    if not masses_dict:
        return masses_dict
    return {code: m * alpha for code, m in masses_dict.items()}


def to_assignment(
    masses_dict: dict[str, float],
    frame: FrameOfDiscernment,
) -> BeliefAssignment:
    """Convert per-code mass dict to BeliefAssignment using the real frame.

    A code without ``*`` suffix maps to the singleton focal element
    ``{code}``.  A ``*``-suffix code (e.g. ``"0*"``) maps to the
    internal-node focal element ``{code} ∪ descendants``.  This is
    the same shape as production fusion in mass_functions.py.
    """
    out: dict[FocalElement, float] = {}
    total = 0.0
    for raw_code, m in masses_dict.items():
        is_subtree = raw_code.endswith("*")
        code = raw_code.rstrip("*")
        if is_subtree and code in frame.internal_nodes:
            fe = frame.internal_nodes[code]
        elif code in frame.singletons:
            fe = frame.singletons[code]
        else:
            # Code not in current vocab (vocab drift); skip rather
            # than fabricate a frame element.
            continue
        out[fe] = out.get(fe, 0.0) + float(m)
        total += float(m)
    total = min(total, 1.0)
    theta_mass = max(0.0, 1.0 - total)
    if theta_mass > 0:
        out[frame.theta] = out.get(frame.theta, 0.0) + theta_mass
    return BeliefAssignment(masses=out)


def fuse(
    channel_masses: list[dict[str, float]],
    frame: FrameOfDiscernment,
) -> BeliefAssignment | None:
    """Dempster-combine a list of per-channel mass dicts."""
    assignments = [to_assignment(m, frame) for m in channel_masses if m]
    assignments = [a for a in assignments if a.masses]
    if not assignments:
        return None
    try:
        combined, _ = combine_multiple(assignments, strategy="dempster")
    except ValueError:
        return None
    return combined


def top_code(assignment: BeliefAssignment, frame: FrameOfDiscernment) -> str | None:
    """Pignistic top1: pick the code with highest pignistic probability.

    Mirrors production's selection — distributes subtree mass equally
    across descendants when scoring per-code commitment.
    """
    best: tuple[str | None, float] = (None, -1.0)
    for code, singleton in frame.singletons.items():
        prob = assignment.pignistic_probability(singleton)
        if prob > best[1]:
            best = (code, prob)
    return best[0]


def _fallback_top1(
    es: dict[str, dict[str, float]],
    exclude_channels: set[str],
) -> str | None:
    """When fusion fails (K=1.0), pick the channel-top1 with highest mass.

    Mirrors production's plausibility-fallback behavior: rather than
    leaving the column unclassified, commit to the loudest single
    channel's top suggestion.
    """
    best: tuple[str | None, float] = (None, -1.0)
    for chan, masses in es.items():
        if not masses or chan in exclude_channels:
            continue
        for code, m in masses.items():
            if m > best[1]:
                best = (code.rstrip("*"), m)
    return best[0]


def score_config(
    records: list[dict],
    alphas: dict[str, float],
    frame: FrameOfDiscernment,
    *,
    include_llm: bool = True,
    exclude_channels: set[str] | None = None,
) -> tuple[int, int, int]:
    """Apply α scaling, re-fuse, score top1 vs reference.

    Returns ``(correct, n_total, n_fallback)``.  ``n_fallback`` counts
    columns where Dempster fusion failed (K=1.0) and we resorted to
    the loudest-channel-top1 fallback.
    """
    exclude_channels = exclude_channels or set()
    if not include_llm:
        exclude_channels = exclude_channels | {"llm"}
    correct = 0
    n = 0
    n_fallback = 0
    for c in records:
        ref = (c.get("reference_code") or "").strip()
        if not ref:
            continue
        es = c.get("evidence_sources", {})
        scaled = []
        for chan, masses in es.items():
            if not masses or chan in exclude_channels:
                continue
            alpha = alphas.get(chan, 1.0)
            scaled.append(scale_channel_mass(masses, alpha))
        combined = fuse(scaled, frame)
        if combined is None:
            pred = _fallback_top1(es, exclude_channels)
            n_fallback += 1
        else:
            pred = top_code(combined, frame)
        if pred == ref:
            correct += 1
        n += 1
    return correct, n, n_fallback


def _parse_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("classifications_path", type=Path,
                   help="Path to classifications.json from a prior run")
    p.add_argument("--no-llm", action="store_true",
                   help="Exclude LLM from fusion (LLM-as-Appellate scenario)")
    p.add_argument("--exclude-channels", default="",
                   help="Additional channels to drop (comma-separated)")
    p.add_argument("--alpha-maxsim", type=str, default="0.3,0.5,0.7,1.0")
    p.add_argument("--alpha-svm", type=str, default="1.0,5.0,15.0,30.0")
    p.add_argument("--alpha-catboost", type=str, default="0.7,1.0,1.3,1.6")
    p.add_argument("--alpha-llm", type=str, default="1.0",
                   help="Comma-separated α values for LLM (default 1.0 = unchanged)")
    p.add_argument("--top-n", type=int, default=15,
                   help="Show top N configs in output (default 15)")
    args = p.parse_args()

    records = json.loads(args.classifications_path.read_text())
    n_scored = sum(1 for c in records if c.get("reference_code"))
    extra_exclude = {c.strip() for c in args.exclude_channels.split(",")
                     if c.strip()}
    mode = "no-LLM" if args.no_llm else "with-LLM"
    excl = f" (exclude: {sorted(extra_exclude)})" if extra_exclude else ""
    print(f"loaded {len(records)} records ({n_scored} with reference)")
    print(f"mode: {mode}{excl}\n")

    # Build frame from the live vocabulary (default.annotations)
    from atelier.optimize.svm.reflect import build_category_set
    cs = build_category_set()
    frame = FrameOfDiscernment(cs)
    print(f"frame: {len(frame.singletons)} singletons, "
          f"{len(frame.internal_nodes)} internal nodes\n")

    # Baselines for context
    baseline_correct, baseline_n, baseline_fb = score_config(
        records, {}, frame, include_llm=not args.no_llm,
        exclude_channels=extra_exclude,
    )
    print(f"baseline (α=1.0 everywhere): "
          f"{baseline_correct}/{baseline_n} = "
          f"{100*baseline_correct/baseline_n:.2f}%  "
          f"(fallback on {baseline_fb} K=1 columns)")

    # Sweep
    a_max = _parse_floats(args.alpha_maxsim)
    a_svm = _parse_floats(args.alpha_svm)
    a_cat = _parse_floats(args.alpha_catboost)
    a_llm = _parse_floats(args.alpha_llm)
    n_combos = len(a_max) * len(a_svm) * len(a_cat) * len(a_llm)
    print(f"\nsweeping {len(a_max)}×{len(a_svm)}×{len(a_cat)}×{len(a_llm)}"
          f" = {n_combos} combos...\n")

    results = []
    for ac, av, ab, al in product(a_max, a_svm, a_cat, a_llm):
        alphas = {"maxsim": ac, "svm": av, "catboost": ab, "llm": al}
        correct, n, fb = score_config(
            records, alphas, frame, include_llm=not args.no_llm,
            exclude_channels=extra_exclude,
        )
        results.append((correct / n, ac, av, ab, al, correct, n, fb))

    results.sort(reverse=True)
    print(f"=== TOP {args.top_n} ===")
    print(f"{'acc':>8}  {'α_max':>6} {'α_svm':>6} {'α_cb':>6} {'α_llm':>6}  "
          f"{'correct':>12}  {'fallback':>10}  Δ vs base")
    for acc, ac, av, ab, al, correct, n, fb in results[:args.top_n]:
        delta = (correct - baseline_correct) / baseline_n
        print(f"  {100*acc:6.2f}%  {ac:6.2f} {av:6.2f} {ab:6.2f} {al:6.2f}  "
              f"{correct:5d}/{n}  {fb:5d} ({100*fb/n:4.1f}%)  "
              f"{100*delta:+6.2f}pp")

    best = results[0]
    print(f"\nbest: α_maxsim={best[1]} α_svm={best[2]} "
          f"α_catboost={best[3]} α_llm={best[4]}  →  {100*best[0]:.2f}% "
          f"({100*(best[5]-baseline_correct)/baseline_n:+.2f}pp), "
          f"fallback on {best[7]}/{best[6]} columns "
          f"({100*best[7]/best[6]:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
