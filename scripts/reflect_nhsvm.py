#!/usr/bin/env python3
"""scripts/reflect_nhsvm.py — NHSVM capacity diagnostic.

Train-on-everything, predict-on-everything sanity check for NHSVM.  The
goal is to verify the algorithm is implemented in a non-degenerate way
and to surface the tunable surface where the shared-Frobenius-norm
regularization prevents fit.

This is a DIAGNOSTIC, not a production training run.  Best-practice
violations are intentional:
  - No held-out test set (we want to verify training-set fit ceiling)
  - Predict on the same rows used for training
  - Sweep configurations to find what knobs move the needle

For the proper train/val/test split + generalization measurement, see
`scripts/eval_nhsvm.py` and the future split-and-baseline experiment.

Outputs:
  build/reflect_nhsvm/report.md       — human-readable analysis
  build/reflect_nhsvm/results.json    — raw per-config numbers
  build/reflect_nhsvm/hive_cache.json — Hive samples (re-runs use cache;
                                        pass --refresh-cache to re-pull)

Usage:
  python scripts/reflect_nhsvm.py                    # baseline + sweep
  python scripts/reflect_nhsvm.py --refresh-cache    # re-pull from Hive
  python scripts/reflect_nhsvm.py --quick            # baseline only,
                                                      no sweep
  python scripts/reflect_nhsvm.py --no-calibration   # skip calibrated proba
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Heavy imports deferred until needed (numpy / sklearn / Hive)
log = logging.getLogger("reflect_nhsvm")

# Module-level paths — diagnostic-output specific (kept local).  The
# library-side HIVE_CACHE + AGENT_MEDIATED + Row + load_rows + etc. now
# live in atelier.optimize.svm.reflect.
REPORT_DIR = Path("build/reflect_nhsvm")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = REPORT_DIR / "results.json"
REPORT_MD = REPORT_DIR / "report.md"

# Re-export the public library API from atelier core.  Other scripts
# that still `from reflect_nhsvm import ...` continue to work via this
# re-export; the canonical location going forward is
# `atelier.optimize.svm.reflect`.
from atelier.optimize.svm.reflect import (  # noqa: E402
    AGENT_MEDIATED,
    HIVE_CACHE,
    Row,
    _pull_hive_table,  # used by svm_target_health.py; re-export until that script migrates too
    build_category_set,
    build_texts_and_labels,
    characterize_failures,
    load_rows,
)


# ──────────────────────────────────────────────────────────────────────
# Fit + predict
# ──────────────────────────────────────────────────────────────────────

def fit_and_predict(
    texts: list[str], labels: list[str], category_set,
    svc_C: float = 1.0,
    nhsvm_svd_components: int = 200,
    char_ngram_min: int = 3,
    char_ngram_max: int = 6,
    word_ngram_min: int = 1,
    word_ngram_max: int = 2,
    max_features: int = 50_000,
    skip_calibration: bool = False,
):
    """Train NHSVM on (texts, labels) and predict on the SAME texts.

    Returns (predicted_top1_per_row, fit_metadata).  If
    `skip_calibration=True`, bypasses CalibratedClassifierCV and uses
    LinearSVC.predict directly — cleanest measurement of the SVM's
    fit-to-train capacity.  If False, mirrors the production path
    (CalibratedClassifierCV with 5-fold CV → out-of-fold calibration).
    """
    from collections import Counter
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.decomposition import TruncatedSVD
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from atelier.classify.svm_classifier import HierarchicalFeatureExpander

    # Drop singletons (NHSVM's filter — they're guaranteed unfittable)
    counts = Counter(labels)
    keep_idx = [i for i, l in enumerate(labels) if counts[l] >= 2]
    dropped = len(labels) - len(keep_idx)
    if dropped:
        log.info("  dropped %d singleton-class rows", dropped)
    texts_k = [texts[i] for i in keep_idx]
    labels_k = [labels[i] for i in keep_idx]
    min_count = min(Counter(labels_k).values())

    # Step 1: TF-IDF
    feature_union = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(char_ngram_min, char_ngram_max),
            max_features=max_features, sublinear_tf=True,
        )),
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(word_ngram_min, word_ngram_max),
            max_features=max_features, sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )),
    ])
    X_tfidf = feature_union.fit_transform(texts_k)

    # Step 2: SVD reduction
    n_components = min(nhsvm_svd_components,
                       X_tfidf.shape[1] - 1, X_tfidf.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_reduced = svd.fit_transform(X_tfidf)

    # Step 3: NHSVM Kronecker expansion
    alphas = category_set.compute_nhsvm_alphas()
    expander = HierarchicalFeatureExpander.from_category_set(
        category_set, alphas, n_features_in=n_components,
    )
    X_expanded = expander.expand_with_labels(X_reduced, labels_k)

    # Step 4: LinearSVC + (optional) calibration
    svc = LinearSVC(C=svc_C, max_iter=10_000,
                    class_weight="balanced", dual="auto")
    if skip_calibration:
        svc.fit(X_expanded, labels_k)
        classes_ = list(svc.classes_)
        # Predict on training set via decision_function + argmax
        X_pred_expanded = expander.expand_universal(X_reduced)
        scores = svc.decision_function(X_pred_expanded)
        if scores.ndim == 1:
            pred_idx = (scores > 0).astype(int)
        else:
            pred_idx = scores.argmax(axis=1)
        pred_labels_k = [classes_[i] for i in pred_idx]
    else:
        calibrated = CalibratedClassifierCV(
            svc, cv=min(5, min_count), method="sigmoid", ensemble=False,
        )
        calibrated.fit(X_expanded, labels_k)
        classes_ = list(calibrated.classes_)
        X_pred_expanded = expander.expand_universal(X_reduced)
        proba = calibrated.predict_proba(X_pred_expanded)
        pred_idx = proba.argmax(axis=1)
        pred_labels_k = [classes_[i] for i in pred_idx]

    # Reassemble: keep_idx → keep_label; dropped rows get None
    pred_full: list[str | None] = [None] * len(labels)
    for ki, idx in enumerate(keep_idx):
        pred_full[idx] = pred_labels_k[ki]

    meta = {
        "n_rows_total": len(labels),
        "n_rows_kept": len(keep_idx),
        "n_singletons_dropped": dropped,
        "tfidf_dim": int(X_tfidf.shape[1]),
        "svd_components": int(n_components),
        "expanded_dim": int(X_expanded.shape[1]),
        "n_classes": len(classes_),
        "calibration": "skipped" if skip_calibration else "sigmoid_cv",
    }
    return pred_full, meta


def measure(predicted, labels):
    """Per-row correctness + headline accuracy."""
    n_total = len(labels)
    n_kept = sum(1 for p in predicted if p is not None)
    n_correct = sum(1 for p, l in zip(predicted, labels) if p == l)
    overall = n_correct / n_total if n_total else 0.0
    on_kept = n_correct / n_kept if n_kept else 0.0
    return {
        "n_total": n_total,
        "n_kept": n_kept,
        "n_correct": n_correct,
        "overall_top1": round(overall, 4),
        "top1_on_kept": round(on_kept, 4),
    }


# characterize_failures is now imported from atelier.optimize.svm.reflect
# (see the re-export block at the top of this file).


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def render_report(results: dict, rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# NHSVM capacity reflection")
    lines.append("")
    lines.append(f"_Generated from {len(rows)} rows in `{AGENT_MEDIATED}`._")
    lines.append("")
    lines.append("This is a **diagnostic**, not a generalization measurement.  "
                 "NHSVM is trained on the full reference and asked to predict "
                 "the same rows.  The question being answered: *can NHSVM "
                 "memorize what we showed it?*  If not, the implementation "
                 "has a structural problem.")
    lines.append("")

    # Headline summary
    lines.append("## Headline numbers")
    lines.append("")
    lines.append("| Configuration | Calibration | rows kept | top-1 on kept | top-1 overall |")
    lines.append("|---|---|---:|---:|---:|")
    for variant in results["variants"]:
        m = variant["measure"]
        meta = variant["meta"]
        lines.append(
            f"| {variant['name']} | {meta['calibration']} | "
            f"{m['n_kept']}/{m['n_total']} | "
            f"{m['top1_on_kept']:.4f} | {m['overall_top1']:.4f} |"
        )
    lines.append("")

    # Singleton context
    n_singletons = results.get("n_singletons", 0)
    lines.append(f"**Singleton drop**: {n_singletons} classes had only one "
                 f"example and were filtered by NHSVM's `_min_class_count() < 2` "
                 f"guard.  Theoretical ceiling on `overall_top1` is "
                 f"`(n_total - n_singletons) / n_total`.")
    lines.append("")

    # Failure mode breakdown
    baseline_failures = results["baseline_failures"]
    lines.append("## Baseline failure-mode breakdown")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, v in sorted(baseline_failures["failure_kinds"].items(),
                       key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Top failure examples
    lines.append("## Sample residual failures (baseline, top 20)")
    lines.append("")
    lines.append("| Row key | True | Predicted | Kind |")
    lines.append("|---|---|---|---|")
    for ex in baseline_failures["examples_top"][:20]:
        if ex["kind"] == "singleton_dropped":
            continue
        lines.append(
            f"| `{ex['key']}` | {ex.get('mnemonic_true','?')} ({ex['true']}) | "
            f"{ex['pred']} | {ex['kind']} |"
        )
    lines.append("")

    # Rescue-knob effect
    if len(results["variants"]) > 1:
        lines.append("## Rescue-knob ranking")
        lines.append("")
        baseline = results["variants"][0]["measure"]["top1_on_kept"]
        lines.append(f"Baseline top-1-on-kept: **{baseline:.4f}**")
        lines.append("")
        lines.append("| Variant | Δ top-1 on kept |")
        lines.append("|---|---:|")
        for v in results["variants"][1:]:
            delta = v["measure"]["top1_on_kept"] - baseline
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {v['name']} | {sign}{delta:.4f} |")
        lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    best = max(results["variants"], key=lambda v: v["measure"]["top1_on_kept"])
    ceiling = (len(rows) - n_singletons) / len(rows) if len(rows) else 0
    best_overall = best["measure"]["overall_top1"]
    if best["measure"]["top1_on_kept"] >= 0.99:
        verdict = ("**Non-degenerate.**  NHSVM achieves ≥99% top-1 on "
                   "non-singleton classes in the best configuration "
                   f"({best['name']}).  Implementation is sound.")
    elif best["measure"]["top1_on_kept"] >= 0.95:
        verdict = ("**Mostly non-degenerate.**  NHSVM achieves ≥95% but not "
                   "≥99% top-1 on kept classes.  Likely some structural "
                   "regularization headroom — check the failure-kind breakdown "
                   "for sibling-subtree confusions that resist capacity.")
    else:
        verdict = ("**Investigation warranted.**  NHSVM top-1 on kept is "
                   f"{best['measure']['top1_on_kept']:.4f} — meaningfully below "
                   "what a working implementation should achieve on predict-"
                   "on-train.  Check the residual examples + consider "
                   "the path-length normalization variant tracked in "
                   "memory: project_hsvm_iteration_directions.md.")
    lines.append(verdict)
    lines.append("")
    lines.append(f"Best variant: **{best['name']}**, top-1 on kept = "
                 f"{best['measure']['top1_on_kept']:.4f}, overall = "
                 f"{best_overall:.4f} (theoretical ceiling: {ceiling:.4f}).")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Variant definitions (the sweep)
# ──────────────────────────────────────────────────────────────────────

BASELINE = {"svc_C": 1.0, "nhsvm_svd_components": 200,
            "char_ngram_max": 6, "max_features": 50_000,
            "skip_calibration": False}

SWEEP_VARIANTS = [
    ("no_calibration", {**BASELINE, "skip_calibration": True}),
    ("highC", {**BASELINE, "svc_C": 100.0}),
    ("highSVD", {**BASELINE, "nhsvm_svd_components": 1000}),
    ("highC_highSVD", {**BASELINE, "svc_C": 100.0,
                       "nhsvm_svd_components": 1000}),
    ("wider_tfidf",
     {**BASELINE, "char_ngram_max": 8, "max_features": 200_000}),
    ("highC_highSVD_nocal",
     {**BASELINE, "svc_C": 100.0, "nhsvm_svd_components": 1000,
      "skip_calibration": True}),
]


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-pull Hive samples (otherwise uses cache)")
    ap.add_argument("--quick", action="store_true",
                    help="Baseline only — skip the sweep variants")
    ap.add_argument("--no-calibration-default", action="store_true",
                    help="Baseline runs without calibration "
                         "(faster + cleaner capacity test)")
    ap.add_argument("--database", default="reference_corpus",
                    help="Hive database containing the corpus tables")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    log.info("=== Phase 0: load reference + Hive samples ===")
    rows = load_rows(refresh_cache=args.refresh_cache, database=args.database)

    log.info("=== Build category set (full vocab) ===")
    cat_set = build_category_set()

    log.info("=== Build features (texts + labels) ===")
    texts, labels = build_texts_and_labels(rows)
    code_counts = Counter(labels)
    singletons = [c for c, n in code_counts.items() if n < 2]
    log.info("  %d singleton classes (will be dropped at NHSVM fit time)",
             len(singletons))

    # ── Baseline ──
    log.info("\n=== Phase 1: BASELINE fit + predict-on-train ===")
    baseline_cfg = {**BASELINE}
    if args.no_calibration_default:
        baseline_cfg["skip_calibration"] = True
    t0 = time.time()
    pred_baseline, meta_baseline = fit_and_predict(
        texts, labels, cat_set, **baseline_cfg)
    log.info("  fit + predict in %.1fs", time.time() - t0)
    m_baseline = measure(pred_baseline, labels)
    log.info("  top-1 overall=%.4f  on-kept=%.4f",
             m_baseline["overall_top1"], m_baseline["top1_on_kept"])

    failures_baseline = characterize_failures(
        rows, pred_baseline, labels, cat_set)
    log.info("  failure breakdown: %s", failures_baseline["failure_kinds"])

    variants = [{
        "name": "baseline",
        "config": baseline_cfg,
        "meta": meta_baseline,
        "measure": m_baseline,
    }]

    # ── Sweep ──
    if not args.quick:
        log.info("\n=== Phase 3: capacity sweep ===")
        for name, cfg in SWEEP_VARIANTS:
            log.info("  variant: %s  (%s)", name, cfg)
            t0 = time.time()
            try:
                pred_v, meta_v = fit_and_predict(texts, labels, cat_set, **cfg)
                m_v = measure(pred_v, labels)
                log.info("    %.1fs  top-1 overall=%.4f  on-kept=%.4f",
                         time.time() - t0, m_v["overall_top1"],
                         m_v["top1_on_kept"])
                variants.append({"name": name, "config": cfg,
                                 "meta": meta_v, "measure": m_v})
            except Exception as exc:
                log.error("    FAILED: %s", exc)
                variants.append({"name": name, "config": cfg,
                                 "meta": {"error": str(exc)},
                                 "measure": {"overall_top1": 0.0,
                                             "top1_on_kept": 0.0,
                                             "n_kept": 0, "n_total": len(labels),
                                             "n_correct": 0}})

    # ── Write results ──
    results = {
        "n_rows": len(rows),
        "n_singletons": len(singletons),
        "singleton_codes": sorted(singletons),
        "variants": variants,
        "baseline_failures": failures_baseline,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", RESULTS_JSON)

    report = render_report(results, rows)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    print(f"=== Best variant ===")
    best = max(variants, key=lambda v: v["measure"]["top1_on_kept"])
    print(f"  name: {best['name']}")
    print(f"  top-1 overall: {best['measure']['overall_top1']:.4f}")
    print(f"  top-1 on kept: {best['measure']['top1_on_kept']:.4f}")
    print(f"  theoretical ceiling: "
          f"{(len(rows)-len(singletons))/len(rows):.4f} "
          f"({len(singletons)} singletons dropped of {len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
