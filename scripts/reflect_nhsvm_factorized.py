#!/usr/bin/env python3
"""scripts/reflect_nhsvm_factorized.py — FactorizedNHSVM capacity diagnostic.

A/B counterpart to ``reflect_nhsvm_modernbert.py``, but uses the
**factorized** NHSVM head (Choi et al. 2015's efficient implementation)
instead of the explicit Kronecker expansion.  Same 1149-row reference,
same ModernBERT-base encoder (cached), same fit-on-train methodology.

Motivation: the explicit Kronecker expansion (HierarchicalFeatureExpander)
catastrophically fails on dense embeddings — TF-IDF best fit-on-train =
98.93% top-1, ModernBERT best = 4.26%.  The paper's factorized form
(one weight vector per node + path summation, no Kronecker product
materialized) is mathematically equivalent to the explicit form for
sparse features but works correctly with dense pretrained embeddings.

Two grid axes:
  text_shape: svm-text vs rich-text (matches modernbert script)
  head_config: epochs, lr, weight_decay (the structured-SVM training knobs)

Outputs:
  build/reflect_nhsvm_factorized/report.md
  build/reflect_nhsvm_factorized/results.json
  build/reflect_nhsvm_factorized/embeddings.npz (shared cache name; reuses
                                                  the modernbert cache if
                                                  present)

Usage:
  python scripts/reflect_nhsvm_factorized.py                # baseline + sweep
  python scripts/reflect_nhsvm_factorized.py --quick        # baseline only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reflect_nhsvm import (
    Row,
    load_rows,
    build_texts_and_labels,
    build_category_set,
    measure,
    characterize_failures,
    AGENT_MEDIATED,
)
from reflect_nhsvm_modernbert import (
    build_rich_texts_and_labels,
    load_or_encode_embeddings,
    MODEL_ID,
    EMB_DIM,
)
from atelier.classify.factorized_nhsvm import (
    FactorizedNHSVMHead,
    fit_factorized_nhsvm,
    TrainResult,
)

log = logging.getLogger("reflect_nhsvm_factorized")

REPORT_DIR = Path("build/reflect_nhsvm_factorized")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = REPORT_DIR / "results.json"
REPORT_MD = REPORT_DIR / "report.md"


# ──────────────────────────────────────────────────────────────────────
# Fit + predict using the factorized head
# ──────────────────────────────────────────────────────────────────────

def fit_and_predict_factorized(
    X,
    labels: list[str],
    category_set,
    *,
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    """Train factorized NHSVM head; predict on the same rows."""
    import numpy as np
    # NHSVM still drops singletons at the structural margin level:
    # classes with a single example can't have a meaningful margin.
    counts = Counter(labels)
    keep_idx = [i for i, l in enumerate(labels) if counts[l] >= 2]
    dropped = len(labels) - len(keep_idx)
    if dropped:
        log.info("  dropped %d singleton-class rows", dropped)
    X_k = X[keep_idx]
    labels_k = [labels[i] for i in keep_idx]

    head, train_result = fit_factorized_nhsvm(
        X_k, labels_k, category_set,
        epochs=epochs, batch_size=batch_size,
        lr=lr, weight_decay=weight_decay,
    )

    # Predict on all rows (including singletons — model can still
    # output a prediction; we just expect it to be wrong for them).
    import torch
    from sklearn.preprocessing import normalize as sk_normalize
    X_all = sk_normalize(X, norm="l2", axis=1)
    device = next(head.parameters()).device
    X_t = torch.tensor(X_all.astype(np.float32), device=device)
    pred_codes = head.predict_codes(X_t)

    # Reassemble: singletons are not "kept" for accuracy measurement
    pred_full: list[str | None] = list(pred_codes)
    kept_mask = set(keep_idx)
    for i in range(len(labels)):
        if i not in kept_mask:
            pred_full[i] = None

    meta = {
        "n_rows_total": len(labels),
        "n_rows_kept": len(keep_idx),
        "n_singletons_dropped": dropped,
        "encoder": MODEL_ID,
        "encoder_dim": int(X.shape[1]),
        "head": "factorized_nhsvm",
        "n_nodes": head.n_nodes,
        "head_params": head.num_params(),
        "epochs": train_result.epochs_run,
        "final_train_loss": train_result.final_train_loss,
        "final_train_acc": train_result.final_train_acc,
        "elapsed_sec": round(train_result.elapsed_sec, 2),
    }
    return pred_full, meta


# ──────────────────────────────────────────────────────────────────────
# Sweep
# ──────────────────────────────────────────────────────────────────────

BASELINE = {
    "epochs": 300,
    "batch_size": 64,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "pooling": "mean",
}

KNOB_VARIANTS = [
    ("baseline", BASELINE),
    ("longer_train", {**BASELINE, "epochs": 600}),
    ("higher_lr", {**BASELINE, "lr": 5e-3}),
    ("lower_wd", {**BASELINE, "weight_decay": 1e-5}),
    ("cls_pool", {**BASELINE, "pooling": "cls"}),
]

TEXT_SHAPES = ["svm", "rich"]


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def render_report(results: dict, rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# Factorized NHSVM capacity reflection")
    lines.append("")
    lines.append(f"_Generated from {len(rows)} rows in `{AGENT_MEDIATED}`._")
    lines.append("")
    lines.append(
        "**Architectural change vs `reflect_nhsvm_modernbert.py`**: this "
        "diagnostic uses the *factorized* NHSVM head (Choi 2015's "
        "efficient form: per-node weight vectors + path summation) rather "
        "than the explicit Kronecker expansion.  The explicit form "
        "catastrophically fails on dense pretrained embeddings "
        "(4.26% best vs TF-IDF's 98.93%); the factorized form is "
        "mathematically equivalent for sparse features but works "
        "correctly with dense ones."
    )
    lines.append("")

    lines.append("## Grid: text_shape × head_config")
    lines.append("")
    lines.append(
        "| text_shape | knob | pooling | epochs | lr | weight_decay | "
        "fit-time | top-1 on kept | top-1 overall |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for v in results["variants"]:
        m = v["measure"]
        meta = v["meta"]
        cfg = v["config"]
        elapsed = meta.get("elapsed_sec", 0.0)
        lines.append(
            f"| {v.get('text_shape', '?')} | {v.get('knob', v['name'])} | "
            f"{cfg.get('pooling', 'mean')} | {cfg['epochs']} | "
            f"{cfg['lr']:.0e} | {cfg['weight_decay']:.0e} | "
            f"{elapsed:.1f}s | "
            f"{m['top1_on_kept']:.4f} | {m['overall_top1']:.4f} |"
        )
    lines.append("")

    n_singletons = results.get("n_singletons", 0)
    ceiling = (len(rows) - n_singletons) / len(rows) if len(rows) else 0
    lines.append(
        f"**Singleton drop**: {n_singletons} classes had a single example "
        f"and can't have a meaningful margin (theoretical ceiling on "
        f"`overall_top1`: {ceiling:.4f})."
    )
    lines.append("")

    # Best per text shape
    by_shape: dict[str, list[dict]] = {}
    for v in results["variants"]:
        by_shape.setdefault(v.get("text_shape", "?"), []).append(v)
    lines.append("### Best per text shape")
    lines.append("")
    lines.append("| text_shape | best knob | top-1 on kept | top-1 overall |")
    lines.append("|---|---|---:|---:|")
    for shape, vs in by_shape.items():
        best = max(vs, key=lambda v: v["measure"]["top1_on_kept"])
        lines.append(
            f"| {shape} | {best.get('knob', '?')} | "
            f"{best['measure']['top1_on_kept']:.4f} | "
            f"{best['measure']['overall_top1']:.4f} |"
        )
    lines.append("")

    # Failure breakdown for the anchor cell
    bf = results.get("baseline_failures", {})
    if bf.get("failure_kinds"):
        lines.append("## svm/baseline failure-mode breakdown")
        lines.append("")
        lines.append("| Failure kind | Count |")
        lines.append("|---|---:|")
        for k, v in sorted(bf["failure_kinds"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("### Sample residual failures (top 20)")
        lines.append("")
        lines.append("| Row key | True | Predicted | Kind |")
        lines.append("|---|---|---|---|")
        for ex in bf["examples_top"][:20]:
            if ex["kind"] == "singleton_dropped":
                continue
            lines.append(
                f"| `{ex['key']}` | {ex.get('mnemonic_true','?')} "
                f"({ex['true']}) | {ex['pred']} | {ex['kind']} |"
            )
        lines.append("")

    # Comparison anchors
    lines.append("## Comparison anchors (read this side-by-side with other reports)")
    lines.append("")
    lines.append(
        "| Source | encoder | head | best top-1 on kept |"
    )
    lines.append("|---|---|---|---:|")
    tfidf_ref = results.get("tfidf_reference_top1_kept")
    mb_kron_ref = results.get("modernbert_kronecker_top1_kept")
    if tfidf_ref is not None:
        lines.append(f"| reflect_nhsvm.py | TF-IDF char+word | Kronecker-NHSVM | {tfidf_ref:.4f} |")
    if mb_kron_ref is not None:
        lines.append(f"| reflect_nhsvm_modernbert.py | ModernBERT-base | Kronecker-NHSVM | {mb_kron_ref:.4f} |")
    best = max(results["variants"], key=lambda v: v["measure"]["top1_on_kept"])
    lines.append(
        f"| **this report** | ModernBERT-base | **factorized-NHSVM** | "
        f"**{best['measure']['top1_on_kept']:.4f}** |"
    )
    lines.append("")

    return "\n".join(lines)


def _read_reference_top1(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return max(v["measure"]["top1_on_kept"] for v in data["variants"])
    except Exception as exc:
        log.warning("  could not parse %s: %s", path, exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-pull Hive samples (otherwise uses reflect_nhsvm cache)")
    ap.add_argument("--refresh-embeddings", action="store_true",
                    help="Re-encode with ModernBERT (otherwise uses cache)")
    ap.add_argument("--quick", action="store_true",
                    help="Baseline only — skip the sweep variants")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Encoder batch size (lower if A10G OOMs; default 64)")
    ap.add_argument("--database", default="reference_corpus",
                    help="Hive database containing the corpus tables")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log.info("=== Phase 0: load reference + Hive samples ===")
    rows = load_rows(refresh_cache=args.refresh_cache, database=args.database)

    log.info("=== Build category set (full vocab) ===")
    cat_set = build_category_set()

    log.info("=== Build texts + labels for each text shape ===")
    text_sets: dict[str, list[str]] = {}
    labels: list[str] | None = None
    for shape in TEXT_SHAPES:
        if shape == "svm":
            texts_s, labels_s = build_texts_and_labels(rows)
        elif shape == "rich":
            texts_s, labels_s = build_rich_texts_and_labels(rows)
        else:
            raise ValueError(f"unknown text shape: {shape}")
        text_sets[shape] = texts_s
        if labels is None:
            labels = labels_s
        else:
            assert labels_s == labels, f"label mismatch: {shape}"
        avg_len = sum(len(t) for t in texts_s) / max(len(texts_s), 1)
        log.info("  text_shape=%-4s  rows=%d  avg_chars=%.0f",
                 shape, len(texts_s), avg_len)
    code_counts = Counter(labels)
    singletons = [c for c, n in code_counts.items() if n < 2]
    log.info("  %d singleton classes", len(singletons))

    # Encode each (text_shape, pooling).  Reuses the modernbert cache
    # by path: load_or_encode_embeddings writes to build/reflect_nhsvm_modernbert/.
    log.info("\n=== Phase 0b: ModernBERT encode (per text_shape × pooling) ===")
    poolings_needed: set[str] = {"mean"}
    for _, cfg in KNOB_VARIANTS:
        poolings_needed.add(cfg.get("pooling", "mean"))

    encodings: dict[tuple[str, str], "np.ndarray"] = {}  # type: ignore
    for shape in TEXT_SHAPES:
        for pool in poolings_needed:
            log.info("  text_shape=%s  pooling=%s", shape, pool)
            encodings[(shape, pool)] = load_or_encode_embeddings(
                text_sets[shape], refresh=args.refresh_embeddings,
                batch_size=args.batch_size, pooling=pool,
            )

    # Grid sweep
    log.info("\n=== Phase 1+2: grid sweep (text_shape × head_config) ===")
    variants: list[dict] = []
    failures_baseline: dict | None = None

    knobs_to_run = [KNOB_VARIANTS[0]] if args.quick else KNOB_VARIANTS

    def _persist_partial():
        RESULTS_JSON.write_text(json.dumps({
            "n_rows": len(rows),
            "n_singletons": len(singletons),
            "singleton_codes": sorted(singletons),
            "variants": variants,
            "baseline_failures": failures_baseline or {
                "failure_kinds": {}, "examples_top": [], "total_failures": 0,
            },
            "incomplete": True,
        }, indent=2))

    for shape in TEXT_SHAPES:
        for name, cfg in knobs_to_run:
            full_name = f"{shape}/{name}"
            log.info("  variant: %s  (%s)", full_name, cfg)
            t0 = time.time()
            try:
                pool = cfg.get("pooling", "mean")
                X_in = encodings[(shape, pool)]
                pred_v, meta_v = fit_and_predict_factorized(
                    X_in, labels, cat_set,
                    **{k: v for k, v in cfg.items() if k != "pooling"},
                )
                m_v = measure(pred_v, labels)
                log.info(
                    "    %.1fs  top-1 overall=%.4f  on-kept=%.4f",
                    time.time() - t0, m_v["overall_top1"], m_v["top1_on_kept"],
                )
                if shape == TEXT_SHAPES[0] and name == "baseline":
                    failures_baseline = characterize_failures(
                        rows, pred_v, labels, cat_set,
                    )
                    log.info("    failure breakdown: %s",
                             failures_baseline["failure_kinds"])
                variants.append({
                    "name": full_name,
                    "text_shape": shape,
                    "knob": name,
                    "config": cfg,
                    "meta": meta_v,
                    "measure": m_v,
                })
                _persist_partial()
            except Exception as exc:
                log.error("    FAILED: %s", exc)
                variants.append({
                    "name": full_name,
                    "text_shape": shape,
                    "knob": name,
                    "config": cfg,
                    "meta": {"error": str(exc)},
                    "measure": {
                        "overall_top1": 0.0, "top1_on_kept": 0.0,
                        "n_kept": 0, "n_total": len(labels), "n_correct": 0,
                    },
                })
                _persist_partial()

    if failures_baseline is None:
        failures_baseline = {"failure_kinds": {}, "examples_top": [],
                              "total_failures": 0}

    # Comparison anchors
    tfidf_ref = _read_reference_top1(Path("build/reflect_nhsvm/results.json"))
    mb_kron_ref = _read_reference_top1(
        Path("build/reflect_nhsvm_modernbert/results.json")
    )

    results = {
        "n_rows": len(rows),
        "n_singletons": len(singletons),
        "singleton_codes": sorted(singletons),
        "variants": variants,
        "baseline_failures": failures_baseline,
        "tfidf_reference_top1_kept": tfidf_ref,
        "modernbert_kronecker_top1_kept": mb_kron_ref,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", RESULTS_JSON)

    report = render_report(results, rows)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    print("=== Best variant ===")
    best = max(variants, key=lambda v: v["measure"]["top1_on_kept"])
    print(f"  name: {best['name']}")
    print(f"  top-1 overall: {best['measure']['overall_top1']:.4f}")
    print(f"  top-1 on kept: {best['measure']['top1_on_kept']:.4f}")
    ceiling = (len(rows)-len(singletons))/len(rows)
    print(f"  theoretical ceiling: {ceiling:.4f}")
    if tfidf_ref is not None:
        print(f"  TF-IDF reference: {tfidf_ref:.4f}")
    if mb_kron_ref is not None:
        print(f"  ModernBERT-Kronecker reference: {mb_kron_ref:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
