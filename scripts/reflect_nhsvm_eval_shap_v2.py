#!/usr/bin/env python3
"""scripts/reflect_nhsvm_eval_shap_v2.py — Phase D of the corpus expansion plan.

Long-run evaluation of the factorized NHSVM head trained on
``corpus_v2`` synth rows + the 855 original real training rows, evaluated
against the SAME 271 held-out real test rows the 61.25% baseline used.

Pins the byte-identical 855/271 split via ``stratified_split(..., seed=42)``
(reused from ``scripts/reflect_nhsvm_eval_shap.py``).  Per-category
accuracy delta + comparison anchors land in the report.

Outputs:
  build/reflect_nhsvm_eval_shap_v2/report.md
  build/reflect_nhsvm_eval_shap_v2/results.json
  build/reflect_nhsvm_eval_shap_v2/per_category_accuracy.json
  build/reflect_nhsvm_eval_shap_v2/run.log
  build/reflect_nhsvm_eval_shap_v2/embeddings_cache.npz (encoder output)

Usage:
  python scripts/reflect_nhsvm_eval_shap_v2.py
  python scripts/reflect_nhsvm_eval_shap_v2.py --refresh-embeddings
  python scripts/reflect_nhsvm_eval_shap_v2.py --rows-per-node-sanity 50
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch

from reflect_nhsvm import (
    Row,
    load_rows,
    build_texts_and_labels,
    build_category_set,
    characterize_failures,
    AGENT_MEDIATED,
)
from reflect_nhsvm_modernbert import (
    encode_modernbert,
    MODEL_ID,
    EMB_DIM,
)
from reflect_nhsvm_eval_shap import (
    stratified_split,
    BEST_KNOB,
)
from atelier.classify.factorized_nhsvm import fit_factorized_nhsvm

log = logging.getLogger("reflect_nhsvm_eval_shap_v2")

REPORT_DIR = Path("build/reflect_nhsvm_eval_shap_v2")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = REPORT_DIR / "results.json"
REPORT_MD = REPORT_DIR / "report.md"
PER_CAT_JSON = REPORT_DIR / "per_category_accuracy.json"
RUN_LOG = REPORT_DIR / "run.log"
EMBED_CACHE = REPORT_DIR / "embeddings_cache.npz"

SYNTH_ROWS_PATH = Path("build/data/svm_training/corpus_v2/synth_rows.jsonl")
MANIFEST_PATH = Path("build/data/svm_training/corpus_v2/manifest.json")


# ──────────────────────────────────────────────────────────────────────
# Load synth corpus + reconstruct as Row objects
# ──────────────────────────────────────────────────────────────────────

def load_synth_rows() -> list[Row]:
    """Load corpus_v2 synth_rows.jsonl, instantiate as Row dataclasses."""
    if not SYNTH_ROWS_PATH.exists():
        raise FileNotFoundError(
            f"Synth corpus missing at {SYNTH_ROWS_PATH}. "
            f"Run scripts/generate_corpus_v2.py first."
        )
    rows: list[Row] = []
    with SYNTH_ROWS_PATH.open() as f:
        for line in f:
            d = json.loads(line)
            rows.append(Row(
                table=d["table"],
                column=d["column"],
                column_type=d["column_type"],
                sample_values=d["sample_values"],
                siblings_full=d["siblings_full"],
                mnemonic=d["mnemonic"],
                code=d["code"],
            ))
    log.info("  loaded %d synth rows from %s", len(rows), SYNTH_ROWS_PATH)
    return rows


def corpus_hash() -> str:
    """Hash of the synth corpus file for cache invalidation."""
    if not SYNTH_ROWS_PATH.exists():
        return "none"
    h = hashlib.sha1()
    with SYNTH_ROWS_PATH.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


# ──────────────────────────────────────────────────────────────────────
# Encoding with cache (keyed on text hash)
# ──────────────────────────────────────────────────────────────────────

def encode_with_cache(
    texts: list[str], cache_key: str,
    *, refresh: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Encode texts with ModernBERT, cache by (text_hash, key)."""
    text_hash = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()[:12]
    full_key = f"{cache_key}_{text_hash}"

    if EMBED_CACHE.exists() and not refresh:
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        if full_key in cache.files:
            log.info("  cache HIT %s  shape=%s", full_key, cache[full_key].shape)
            return cache[full_key]

    log.info("  cache MISS %s — encoding %d texts...", full_key, len(texts))
    t0 = time.time()
    embeddings = encode_modernbert(texts, batch_size=batch_size, pooling="mean")
    elapsed = time.time() - t0
    log.info("  encoded %d in %.1fs (%.0f rows/s)", len(texts), elapsed,
             len(texts) / max(elapsed, 1e-6))

    existing: dict = {}
    if EMBED_CACHE.exists():
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        existing = {k: cache[k] for k in cache.files}
    existing[full_key] = embeddings
    np.savez_compressed(EMBED_CACHE, **existing)
    return embeddings


# ──────────────────────────────────────────────────────────────────────
# Per-category eval breakdown
# ──────────────────────────────────────────────────────────────────────

def per_category_accuracy(
    pred: list[str], true: list[str],
) -> dict[str, dict]:
    """Per-class accuracy + n_test."""
    by_class: dict[str, list[bool]] = defaultdict(list)
    for p, t in zip(pred, true):
        by_class[t].append(p == t)
    result = {}
    for cls, hits in by_class.items():
        result[cls] = {
            "n_test": len(hits),
            "n_correct": sum(hits),
            "accuracy": round(sum(hits) / len(hits), 4) if hits else 0.0,
        }
    return result


# ──────────────────────────────────────────────────────────────────────
# Phase D core
# ──────────────────────────────────────────────────────────────────────

def run_eval(
    *,
    refresh_embeddings: bool = False,
    batch_size: int = 64,
    knob: dict | None = None,
    rows_per_node_sanity: int | None = None,
) -> dict:
    knob = knob or BEST_KNOB
    log.info("=== Phase D: long-run eval (synth + real → real held-out) ===")
    log.info("Loading reference rows...")
    real_rows = load_rows(refresh_cache=False, database="reference_corpus")
    _, real_labels = build_texts_and_labels(real_rows)

    # Filter to trainable subset (drop singletons)
    code_counts = Counter(real_labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    trainable_mask = [l not in singletons for l in real_labels]
    trainable_real_rows = [r for r, m in zip(real_rows, trainable_mask) if m]
    trainable_real_labels = [l for l, m in zip(real_labels, trainable_mask) if m]
    log.info("  real trainable rows: %d / %d (%d singletons dropped)",
             len(trainable_real_rows), len(real_rows), len(singletons))

    # Pin SAME train/test split as the 61.25% baseline
    train_idx, test_idx = stratified_split(
        trainable_real_labels, test_size=0.2, seed=42,
    )
    real_train_rows = [trainable_real_rows[i] for i in train_idx]
    real_train_labels = [trainable_real_labels[i] for i in train_idx]
    real_test_rows = [trainable_real_rows[i] for i in test_idx]
    real_test_labels = [trainable_real_labels[i] for i in test_idx]
    log.info("  pinned split: train=%d  test=%d  test_classes=%d",
             len(train_idx), len(test_idx), len(set(real_test_labels)))

    # Load synth corpus
    log.info("Loading synth corpus...")
    synth_rows = load_synth_rows()
    if rows_per_node_sanity is not None:
        # Sanity-mode: cap per-node to a smaller number for quick iteration
        by_code: dict[str, list[Row]] = defaultdict(list)
        for r in synth_rows:
            if len(by_code[r.code]) < rows_per_node_sanity:
                by_code[r.code].append(r)
        synth_rows = [r for rows_for_code in by_code.values() for r in rows_for_code]
        log.info("  sanity mode: capped to %d rows total", len(synth_rows))
    synth_labels = [r.code for r in synth_rows]

    # Augmented training set = synth + real_train
    aug_rows = synth_rows + real_train_rows
    aug_labels = synth_labels + real_train_labels
    log.info("  augmented train set: %d synth + %d real = %d total",
             len(synth_rows), len(real_train_rows), len(aug_rows))

    # Drop classes in augmented train with <2 examples (NHSVM constraint)
    aug_counts = Counter(aug_labels)
    aug_keep = [i for i, l in enumerate(aug_labels) if aug_counts[l] >= 2]
    aug_dropped = len(aug_labels) - len(aug_keep)
    if aug_dropped:
        log.info("  dropped %d aug-train rows with singleton classes", aug_dropped)
    aug_rows_kept = [aug_rows[i] for i in aug_keep]
    aug_labels_kept = [aug_labels[i] for i in aug_keep]

    # Build texts via the lean svm-text shape (per SHAP findings: lean > rich)
    aug_texts, _ = build_texts_and_labels(aug_rows_kept)
    log.info("  built %d aug-train texts", len(aug_texts))

    test_texts, _ = build_texts_and_labels(real_test_rows)
    log.info("  built %d test texts", len(test_texts))

    # Encode (cached)
    log.info("Encoding aug-train texts with ModernBERT...")
    X_train = encode_with_cache(
        aug_texts, cache_key=f"aug-train-{corpus_hash()}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("Encoding test texts with ModernBERT...")
    X_test = encode_with_cache(
        test_texts, cache_key="real-test",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  X_train shape=%s  X_test shape=%s", X_train.shape, X_test.shape)

    # Train factorized head
    log.info("Training factorized NHSVM head...")
    cat_set = build_category_set()
    train_t0 = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, aug_labels_kept, cat_set,
        **knob, verbose=True, eval_every=50,
    )
    train_elapsed = time.time() - train_t0
    log.info("  train fit-acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, train_elapsed)

    # Evaluate on test
    log.info("Predicting on held-out test...")
    from sklearn.preprocessing import normalize as sk_normalize
    X_test_norm = sk_normalize(X_test, norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    pred_codes = head.predict_codes(
        torch.tensor(X_test_norm, device=device),
    )

    n_correct = sum(1 for p, t in zip(pred_codes, real_test_labels) if p == t)
    test_top1 = n_correct / len(real_test_labels) if real_test_labels else 0.0
    log.info("  TEST top-1=%.4f (%d/%d)",
             test_top1, n_correct, len(real_test_labels))

    # Per-category accuracy
    per_cat = per_category_accuracy(pred_codes, real_test_labels)

    # Failure characterization
    failures = characterize_failures(
        real_test_rows, pred_codes, real_test_labels, cat_set,
    )

    return {
        "config": dict(knob),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_train.shape[1]),
        "corpus_hash": corpus_hash(),
        "split": {
            "n_real_train": len(real_train_rows),
            "n_synth_train": len(synth_rows),
            "n_aug_train_after_dropout": len(aug_rows_kept),
            "n_test": len(real_test_labels),
            "n_test_classes": len(set(real_test_labels)),
            "seed": 42,
        },
        "train": {
            "fit_acc": round(train_result.final_train_acc, 4),
            "final_loss": round(train_result.final_train_loss, 4),
            "elapsed_sec": round(train_elapsed, 1),
            "epochs": train_result.epochs_run,
        },
        "test": {
            "top1": round(test_top1, 4),
            "n_correct": n_correct,
            "n_test": len(real_test_labels),
        },
        "per_category_accuracy": per_cat,
        "failures": {
            "kinds": failures["failure_kinds"],
            "examples_top": failures["examples_top"][:30],
        },
        "predictions": [
            {"key": f"{r.table}.{r.column}", "true": t, "pred": p, "correct": p == t}
            for r, t, p in zip(real_test_rows, real_test_labels, pred_codes)
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────

BASELINE_ANCHORS = {
    "tfidf_kronecker_fit_on_train_top1_kept": 0.9893,
    "modernbert_kronecker_fit_on_train_top1_kept": 0.0426,
    "modernbert_factorized_fit_on_train_top1_kept": 0.9938,
    "modernbert_factorized_HELD_OUT_baseline_top1": 0.6125,
}


def _baseline_per_category() -> dict[str, float]:
    """Load the baseline (pre-corpus_v2) per-category accuracy from
    build/reflect_nhsvm_eval_shap/per_row_attribution.json if present, so
    the report can show per-category lift.  Falls back to empty dict."""
    p = Path("build/reflect_nhsvm_eval_shap/per_row_attribution.json")
    if not p.exists():
        return {}
    try:
        records = json.loads(p.read_text())
        by_class: dict[str, list[bool]] = defaultdict(list)
        for rec in records:
            by_class[rec["true"]].append(rec["correct"])
        return {cls: sum(hits) / len(hits) for cls, hits in by_class.items()}
    except Exception:
        return {}


def render_report(result: dict) -> str:
    lines: list[str] = []
    lines.append("# Factorized NHSVM eval — corpus_v2 augmented training")
    lines.append("")
    s = result["split"]
    t = result["train"]
    e = result["test"]
    lines.append(
        f"Training set: **{s['n_synth_train']} synth + {s['n_real_train']} real "
        f"= {s['n_aug_train_after_dropout']} (post-NHSVM-singleton-dropout)**.  "
        f"Test: **{s['n_test']} held-out real rows** across "
        f"{s['n_test_classes']} classes, byte-identical to the 61.25% baseline "
        f"split (seed=42).  Encoder: {result['encoder']}."
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **Held-out TEST top-1: {e['top1']:.4f} ({e['n_correct']}/{e['n_test']})**")
    delta = e["top1"] - BASELINE_ANCHORS["modernbert_factorized_HELD_OUT_baseline_top1"]
    sign = "+" if delta >= 0 else ""
    lines.append(
        f"- vs **baseline 61.25%** held-out (855-real-only training): "
        f"**{sign}{delta:.4f}** ({sign}{delta*100:.2f}pp)"
    )
    lines.append(f"- Train fit-acc: {t['fit_acc']:.4f}")
    lines.append(f"- Train time: {t['elapsed_sec']:.1f}s "
                 f"({t['epochs']} epochs, loss={t['final_loss']:.4f})")
    lines.append("")

    lines.append("## Comparison anchors")
    lines.append("")
    lines.append("| Source | regime | top-1 |")
    lines.append("|---|---|---:|")
    lines.append("| reflect_nhsvm.py | TF-IDF + Kronecker, fit-on-train | 0.9893 on kept |")
    lines.append("| reflect_nhsvm_modernbert.py | ModernBERT + Kronecker, fit-on-train | 0.0426 (collapsed) |")
    lines.append("| reflect_nhsvm_factorized.py | ModernBERT + Factorized, fit-on-train | 0.9938 on kept |")
    lines.append(f"| reflect_nhsvm_eval_shap.py (Phase A) | ModernBERT + Factorized, HELD-OUT TEST, real-only train | **0.6125** |")
    lines.append(f"| **this report** | ModernBERT + Factorized, HELD-OUT TEST, synth+real train | **{e['top1']:.4f}** |")
    lines.append("")

    # Per-category lift
    baseline_per_cat = _baseline_per_category()
    if baseline_per_cat:
        lines.append("## Per-category lift (top 20 improvements + top 10 regressions)")
        lines.append("")
        deltas = []
        for cls, info in result["per_category_accuracy"].items():
            base_acc = baseline_per_cat.get(cls, 0.0)
            new_acc = info["accuracy"]
            deltas.append((cls, base_acc, new_acc, new_acc - base_acc, info["n_test"]))
        deltas_imp = sorted([d for d in deltas if d[3] > 0],
                             key=lambda x: -x[3])[:20]
        deltas_reg = sorted([d for d in deltas if d[3] < 0],
                             key=lambda x: x[3])[:10]
        lines.append("### Improvements")
        lines.append("")
        lines.append("| code | baseline | corpus_v2 | Δ | n_test |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, b, n, d, nt in deltas_imp:
            lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | +{d:.3f} | {nt} |")
        lines.append("")
        if deltas_reg:
            lines.append("### Regressions")
            lines.append("")
            lines.append("| code | baseline | corpus_v2 | Δ | n_test |")
            lines.append("|---|---:|---:|---:|---:|")
            for cls, b, n, d, nt in deltas_reg:
                lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | {d:.3f} | {nt} |")
            lines.append("")

    # Failure breakdown
    lines.append("## Failure-mode breakdown (test set)")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, v in sorted(result["failures"]["kinds"].items(),
                       key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Success criterion check")
    lines.append("")
    if e["top1"] >= 0.80:
        lines.append(
            f"✓ **Primary criterion MET**: held-out test top-1 = {e['top1']:.4f} ≥ 0.80 target. "
            f"The corpus expansion delivers the architectural pivot's promised lift."
        )
    elif e["top1"] >= 0.70:
        lines.append(
            f"~ **Promising but below target**: held-out test top-1 = {e['top1']:.4f} in the 70-80% zone. "
            f"Phase E iterative refinement (scripts/refine_generators_from_failures.py) recommended — "
            f"focus on the bottom-20 categories in the regression table above."
        )
    else:
        lines.append(
            f"✗ **Below 70% — corpus alone may not be the answer**: held-out test top-1 = {e['top1']:.4f}. "
            f"Consider structural simplification (low-rank `W_n`, weight sharing across siblings) "
            f"in addition to / instead of more corpus."
        )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rows-per-node-sanity", type=int, default=None,
                    help="Cap synth rows per code (for fast iteration)")
    args = ap.parse_args()

    # Logging both to console and to RUN_LOG
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RUN_LOG, mode="w"),
        ],
    )

    result = run_eval(
        refresh_embeddings=args.refresh_embeddings,
        batch_size=args.batch_size,
        rows_per_node_sanity=args.rows_per_node_sanity,
    )

    # Strip the verbose `predictions` from the main JSON; persist separately
    predictions = result.pop("predictions")
    RESULTS_JSON.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", RESULTS_JSON)

    # Per-category accuracy dump
    PER_CAT_JSON.write_text(json.dumps({
        "per_category": result["per_category_accuracy"],
        "predictions": predictions,
    }, indent=2))
    log.info("Wrote %s", PER_CAT_JSON)

    report = render_report(result)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    print("=== Phase D headline ===")
    e = result["test"]
    delta = e["top1"] - BASELINE_ANCHORS["modernbert_factorized_HELD_OUT_baseline_top1"]
    sign = "+" if delta >= 0 else ""
    print(f"  TEST top-1: {e['top1']:.4f} ({e['n_correct']}/{e['n_test']})")
    print(f"  vs baseline 61.25%: {sign}{delta*100:.2f}pp")
    print(f"  train fit-acc: {result['train']['fit_acc']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
