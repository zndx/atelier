#!/usr/bin/env python3
"""scripts/reflect_nhsvm_eval_shap.py — combined eval + SHAP diagnostic.

Two diagnostics in one run, both on the validated factorized NHSVM head:

  PHASE A — Held-out generalization eval
    Stratified 80/20 train/test split on the 1126-row trainable subset
    (post singleton drop).  Trains the factorized head on the train
    split with the validated best knob (svm/higher_lr: lr=5e-3, 300
    epochs, mean-pool, lean svm-text), reports held-out test top-1 +
    failure breakdown.  This is the FIRST real generalization number
    for the ModernBERT+factorized stack — fit-on-train told us capacity
    is sufficient; this tells us whether it transfers.

  PHASE B — Per-slice SHAP attribution
    Re-encodes the 7 ColumnFeatures text slices (column_name,
    column_type, sample_values, pattern_signals, sibling_context,
    value_description, source_table) SEPARATELY with ModernBERT,
    concatenates into a (N, 7*768) input matrix, trains the factorized
    head on the same 80/20 split, then computes EXACT Shapley values
    via the model's linear structure: phi_i = (M_alpha @ W)[pred_class, i]
    * (x_i - E[x_i]).  Aggregates per-dim attributions into 7 slice
    groups for operator-meaningful per-feature attribution.

    The interesting comparisons this surfaces:
      - Per-slice attribution magnitude — which features actually drive
        the model's decisions (sample_values? pattern_signals? source_table?)
      - svm-lean vs slices test accuracy — does per-slice encoding
        rescue the rich-text underperformance (hypothesis: each slice
        gets its own attention budget instead of competing in one pool)
      - Per-row attribution profile for failure cases — which slices
        the model relied on when it got the answer wrong

Outputs:
    build/reflect_nhsvm_eval_shap/report.md
    build/reflect_nhsvm_eval_shap/results.json
    build/reflect_nhsvm_eval_shap/per_row_attribution.json
    build/reflect_nhsvm_eval_shap/embeddings_slices.npz (new cache)

Usage:
    python scripts/reflect_nhsvm_eval_shap.py             # both phases
    python scripts/reflect_nhsvm_eval_shap.py --phase A   # eval only
    python scripts/reflect_nhsvm_eval_shap.py --phase B   # SHAP only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import Counter
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
    load_or_encode_embeddings,
    MODEL_ID,
    EMB_DIM,
)
from atelier.classify.factorized_nhsvm import (
    FactorizedNHSVMHead,
    fit_factorized_nhsvm,
)
from atelier.classify.features import extract_features, FEATURE_NAMES

log = logging.getLogger("reflect_nhsvm_eval_shap")

REPORT_DIR = Path("build/reflect_nhsvm_eval_shap")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = REPORT_DIR / "results.json"
ATTRIB_JSON = REPORT_DIR / "per_row_attribution.json"
REPORT_MD = REPORT_DIR / "report.md"
SLICE_EMBED_CACHE = REPORT_DIR / "embeddings_slices.npz"

# The 7 text-feature names CatBoost's pipeline uses (per
# src/atelier/classify/embedding.py:EMBED_TEXT_FEATURES).  We use this
# exact set so the slice-level attribution is directly comparable
# across channels in any future cross-channel diagnostic.
SLICE_NAMES = [
    "column_name",
    "column_type",
    "sample_values",
    "pattern_signals",
    "sibling_context",
    "value_description",
    "source_table",
]

# Re-export the public library API from atelier core.  Scripts that
# still `from reflect_nhsvm_eval_shap import ...` continue to work; the
# canonical location going forward is `atelier.optimize.svm.kfold`.
from atelier.optimize.svm.kfold import (  # noqa: E402
    BEST_KNOB,
    stratified_k_fold,
    stratified_split,
)


# ──────────────────────────────────────────────────────────────────────
# Per-slice text extraction
# ──────────────────────────────────────────────────────────────────────

def build_slice_texts(rows: list[Row]) -> dict[str, list[str]]:
    """For each row, build the per-feature text string used by the
    CatBoost-style embedding pipeline.  Returns dict[slice_name, list[str]]
    with len(rows) entries each.
    """
    result: dict[str, list[str]] = {name: [] for name in SLICE_NAMES}
    for r in rows:
        cf = extract_features(
            column_name=r.column,
            column_type=r.column_type,
            values=r.sample_values,
            siblings=r.siblings_full,
            source_table=r.table,
        )
        for slice_name in SLICE_NAMES:
            mask = {n: (n == slice_name) for n in FEATURE_NAMES}
            text = cf.to_embedding_text(mask).strip()
            # Avoid empty-string into ModernBERT — pad with a sentinel
            # so the encoder always produces a real vector.
            result[slice_name].append(text or "<empty>")
    return result


def load_or_encode_slice(
    slice_name: str, texts: list[str],
    *, refresh: bool = False, batch_size: int = 64,
) -> np.ndarray:
    """Encode one slice's texts, cached per (slice_name, text_hash)."""
    text_hash = hashlib.sha1(
        ("\n".join(texts) + f"|slice={slice_name}").encode("utf-8")
    ).hexdigest()[:12]
    cache_key = f"mb-base_slice-{slice_name}_{text_hash}"
    if SLICE_EMBED_CACHE.exists() and not refresh:
        cache = np.load(SLICE_EMBED_CACHE, allow_pickle=False)
        if cache_key in cache.files:
            log.info("  loaded cached slice (%s)", cache_key)
            return cache[cache_key]
    embeddings = encode_modernbert(texts, batch_size=batch_size, pooling="mean")
    existing: dict = {}
    if SLICE_EMBED_CACHE.exists():
        cache = np.load(SLICE_EMBED_CACHE, allow_pickle=False)
        existing = {k: cache[k] for k in cache.files}
    existing[cache_key] = embeddings
    np.savez_compressed(SLICE_EMBED_CACHE, **existing)
    log.info("  wrote slice cache → %s (%d variants stored)",
             SLICE_EMBED_CACHE, len(existing))
    return embeddings


# ──────────────────────────────────────────────────────────────────────
# Train / eval helper
# ──────────────────────────────────────────────────────────────────────

def train_and_eval(
    X: np.ndarray,
    labels: list[str],
    category_set,
    train_idx: list[int],
    test_idx: list[int],
    *,
    knob: dict | None = None,
    label_for_log: str = "",
) -> tuple[FactorizedNHSVMHead, dict]:
    """Train factorized head on X[train_idx], eval on X[test_idx]."""
    knob = knob or BEST_KNOB
    X_train = X[train_idx]
    y_train = [labels[i] for i in train_idx]
    X_test = X[test_idx]
    y_test = [labels[i] for i in test_idx]

    log.info("[%s] train n=%d  test n=%d  train-classes=%d  test-classes=%d",
             label_for_log,
             len(train_idx), len(test_idx),
             len(set(y_train)), len(set(y_test)))

    head, train_result = fit_factorized_nhsvm(
        X_train, y_train, category_set,
        **knob, verbose=False,
    )

    # Predict on test set
    from sklearn.preprocessing import normalize as sk_normalize
    X_test_norm = sk_normalize(X_test, norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    X_test_t = torch.tensor(X_test_norm, device=device)
    pred_codes = head.predict_codes(X_test_t)

    n_correct = sum(1 for p, t in zip(pred_codes, y_test) if p == t)
    n_test = len(y_test)
    test_acc = n_correct / n_test if n_test else 0.0

    log.info("[%s] train fit-acc=%.4f  TEST top-1=%.4f (%d/%d)",
             label_for_log, train_result.final_train_acc,
             test_acc, n_correct, n_test)

    return head, {
        "n_train": len(train_idx),
        "n_test": n_test,
        "n_test_classes": len(set(y_test)),
        "train_fit_acc": train_result.final_train_acc,
        "test_top1": test_acc,
        "n_test_correct": n_correct,
        "elapsed_sec": round(train_result.elapsed_sec, 2),
        "test_predictions": pred_codes,
        "test_true": y_test,
        "test_indices": test_idx,
    }


# ──────────────────────────────────────────────────────────────────────
# SHAP — exact Shapley values for linear models
# ──────────────────────────────────────────────────────────────────────

def linear_shap_per_row(
    head: FactorizedNHSVMHead,
    X: np.ndarray,
    background_mean: np.ndarray,
    predicted_classes: list[int],
) -> np.ndarray:
    """Exact Shapley values for the factorized head, treating it as a
    linear model post-fit.

    The factorized head's path score for class y is:
        gamma(x, y) = sum_n M_alpha[y, n] * (W_n^T x)
                    = ((M_alpha @ W) @ x)[y]
    So for a fixed predicted class y*, the model is literally a linear
    function of x with effective coefficient (M_alpha @ W)[y*, :].
    Exact Shapley values for a linear model: phi_i = coef_i * (x_i - E[x_i]).
    """
    W = head.W.detach().cpu().numpy()                # (n_nodes, d)
    M_alpha = head.M_alpha.detach().cpu().numpy()    # (n_nodes, n_nodes)
    effective_coef = M_alpha @ W                     # (n_nodes, d)

    N, d = X.shape
    attributions = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        y = predicted_classes[i]
        coef_y = effective_coef[y]                   # (d,)
        attributions[i] = coef_y * (X[i] - background_mean)
    return attributions


def aggregate_to_groups(
    per_dim: np.ndarray,
    group_slices: list[tuple[int, int]],
    group_names: list[str],
) -> dict[str, np.ndarray]:
    """Sum per-dim attributions within slice groups → (N, n_groups)."""
    grouped = {}
    for name, (start, end) in zip(group_names, group_slices):
        grouped[name] = per_dim[:, start:end].sum(axis=1)
    return grouped


# ──────────────────────────────────────────────────────────────────────
# Phase A: lean svm-text test eval
# ──────────────────────────────────────────────────────────────────────

def run_phase_a(
    rows: list[Row], labels: list[str], category_set,
    train_idx: list[int], test_idx: list[int],
    *, batch_size: int,
) -> dict:
    log.info("\n=== PHASE A: lean svm-text held-out test eval ===")
    texts, labels_check = build_texts_and_labels(rows)
    assert labels_check == labels, "label mismatch in build_texts_and_labels"
    X = load_or_encode_embeddings(
        texts, refresh=False, batch_size=batch_size, pooling="mean",
    )
    log.info("  embeddings shape: %s", X.shape)

    head, eval_result = train_and_eval(
        X, labels, category_set, train_idx, test_idx,
        label_for_log="phase A (svm-text)",
    )

    # Failure characterization on test set
    test_pred_full: list[str | None] = [None] * len(labels)
    for ti, pred in zip(test_idx, eval_result["test_predictions"]):
        test_pred_full[ti] = pred
    failures = characterize_failures(
        [rows[i] for i in test_idx],
        eval_result["test_predictions"],
        eval_result["test_true"],
        category_set,
    )

    return {
        "config": dict(BEST_KNOB, pooling="mean", text_shape="svm-lean"),
        "encoder": MODEL_ID,
        "encoder_dim": int(X.shape[1]),
        "eval": {k: v for k, v in eval_result.items()
                  if k not in {"test_predictions", "test_true", "test_indices"}},
        "test_failure_kinds": failures["failure_kinds"],
        "test_failure_examples": failures["examples_top"][:20],
    }


# ──────────────────────────────────────────────────────────────────────
# Phase B: per-slice encoding + SHAP
# ──────────────────────────────────────────────────────────────────────

def run_phase_b(
    rows: list[Row], labels: list[str], category_set,
    train_idx: list[int], test_idx: list[int],
    *, batch_size: int, refresh_slices: bool = False,
) -> dict:
    log.info("\n=== PHASE B: per-slice encoding + SHAP ===")

    log.info("  building per-slice texts (%d slices)", len(SLICE_NAMES))
    slice_texts = build_slice_texts(rows)
    for name in SLICE_NAMES:
        avg_chars = sum(len(t) for t in slice_texts[name]) / max(len(slice_texts[name]), 1)
        log.info("    slice %-20s  avg_chars=%.0f", name, avg_chars)

    log.info("  encoding %d slices with ModernBERT...", len(SLICE_NAMES))
    slice_embs: dict[str, np.ndarray] = {}
    for name in SLICE_NAMES:
        slice_embs[name] = load_or_encode_slice(
            name, slice_texts[name],
            refresh=refresh_slices, batch_size=batch_size,
        )

    # Concatenate slices into one (N, 7*768) input
    X_concat = np.concatenate(
        [slice_embs[name] for name in SLICE_NAMES], axis=1,
    ).astype(np.float32)
    log.info("  concatenated input shape: %s", X_concat.shape)

    # Slice index map for SHAP grouping
    group_slices: list[tuple[int, int]] = []
    offset = 0
    for name in SLICE_NAMES:
        d = slice_embs[name].shape[1]
        group_slices.append((offset, offset + d))
        offset += d

    head, eval_result = train_and_eval(
        X_concat, labels, category_set, train_idx, test_idx,
        label_for_log="phase B (slices)",
    )

    # SHAP on test set
    log.info("  computing per-row SHAP attribution on test set...")
    from sklearn.preprocessing import normalize as sk_normalize
    X_train_norm = sk_normalize(X_concat[train_idx], norm="l2", axis=1).astype(np.float32)
    X_test_norm = sk_normalize(X_concat[test_idx], norm="l2", axis=1).astype(np.float32)
    bg_mean = X_train_norm.mean(axis=0)

    # Map predicted code strings → node indices
    pred_indices = [head.code_to_idx[p] for p in eval_result["test_predictions"]]

    t0 = time.time()
    per_dim_attr = linear_shap_per_row(head, X_test_norm, bg_mean, pred_indices)
    grouped_attr = aggregate_to_groups(per_dim_attr, group_slices, SLICE_NAMES)
    log.info("  SHAP done in %.1fs", time.time() - t0)

    # Per-slice mean attribution magnitude across test set
    slice_mean_abs = {
        name: float(np.abs(grouped_attr[name]).mean())
        for name in SLICE_NAMES
    }
    slice_mean_signed = {
        name: float(grouped_attr[name].mean())
        for name in SLICE_NAMES
    }
    log.info("  per-slice |attribution| mean across %d test rows:",
             len(test_idx))
    for name in sorted(SLICE_NAMES, key=lambda n: -slice_mean_abs[n]):
        log.info("    %-20s  |attr|=%+.4f  signed=%+.4f",
                 name, slice_mean_abs[name], slice_mean_signed[name])

    # Failure characterization
    failures = characterize_failures(
        [rows[i] for i in test_idx],
        eval_result["test_predictions"],
        eval_result["test_true"],
        category_set,
    )

    # Per-row attribution dump (for the operator-side detailed view)
    per_row_records = []
    for k, ti in enumerate(test_idx):
        true = eval_result["test_true"][k]
        pred = eval_result["test_predictions"][k]
        row = rows[ti]
        rec = {
            "key": f"{row.table}.{row.column}",
            "true": true,
            "pred": pred,
            "correct": pred == true,
            "attributions": {
                name: float(grouped_attr[name][k]) for name in SLICE_NAMES
            },
        }
        per_row_records.append(rec)

    return {
        "config": dict(BEST_KNOB, pooling="mean", text_shape="per-slice"),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_concat.shape[1]),
        "slice_dims": [int(slice_embs[name].shape[1]) for name in SLICE_NAMES],
        "eval": {k: v for k, v in eval_result.items()
                  if k not in {"test_predictions", "test_true", "test_indices"}},
        "test_failure_kinds": failures["failure_kinds"],
        "test_failure_examples": failures["examples_top"][:20],
        "shap": {
            "slice_mean_abs": slice_mean_abs,
            "slice_mean_signed": slice_mean_signed,
            "per_row_count": len(per_row_records),
        },
        "_per_row_records": per_row_records,  # stripped before report json
    }


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def render_report(results: dict, rows: list[Row]) -> str:
    lines = []
    lines.append("# NHSVM held-out eval + SHAP attribution")
    lines.append("")
    lines.append(
        f"_Generated from {len(rows)} rows in `{AGENT_MEDIATED}`._  "
        f"Best-knob config: lr={BEST_KNOB['lr']}, epochs={BEST_KNOB['epochs']}, "
        f"mean-pool ModernBERT, factorized NHSVM head."
    )
    lines.append("")
    lines.append(
        f"**Train/test split**: stratified 80/20 on the {results['split']['n_trainable']}-row "
        f"trainable subset (post {results['split']['n_singletons']}-singleton drop), "
        f"train={results['split']['n_train']}, test={results['split']['n_test']}, "
        f"test classes covered = {results['split']['n_test_classes']}/"
        f"{results['split']['n_trainable_classes']}."
    )
    lines.append("")

    if "phase_a" in results:
        a = results["phase_a"]
        lines.append("## Phase A — lean svm-text held-out eval")
        lines.append("")
        lines.append(f"- Encoder: `{a['encoder']}` ({a['encoder_dim']}-dim mean-pool)")
        lines.append(f"- Train fit-acc: **{a['eval']['train_fit_acc']:.4f}**")
        lines.append(f"- **TEST top-1: {a['eval']['test_top1']:.4f}** "
                     f"({a['eval']['n_test_correct']}/{a['eval']['n_test']})")
        lines.append(f"- Train time: {a['eval']['elapsed_sec']:.1f}s")
        lines.append("")
        lines.append("### Phase A — test-set failure breakdown")
        lines.append("")
        lines.append("| Failure kind | Count |")
        lines.append("|---|---:|")
        for k, v in sorted(a["test_failure_kinds"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("### Sample test residual failures (top 15)")
        lines.append("")
        lines.append("| key | true | predicted | kind |")
        lines.append("|---|---|---|---|")
        for ex in a["test_failure_examples"][:15]:
            if ex["kind"] == "singleton_dropped":
                continue
            lines.append(
                f"| `{ex['key']}` | {ex.get('mnemonic_true','?')} "
                f"({ex['true']}) | {ex['pred']} | {ex['kind']} |"
            )
        lines.append("")

    if "phase_b" in results:
        b = results["phase_b"]
        lines.append("## Phase B — per-slice encoding + SHAP")
        lines.append("")
        lines.append(f"- Encoder: `{b['encoder']}`")
        lines.append(f"- 7 slices × 768 dim = {b['encoder_dim']} input dims")
        lines.append(f"- Train fit-acc: **{b['eval']['train_fit_acc']:.4f}**")
        lines.append(f"- **TEST top-1: {b['eval']['test_top1']:.4f}** "
                     f"({b['eval']['n_test_correct']}/{b['eval']['n_test']})")
        lines.append(f"- Train time: {b['eval']['elapsed_sec']:.1f}s")
        lines.append("")

        if "phase_a" in results:
            a = results["phase_a"]
            delta = b['eval']['test_top1'] - a['eval']['test_top1']
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"**Phase A vs Phase B test delta**: {sign}{delta:.4f} "
                f"(slices {'beats' if delta > 0 else 'loses to'} lean svm-text)"
            )
            lines.append("")

        lines.append("### Per-slice SHAP attribution (mean across test set)")
        lines.append("")
        lines.append(
            "`|attr|` is the mean absolute attribution magnitude per slice — "
            "how much the slice MOVES the prediction on average, regardless "
            "of direction.  `signed` is the mean signed attribution — "
            "positive means the slice pushes toward the predicted class on "
            "average."
        )
        lines.append("")
        lines.append("| Slice | mean &#124;attribution&#124; | mean signed |")
        lines.append("|---|---:|---:|")
        slice_abs = b["shap"]["slice_mean_abs"]
        slice_signed = b["shap"]["slice_mean_signed"]
        for name in sorted(SLICE_NAMES, key=lambda n: -slice_abs[n]):
            lines.append(
                f"| {name} | {slice_abs[name]:+.4f} | {slice_signed[name]:+.4f} |"
            )
        lines.append("")

        lines.append("### Phase B — test-set failure breakdown")
        lines.append("")
        lines.append("| Failure kind | Count |")
        lines.append("|---|---:|")
        for k, v in sorted(b["test_failure_kinds"].items(),
                           key=lambda kv: -kv[1]):
            lines.append(f"| {k} | {v} |")
        lines.append("")
        lines.append("### Sample test residual failures (top 15)")
        lines.append("")
        lines.append("| key | true | predicted | kind |")
        lines.append("|---|---|---|---|")
        for ex in b["test_failure_examples"][:15]:
            if ex["kind"] == "singleton_dropped":
                continue
            lines.append(
                f"| `{ex['key']}` | {ex.get('mnemonic_true','?')} "
                f"({ex['true']}) | {ex['pred']} | {ex['kind']} |"
            )
        lines.append("")

    lines.append("## What to read this against")
    lines.append("")
    lines.append(
        "| Source | regime | best top-1 |\n"
        "|---|---|---:|"
    )
    tfidf_ref = results.get("tfidf_reference_top1_kept")
    fact_fitontrain = results.get("factorized_fit_on_train_top1_kept")
    if tfidf_ref is not None:
        lines.append(f"| reflect_nhsvm.py | TF-IDF + Kronecker, fit-on-train | {tfidf_ref:.4f} on kept |")
    if fact_fitontrain is not None:
        lines.append(f"| reflect_nhsvm_factorized.py | ModernBERT + Factorized, fit-on-train | {fact_fitontrain:.4f} on kept |")
    if "phase_a" in results:
        lines.append(f"| **this report — Phase A** | **ModernBERT + Factorized, HELD-OUT TEST** | **{results['phase_a']['eval']['test_top1']:.4f}** |")
    if "phase_b" in results:
        lines.append(f"| **this report — Phase B** | **ModernBERT + Factorized + per-slice, HELD-OUT TEST** | **{results['phase_b']['eval']['test_top1']:.4f}** |")
    lines.append("")
    lines.append(
        "Fit-on-train tells us capacity; held-out test tells us whether "
        "what the model learned transfers.  A large gap (fit-acc >> test-acc) "
        "means the head is memorizing rather than generalizing."
    )
    lines.append("")

    return "\n".join(lines)


def _read_reference_top1(path: Path) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return max(v["measure"]["top1_on_kept"] for v in data["variants"])
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase", choices=["A", "B", "both"], default="both",
                    help="Which phase(s) to run (default: both)")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="Re-pull Hive samples")
    ap.add_argument("--refresh-embeddings", action="store_true",
                    help="Re-encode lean svm-text with ModernBERT")
    ap.add_argument("--refresh-slices", action="store_true",
                    help="Re-encode per-slice texts with ModernBERT")
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Encoder batch size")
    ap.add_argument("--database", default="reference_corpus",
                    help="Hive database")
    ap.add_argument("--seed", type=int, default=42,
                    help="Train/test split seed")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log.info("=== Phase 0: load reference + Hive samples ===")
    rows = load_rows(refresh_cache=args.refresh_cache, database=args.database)

    log.info("=== Build category set (full vocab) ===")
    category_set = build_category_set()

    # Get canonical labels (same across both phases)
    _, labels = build_texts_and_labels(rows)
    code_counts = Counter(labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    log.info("  %d singleton classes (excluded from split)", len(singletons))

    # Filter rows/labels to trainable subset (singletons removed)
    trainable_mask = [l not in singletons for l in labels]
    trainable_rows = [r for r, m in zip(rows, trainable_mask) if m]
    trainable_labels = [l for l, m in zip(labels, trainable_mask) if m]
    log.info("  trainable subset: %d rows, %d classes",
             len(trainable_rows), len(set(trainable_labels)))

    # Stratified 80/20 split on trainable rows.  Note: indices here are
    # into the FILTERED trainable arrays, NOT the original rows.
    train_idx, test_idx = stratified_split(
        trainable_labels, test_size=0.2, seed=args.seed,
    )
    log.info("  split: train=%d  test=%d  test_classes=%d",
             len(train_idx), len(test_idx), len({trainable_labels[i] for i in test_idx}))

    results: dict = {
        "split": {
            "n_trainable": len(trainable_rows),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_singletons": len(singletons),
            "n_trainable_classes": len(set(trainable_labels)),
            "n_test_classes": len({trainable_labels[i] for i in test_idx}),
            "seed": args.seed,
        },
        "tfidf_reference_top1_kept": _read_reference_top1(
            Path("build/reflect_nhsvm/results.json")
        ),
        "factorized_fit_on_train_top1_kept": _read_reference_top1(
            Path("build/reflect_nhsvm_factorized/results.json")
        ),
    }

    if args.phase in ("A", "both"):
        results["phase_a"] = run_phase_a(
            trainable_rows, trainable_labels, category_set,
            train_idx, test_idx,
            batch_size=args.batch_size,
        )

    if args.phase in ("B", "both"):
        phase_b = run_phase_b(
            trainable_rows, trainable_labels, category_set,
            train_idx, test_idx,
            batch_size=args.batch_size,
            refresh_slices=args.refresh_slices,
        )
        # Dump per-row attribution to a separate JSON
        per_row_records = phase_b.pop("_per_row_records")
        ATTRIB_JSON.write_text(json.dumps(per_row_records, indent=2))
        log.info("Wrote per-row attribution → %s", ATTRIB_JSON)
        results["phase_b"] = phase_b

    RESULTS_JSON.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", RESULTS_JSON)

    report = render_report(results, trainable_rows)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    print("=== Headline numbers ===")
    if "phase_a" in results:
        a = results["phase_a"]
        print(f"  Phase A (lean svm-text):  fit={a['eval']['train_fit_acc']:.4f}  "
              f"TEST={a['eval']['test_top1']:.4f}")
    if "phase_b" in results:
        b = results["phase_b"]
        print(f"  Phase B (per-slice+SHAP): fit={b['eval']['train_fit_acc']:.4f}  "
              f"TEST={b['eval']['test_top1']:.4f}")
        top_slice = max(SLICE_NAMES, key=lambda n: b["shap"]["slice_mean_abs"][n])
        print(f"  Top SHAP attribution: {top_slice} "
              f"(|attr|={b['shap']['slice_mean_abs'][top_slice]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
