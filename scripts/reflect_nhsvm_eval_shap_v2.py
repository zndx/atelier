#!/usr/bin/env python3
"""scripts/reflect_nhsvm_eval_shap_v2.py — Phase D under train/validate/test framing.

Trains the factorized NHSVM head on the synthetic corpus alone (no
reference data in training); validates against the full agent-mediated
reference set.  The 271-example stratified slice (seed=42) is reported
as a named subset of validate for continuity with the historical 0.6125
baseline, but the load-bearing number is full-validate top-1.

The reference is NOT held out for "test" — the test surface is hive-poc
at large, evaluated separately by ``scripts/svm_target_health.py`` with
the best-pass model produced by the refinement loop.

Outputs (per pass when --pass-idx N is supplied):
  build/reflect_nhsvm_eval_shap_v2/report.md
  build/reflect_nhsvm_eval_shap_v2/results_pass{N}.json
  build/reflect_nhsvm_eval_shap_v2/per_category_accuracy_pass{N}.json
  build/reflect_nhsvm_eval_shap_v2/run.log
  build/reflect_nhsvm_eval_shap_v2/embeddings_cache.npz
  build/reflect_nhsvm_eval_shap_v2/results.json  (symlink to latest pass)

Usage:
  python scripts/reflect_nhsvm_eval_shap_v2.py
  python scripts/reflect_nhsvm_eval_shap_v2.py --pass-idx 3
  python scripts/reflect_nhsvm_eval_shap_v2.py --refresh-embeddings
  python scripts/reflect_nhsvm_eval_shap_v2.py --corpus-dir build/data/svm_training/corpus_v3
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

DEFAULT_CORPUS_DIR = Path("build/data/svm_training/corpus_v2")


# ──────────────────────────────────────────────────────────────────────
# Load synth corpus + reconstruct as Row objects
# ──────────────────────────────────────────────────────────────────────

def load_synth_rows(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> list[Row]:
    """Load synth_rows.jsonl from a corpus directory."""
    synth_path = corpus_dir / "synth_rows.jsonl"
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synth corpus missing at {synth_path}. "
            f"Run scripts/generate_corpus_v2.py first."
        )
    rows: list[Row] = []
    with synth_path.open() as f:
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
    log.info("  loaded %d synth rows from %s", len(rows), synth_path)
    return rows


def corpus_hash(corpus_dir: Path = DEFAULT_CORPUS_DIR) -> str:
    """Hash of the synth corpus file for cache invalidation."""
    synth_path = corpus_dir / "synth_rows.jsonl"
    if not synth_path.exists():
        return "none"
    h = hashlib.sha1()
    with synth_path.open("rb") as f:
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
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    refresh_embeddings: bool = False,
    batch_size: int = 64,
    knob: dict | None = None,
    pass_idx: int | None = None,
) -> dict:
    """Phase D under the train/validate/test framing.

    Trains on synth-only; evaluates on the full agent-mediated reference
    set (validate).  Reports two top-1 numbers: full-validate (primary
    refinement signal) and the 271-example seed=42 stratified slice
    (continuity with the historical 0.6125 baseline).
    """
    knob = knob or BEST_KNOB
    pass_label = f"pass {pass_idx}" if pass_idx is not None else "single run"
    log.info("=== Phase D (%s): synth-only train → full-reference validate ===",
             pass_label)

    log.info("Loading reference rows (full validate dataset)...")
    real_rows = load_rows(refresh_cache=False, database="reference_corpus")
    _, real_labels = build_texts_and_labels(real_rows)

    # Drop singletons — needed for the continuity-slice stratified split
    # to be well-defined.  Has no effect on synth training.
    code_counts = Counter(real_labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    val_mask = [l not in singletons for l in real_labels]
    validate_rows = [r for r, m in zip(real_rows, val_mask) if m]
    validate_labels = [l for l, m in zip(real_labels, val_mask) if m]
    log.info("  validate (full reference): %d rows  (%d singletons dropped)",
             len(validate_rows), len(singletons))

    # Compute the historical 271-example stratified slice (seed=42) within
    # validate — used only as a named subset for continuity reporting.
    # Train/test naming reflects the historical seed-42 80/20 partition; we
    # take the held-out 20% as the continuity slice.
    _, continuity_idx = stratified_split(
        validate_labels, test_size=0.2, seed=42,
    )
    continuity_set = set(continuity_idx)
    log.info("  continuity slice (seed=42, top-1 historical anchor): %d rows",
             len(continuity_idx))

    # Load synth corpus (training set)
    log.info("Loading synth corpus from %s...", corpus_dir)
    synth_rows = load_synth_rows(corpus_dir)
    synth_labels = [r.code for r in synth_rows]

    # Drop synth classes with < 2 examples (NHSVM constraint)
    synth_counts = Counter(synth_labels)
    synth_keep = [i for i, l in enumerate(synth_labels) if synth_counts[l] >= 2]
    train_rows = [synth_rows[i] for i in synth_keep]
    train_labels = [synth_labels[i] for i in synth_keep]
    dropped = len(synth_rows) - len(train_rows)
    if dropped:
        log.info("  dropped %d synth rows with singleton classes", dropped)
    log.info("  train (synth-only): %d rows  %d distinct codes",
             len(train_rows), len(set(train_labels)))

    # Build texts (lean svm-text shape)
    train_texts, _ = build_texts_and_labels(train_rows)
    validate_texts, _ = build_texts_and_labels(validate_rows)

    # Encode (cached)
    log.info("Encoding synth-only train texts with ModernBERT...")
    ch = corpus_hash(corpus_dir)
    X_train = encode_with_cache(
        train_texts, cache_key=f"synth-only-{ch}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("Encoding full-reference validate texts with ModernBERT...")
    X_validate = encode_with_cache(
        validate_texts, cache_key="full-reference",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  X_train shape=%s  X_validate shape=%s",
             X_train.shape, X_validate.shape)

    # Train factorized head
    log.info("Training factorized NHSVM head...")
    cat_set = build_category_set()
    train_t0 = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, train_labels, cat_set,
        **knob, verbose=True, eval_every=50,
    )
    train_elapsed = time.time() - train_t0
    log.info("  train fit-acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, train_elapsed)

    # Predict on full validate
    log.info("Predicting on full-reference validate...")
    from sklearn.preprocessing import normalize as sk_normalize
    X_val_norm = sk_normalize(X_validate, norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    pred_codes = head.predict_codes(
        torch.tensor(X_val_norm, device=device),
    )

    # Two reported numbers: full-validate + continuity-271
    full_correct = sum(1 for p, t in zip(pred_codes, validate_labels) if p == t)
    full_top1 = full_correct / len(validate_labels) if validate_labels else 0.0

    cont_correct = 0
    cont_total = 0
    for i, (p, t) in enumerate(zip(pred_codes, validate_labels)):
        if i in continuity_set:
            cont_total += 1
            if p == t:
                cont_correct += 1
    cont_top1 = cont_correct / cont_total if cont_total else 0.0

    log.info("  === VALIDATE NUMBERS (%s) ===", pass_label)
    log.info("    full-reference top-1: %.4f (%d/%d)",
             full_top1, full_correct, len(validate_labels))
    log.info("    continuity-271 slice top-1: %.4f (%d/%d)",
             cont_top1, cont_correct, cont_total)

    # Per-category accuracy on FULL validate (more stable than 271-only)
    per_cat = per_category_accuracy(pred_codes, validate_labels)

    # Failure characterization
    failures = characterize_failures(
        validate_rows, pred_codes, validate_labels, cat_set,
    )

    return {
        "pass_idx": pass_idx,
        "config": dict(knob),
        "encoder": MODEL_ID,
        "encoder_dim": int(X_train.shape[1]),
        "corpus_hash": ch,
        "corpus_dir": str(corpus_dir),
        "split": {
            "n_synth_train": len(train_rows),
            "n_synth_train_classes": len(set(train_labels)),
            "n_validate": len(validate_labels),
            "n_validate_classes": len(set(validate_labels)),
            "n_continuity_slice": cont_total,
            "continuity_seed": 42,
        },
        "train": {
            "fit_acc": round(train_result.final_train_acc, 4),
            "final_loss": round(train_result.final_train_loss, 4),
            "elapsed_sec": round(train_elapsed, 1),
            "epochs": train_result.epochs_run,
        },
        "validate": {
            "full_top1": round(full_top1, 4),
            "full_n_correct": full_correct,
            "full_n": len(validate_labels),
            "continuity_top1": round(cont_top1, 4),
            "continuity_n_correct": cont_correct,
            "continuity_n": cont_total,
        },
        "per_category_accuracy": per_cat,
        "failures": {
            "kinds": failures["failure_kinds"],
            "examples_top": failures["examples_top"][:30],
        },
        "predictions": [
            {"key": f"{r.table}.{r.column}", "true": t, "pred": p,
             "correct": p == t,
             "in_continuity_slice": i in continuity_set}
            for i, (r, t, p) in enumerate(zip(validate_rows, validate_labels, pred_codes))
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────

BASELINE_ANCHORS = {
    "tfidf_kronecker_fit_on_train_top1_kept": 0.9893,
    "modernbert_kronecker_fit_on_train_top1_kept": 0.0426,
    "modernbert_factorized_fit_on_train_top1_kept": 0.9938,
    # Different training regime (855 reference-train + 0 synth, evaluated
    # on 271 reference-test).  Preserved as historical context only.
    "historical_modernbert_factorized_held_out_271_real_only_train": 0.6125,
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
    pass_label = (f" — pass {result['pass_idx']}"
                  if result.get("pass_idx") is not None else "")
    lines.append(f"# Factorized NHSVM eval{pass_label} — synth-only train, full-reference validate")
    lines.append("")
    s = result["split"]
    t = result["train"]
    v = result["validate"]
    lines.append(
        f"Training set: **{s['n_synth_train']} synthetic examples** across "
        f"{s['n_synth_train_classes']} codes (synth corpus from {result['corpus_dir']}).  "
        f"Validate: **{s['n_validate']} reference examples** across "
        f"{s['n_validate_classes']} codes (full agent-mediated reference, "
        f"no held-out test slice — test is hive-poc-at-large via svm_target_health.py).  "
        f"Encoder: {result['encoder']}."
    )
    lines.append("")

    lines.append("## Headline (two validate numbers)")
    lines.append("")
    lines.append(
        f"- **Full-reference VALIDATE top-1: {v['full_top1']:.4f} "
        f"({v['full_n_correct']}/{v['full_n']})** — primary refinement signal"
    )
    lines.append(
        f"- **Continuity-271 slice VALIDATE top-1: {v['continuity_top1']:.4f} "
        f"({v['continuity_n_correct']}/{v['continuity_n']})** — seed=42 "
        f"stratified, comparable to the historical 0.6125 anchor in regime"
    )
    lines.append(f"- Train fit-acc: {t['fit_acc']:.4f}")
    lines.append(f"- Train time: {t['elapsed_sec']:.1f}s "
                 f"({t['epochs']} epochs, loss={t['final_loss']:.4f})")
    lines.append("")
    lines.append(
        "Note: the historical 0.6125 baseline used a DIFFERENT training "
        "regime (855 reference-train + 0 synth, evaluated on 271 reference-"
        "test).  The continuity-271 number here is on the SAME 271 examples "
        "but with synth-only training, so it is regime-comparable in "
        "evaluation surface but not in training data.  Both numbers should "
        "rise together as the corpus improves; if continuity-271 lags full-"
        "validate substantially, that signals the 271 slice over-represents "
        "structurally-hard codes."
    )
    lines.append("")

    lines.append("## Comparison anchors")
    lines.append("")
    lines.append("| Source | regime | top-1 |")
    lines.append("|---|---|---:|")
    lines.append("| reflect_nhsvm.py | TF-IDF + Kronecker, fit-on-train | 0.9893 on kept |")
    lines.append("| reflect_nhsvm_modernbert.py | ModernBERT + Kronecker, fit-on-train | 0.0426 (collapsed) |")
    lines.append("| reflect_nhsvm_factorized.py | ModernBERT + Factorized, fit-on-train | 0.9938 on kept |")
    lines.append("| historical Phase A | ModernBERT + Factorized, 271-test, real-only train | 0.6125 |")
    lines.append(f"| **this report (full-validate)** | ModernBERT + Factorized, full-reference validate, synth-only train | **{v['full_top1']:.4f}** |")
    lines.append(f"| **this report (continuity-271)** | ModernBERT + Factorized, 271-slice of validate, synth-only train | **{v['continuity_top1']:.4f}** |")
    lines.append("")

    # Per-category lift
    baseline_per_cat = _baseline_per_category()
    if baseline_per_cat:
        lines.append("## Per-category lift vs Phase A baseline (top 20 / top 10)")
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
        lines.append("| code | baseline | this pass | Δ | n_validate |")
        lines.append("|---|---:|---:|---:|---:|")
        for cls, b, n, d, nt in deltas_imp:
            lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | +{d:.3f} | {nt} |")
        lines.append("")
        if deltas_reg:
            lines.append("### Regressions")
            lines.append("")
            lines.append("| code | baseline | this pass | Δ | n_validate |")
            lines.append("|---|---:|---:|---:|---:|")
            for cls, b, n, d, nt in deltas_reg:
                lines.append(f"| `{cls}` | {b:.3f} | {n:.3f} | {d:.3f} | {nt} |")
            lines.append("")

    # Failure breakdown
    lines.append("## Failure-mode breakdown (full validate)")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, val in sorted(result["failures"]["kinds"].items(),
                          key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {val} |")
    lines.append("")

    lines.append("## Success criterion check (against plan thresholds)")
    lines.append("")
    full_top1 = v["full_top1"]
    if full_top1 >= 0.65:
        lines.append(
            f"✓ **Stretch target MET**: full-validate top-1 = {full_top1:.4f} ≥ 0.65. "
            f"Metrology-driven agent feedback produced net-positive iteration."
        )
    elif full_top1 >= 0.62:
        lines.append(
            f"~ **Primary criterion MET**: full-validate top-1 = {full_top1:.4f} ≥ 0.62. "
            f"Volume-cap + metrology demonstrate that volume wasn't the constraint; "
            f"refinement loop has room to push toward 0.65 stretch."
        )
    elif full_top1 >= 0.60:
        lines.append(
            f"~ **Approaching target**: full-validate top-1 = {full_top1:.4f} in 0.60-0.62 zone. "
            f"Refinement loop should improve from here; if it plateaus, the next lever is "
            f"structural (pulling A_HD/A_ID/ENOS from SVM target, merging BILL/SHIP at parent)."
        )
    else:
        lines.append(
            f"✗ **Below 0.60**: full-validate top-1 = {full_top1:.4f}. "
            f"Likely indicates the corpus alone can't lift past structural limits. "
            f"Inspect per-code metrology trajectories for `abandon_structural` flags; "
            f"those codes need DST channel restructuring, not more synth."
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
    ap.add_argument("--pass-idx", type=int, default=None,
                    help="Refinement pass index; pass-numbered outputs "
                         "written when set.  Without, single-run outputs.")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Corpus directory containing synth_rows.jsonl")
    args = ap.parse_args()

    # Logging both to console and to RUN_LOG (append mode so pass-by-pass
    # runs accumulate in one log).
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(RUN_LOG, mode="a"),
        ],
    )

    result = run_eval(
        corpus_dir=args.corpus_dir,
        refresh_embeddings=args.refresh_embeddings,
        batch_size=args.batch_size,
        pass_idx=args.pass_idx,
    )

    # Pass-numbered output paths
    if args.pass_idx is not None:
        results_json = REPORT_DIR / f"results_pass{args.pass_idx}.json"
        per_cat_json = REPORT_DIR / f"per_category_accuracy_pass{args.pass_idx}.json"
    else:
        results_json = RESULTS_JSON
        per_cat_json = PER_CAT_JSON

    # Strip the verbose `predictions` from the main JSON; persist separately
    predictions = result.pop("predictions")
    results_json.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", results_json)

    per_cat_json.write_text(json.dumps({
        "per_category": result["per_category_accuracy"],
        "predictions": predictions,
    }, indent=2))
    log.info("Wrote %s", per_cat_json)

    # Symlink results.json + per_category_accuracy.json → latest pass so
    # downstream consumers that don't pass --pass-idx keep working.
    if args.pass_idx is not None:
        for live, target in [
            (RESULTS_JSON, results_json),
            (PER_CAT_JSON, per_cat_json),
        ]:
            if live.exists() or live.is_symlink():
                live.unlink()
            live.symlink_to(target.name)
            log.info("Updated symlink %s → %s", live, target.name)

    report = render_report(result)
    REPORT_MD.write_text(report)
    log.info("Wrote %s", REPORT_MD)

    print()
    pass_label = (f"pass {args.pass_idx}" if args.pass_idx is not None
                  else "single run")
    print(f"=== Phase D headline ({pass_label}) ===")
    v = result["validate"]
    print(f"  full-validate top-1:     {v['full_top1']:.4f} "
          f"({v['full_n_correct']}/{v['full_n']})")
    print(f"  continuity-271 top-1:    {v['continuity_top1']:.4f} "
          f"({v['continuity_n_correct']}/{v['continuity_n']})")
    print(f"  train fit-acc:           {result['train']['fit_acc']:.4f}")
    print(f"  historical 0.6125 anchor (different regime): "
          f"reference-only train + 271-test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
