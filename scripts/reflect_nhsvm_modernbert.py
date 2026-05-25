#!/usr/bin/env python3
"""scripts/reflect_nhsvm_modernbert.py — ModernBERT-NHSVM capacity diagnostic.

A/B counterpart to ``reflect_nhsvm.py``.  Same 1149-row reference, same
NHSVM head, same fit-on-train methodology.  Only difference: replace
TF-IDF char-ngram + TruncatedSVD with ``answerdotai/ModernBERT-base``
mean-pooled embeddings (off-the-shelf, no fine-tuning).

The interesting measurement is not "does ModernBERT beat the TF-IDF
98.93% ceiling" — that's expected on a fit-on-train regime.  The
falsifiable claim from ``docs/src/architecture/classification.md:114``
is that TF-IDF's representational failures (CPF identifiers confused
with date-shaped strings, sub-word token overlap creating spurious
confusables) are *tokenization artifacts of the featurizer*, not
properties of the data.  ModernBERT operating over learned
representations should structurally separate the cases TF-IDF
conflates.  The diagnostic to look for is whether the residual failure
set under ModernBERT *excludes* those documented modes.

Outputs:
  build/reflect_nhsvm_modernbert/report.md       — human analysis
  build/reflect_nhsvm_modernbert/results.json    — raw per-config numbers
  build/reflect_nhsvm_modernbert/embeddings.npz  — cached encoder output
                                                   (re-runs use cache;
                                                   pass --refresh-embeddings
                                                   to re-encode)

Usage:
  python scripts/reflect_nhsvm_modernbert.py                  # baseline + sweep
  python scripts/reflect_nhsvm_modernbert.py --quick          # baseline only
  python scripts/reflect_nhsvm_modernbert.py --batch-size 32  # if A10G OOMs
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

# Reuse data loading + reference handling from the TF-IDF diagnostic so
# the two scripts measure the SAME rows with the SAME splits.
from reflect_nhsvm import (
    Row,
    load_rows,
    build_texts_and_labels,       # lean SVM-text: name | type | vals | siblings
    build_category_set,
    measure,
    characterize_failures,
    AGENT_MEDIATED,
)


def build_rich_texts_and_labels(rows: list[Row]) -> tuple[list[str], list[str]]:
    """Build per-row text using the rich ColumnFeatures.to_embedding_text()
    shape — same one cosine and the LLM prompts consume.  Adds pattern
    signals, ontology priors, value descriptions, scalar stats, source
    table — everything the lean SVM text omits.
    """
    from atelier.classify.features import extract_features
    texts: list[str] = []
    labels: list[str] = []
    for r in rows:
        cf = extract_features(
            column_name=r.column,
            column_type=r.column_type,
            values=r.sample_values,
            siblings=r.siblings_full,
            source_table=r.table,
        )
        texts.append(cf.to_embedding_text())
        labels.append(r.code)
    return texts, labels

log = logging.getLogger("reflect_nhsvm_modernbert")

REPORT_DIR = Path("build/reflect_nhsvm_modernbert")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE = REPORT_DIR / "embeddings.npz"
RESULTS_JSON = REPORT_DIR / "results.json"
REPORT_MD = REPORT_DIR / "report.md"

MODEL_ID = "answerdotai/ModernBERT-base"
EMB_DIM = 768  # ModernBERT-base hidden size


# ──────────────────────────────────────────────────────────────────────
# ModernBERT encoding
# ──────────────────────────────────────────────────────────────────────

def encode_modernbert(
    texts: list[str],
    *,
    batch_size: int = 64,
    max_length: int = 256,
    pooling: str = "mean",
    device: str | None = None,
):
    """Encode texts with ModernBERT-base; return (N, 768) float32 ndarray.

    ``pooling=mean`` uses attention-mask-weighted mean over token states
    (standard retrieval pooling).  ``pooling=cls`` uses the [CLS] token
    state directly.
    """
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("  encoder=%s  pooling=%s  device=%s  batch=%d  max_len=%d",
             MODEL_ID, pooling, device, batch_size, max_length)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval().to(device)

    out = np.empty((len(texts), EMB_DIM), dtype=np.float32)
    t0 = time.time()
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            enc = tok(chunk, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state  # (B, T, H)
            if pooling == "cls":
                vec = hidden[:, 0, :]
            elif pooling == "mean":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                summed = (hidden * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                vec = summed / counts
            else:
                raise ValueError(f"unknown pooling: {pooling}")
            out[start:start + len(chunk)] = vec.cpu().numpy()
            if (start // batch_size) % 10 == 0:
                done = start + len(chunk)
                rate = done / max(time.time() - t0, 1e-6)
                log.info("    %d/%d (%.0f rows/s)", done, len(texts), rate)
    log.info("  encoded %d texts in %.1fs", len(texts), time.time() - t0)
    return out


def load_or_encode_embeddings(
    texts: list[str], *, refresh: bool, batch_size: int,
    pooling: str = "mean",
):
    """Encode once per (texts-hash, pooling) tuple; cache to disk."""
    import hashlib
    import numpy as np

    text_hash = hashlib.sha1(
        ("\n".join(texts) + f"|pool={pooling}").encode("utf-8")
    ).hexdigest()[:12]
    cache_key = f"mb-base_{pooling}_{text_hash}"

    if EMBED_CACHE.exists() and not refresh:
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        if cache_key in cache.files:
            log.info("  loaded cached embeddings (%s)", cache_key)
            return cache[cache_key]

    embeddings = encode_modernbert(
        texts, batch_size=batch_size, pooling=pooling,
    )

    # Merge with existing cache (preserve other pooling variants)
    existing: dict = {}
    if EMBED_CACHE.exists():
        cache = np.load(EMBED_CACHE, allow_pickle=False)
        existing = {k: cache[k] for k in cache.files}
    existing[cache_key] = embeddings
    np.savez_compressed(EMBED_CACHE, **existing)
    log.info("  wrote embeddings cache → %s (%d variants stored)",
             EMBED_CACHE, len(existing))
    return embeddings


# ──────────────────────────────────────────────────────────────────────
# Fit + predict (NHSVM head on dense embeddings)
# ──────────────────────────────────────────────────────────────────────

def fit_and_predict_dense(
    X: "np.ndarray",  # type: ignore[name-defined]
    labels: list[str],
    category_set,
    *,
    svc_C: float = 1.0,
    svd_components: int | None = None,
    skip_calibration: bool = False,
):
    """Fit NHSVM head on precomputed dense embeddings; predict on same.

    The head is structurally identical to ``reflect_nhsvm.fit_and_predict``:
    Kronecker expansion + LinearSVC + (optional) CalibratedClassifierCV.
    The differences are:
    - Input is precomputed (N, D) dense embeddings, not raw texts
    - No TF-IDF, no FeatureUnion
    - L2-normalization of embeddings is ALWAYS applied (not a knob) —
      mathematically required for the LinearSVC L2-regularized objective
      to weight dimensions comparably; without it the optimizer's penalty
      biases the solution away from high-variance signal-bearing dims.
      Empirically confirmed: omitting normalization tanked the original
      baseline to 0.27% top-1 vs random's 0.56% on 177 classes.
    - SVD is OFF by default (svd_components=None): we have memory headroom
      (23GB on A10G, 220k × 1126 × 8B ≈ 2GB) and ModernBERT's natural rank
      is close to its full 768 dims.  SVD variants in the sweep are
      diagnostic comparators against the TF-IDF baseline shape.
    """
    from collections import Counter
    import numpy as np
    from sklearn.decomposition import TruncatedSVD
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import normalize as sk_normalize
    from atelier.classify.svm_classifier import HierarchicalFeatureExpander

    counts = Counter(labels)
    keep_idx = [i for i, l in enumerate(labels) if counts[l] >= 2]
    dropped = len(labels) - len(keep_idx)
    if dropped:
        log.info("  dropped %d singleton-class rows", dropped)
    X_k = X[keep_idx]
    labels_k = [labels[i] for i in keep_idx]
    min_count = min(Counter(labels_k).values())

    # Always L2-normalize — mathematically required for L2-regularized
    # LinearSVC to weight dimensions comparably.  Not a knob.
    X_k = sk_normalize(X_k, norm="l2", axis=1)

    if svd_components is not None:
        n_components = min(svd_components, X_k.shape[1] - 1, X_k.shape[0] - 1)
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        X_reduced = svd.fit_transform(X_k)
        n_features_in = n_components
    else:
        X_reduced = X_k
        n_features_in = X_k.shape[1]
        n_components = None

    alphas = category_set.compute_nhsvm_alphas()
    expander = HierarchicalFeatureExpander.from_category_set(
        category_set, alphas, n_features_in=n_features_in,
    )
    X_expanded = expander.expand_with_labels(X_reduced, labels_k)

    svc = LinearSVC(
        C=svc_C, max_iter=10_000,
        class_weight="balanced", dual="auto",
    )
    if skip_calibration:
        svc.fit(X_expanded, labels_k)
        classes_ = list(svc.classes_)
        X_pred_expanded = expander.expand_universal(X_reduced)
        scores = svc.decision_function(X_pred_expanded)
        pred_idx = (
            (scores > 0).astype(int) if scores.ndim == 1 else scores.argmax(axis=1)
        )
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

    pred_full: list[str | None] = [None] * len(labels)
    for ki, idx in enumerate(keep_idx):
        pred_full[idx] = pred_labels_k[ki]

    meta = {
        "n_rows_total": len(labels),
        "n_rows_kept": len(keep_idx),
        "n_singletons_dropped": dropped,
        "encoder": MODEL_ID,
        "encoder_dim": int(X.shape[1]),
        "svd_components": n_components,
        "expanded_dim": int(X_expanded.shape[1]),
        "n_classes": len(classes_),
        "calibration": "skipped" if skip_calibration else "sigmoid_cv",
    }
    return pred_full, meta


# ──────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────

def render_report(results: dict, rows: list[Row]) -> str:
    lines: list[str] = []
    lines.append("# NHSVM capacity reflection — ModernBERT-base encoder")
    lines.append("")
    lines.append(f"_Generated from {len(rows)} rows in `{AGENT_MEDIATED}`._")
    lines.append("")
    lines.append(
        "Same diagnostic shape as `reflect_nhsvm.py`, but the feature pipeline "
        "is **off-the-shelf ModernBERT-base** mean-pooled embeddings (no "
        "fine-tuning) instead of TF-IDF char+word ngrams.  Read the residual "
        "failures comparatively against the TF-IDF report — the falsifiable "
        "claim from `docs/src/architecture/classification.md:114-122` is that "
        "ModernBERT structurally separates cases TF-IDF systematically "
        "conflates (CPF↔date, sub-word token overlap)."
    )
    lines.append("")

    lines.append("## Grid: text_shape × knob_variant")
    lines.append("")
    lines.append(
        "L2-normalization is always applied (structurally required for "
        "L2-regularized LinearSVC).  All variants use ModernBERT-base."
    )
    lines.append("")
    lines.append(
        "| text_shape | knob | Pooling | SVD | Calibration | rows kept | "
        "top-1 on kept | top-1 overall |"
    )
    lines.append("|---|---|---|---|---|---:|---:|---:|")
    for v in results["variants"]:
        m = v["measure"]
        meta = v["meta"]
        svd = meta.get("svd_components") or "—"
        pooling = v["config"].get("pooling", "mean")
        cal = meta.get("calibration", "—")
        lines.append(
            f"| {v.get('text_shape', '?')} | {v.get('knob', v['name'])} | "
            f"{pooling} | {svd} | {cal} | "
            f"{m['n_kept']}/{m['n_total']} | "
            f"{m['top1_on_kept']:.4f} | {m['overall_top1']:.4f} |"
        )
    lines.append("")

    # Per-text-shape best (so the text-shape comparison is one-glance)
    lines.append("### Best per text shape")
    lines.append("")
    by_shape: dict[str, list[dict]] = {}
    for v in results["variants"]:
        by_shape.setdefault(v.get("text_shape", "?"), []).append(v)
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

    n_singletons = results.get("n_singletons", 0)
    lines.append(
        f"**Singleton drop**: {n_singletons} classes had only one example "
        f"and were filtered by NHSVM's `_min_class_count() < 2` guard."
    )
    lines.append("")

    baseline_failures = results["baseline_failures"]
    lines.append("## Baseline failure-mode breakdown")
    lines.append("")
    lines.append("| Failure kind | Count |")
    lines.append("|---|---:|")
    for k, v in sorted(baseline_failures["failure_kinds"].items(),
                       key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

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

    # Rescue-knob ranking — anchor on svm/baseline so deltas are
    # interpretable as "how much does this knob lift over the lean-text
    # default that mirrors reflect_nhsvm.py's baseline".
    if len(results["variants"]) > 1:
        anchor = next(
            (v for v in results["variants"]
             if v.get("text_shape") == TEXT_SHAPES[0]
             and v.get("knob") == "baseline"),
            results["variants"][0],
        )
        baseline_score = anchor["measure"]["top1_on_kept"]
        lines.append("## Rescue-knob ranking (vs svm/baseline)")
        lines.append("")
        lines.append(
            f"Anchor: **{anchor['name']}** top-1-on-kept = "
            f"**{baseline_score:.4f}**"
        )
        lines.append("")
        lines.append("| Variant | Δ top-1 on kept |")
        lines.append("|---|---:|")
        for v in results["variants"]:
            if v["name"] == anchor["name"]:
                continue
            delta = v["measure"]["top1_on_kept"] - baseline_score
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {v['name']} | {sign}{delta:.4f} |")
        lines.append("")

    lines.append("## Verdict (vs TF-IDF baseline)")
    lines.append("")
    best = max(results["variants"], key=lambda v: v["measure"]["top1_on_kept"])
    ceiling = (len(rows) - n_singletons) / len(rows) if len(rows) else 0
    tfidf_best = results.get("tfidf_reference_top1_kept")
    if tfidf_best is not None:
        delta_vs_tfidf = best["measure"]["top1_on_kept"] - tfidf_best
        sign = "+" if delta_vs_tfidf >= 0 else ""
        lines.append(
            f"Best ModernBERT variant: **{best['name']}**, "
            f"top-1 on kept = {best['measure']['top1_on_kept']:.4f}, "
            f"overall = {best['measure']['overall_top1']:.4f} "
            f"(theoretical ceiling: {ceiling:.4f}).  "
            f"TF-IDF best from `reflect_nhsvm.py`: {tfidf_best:.4f}.  "
            f"Δ vs TF-IDF: **{sign}{delta_vs_tfidf:.4f}**."
        )
    else:
        lines.append(
            f"Best variant: **{best['name']}**, "
            f"top-1 on kept = {best['measure']['top1_on_kept']:.4f}, "
            f"overall = {best['measure']['overall_top1']:.4f} "
            f"(theoretical ceiling: {ceiling:.4f}).  "
            f"Run `reflect_nhsvm.py` first for a TF-IDF reference number."
        )
    lines.append("")
    lines.append(
        "Headline accuracy is *one* signal — the more informative comparison "
        "is the **failure-mode shift**.  Open `reflect_nhsvm/report.md` and "
        "this one side-by-side; look for residuals that disappeared (TF-IDF "
        "artifacts ModernBERT resolved) and residuals that appeared (semantic "
        "boundaries ModernBERT smooths that TF-IDF kept sharp)."
    )
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Variant definitions
# ──────────────────────────────────────────────────────────────────────

# Knob configurations — each will be run against EACH text shape.
# Keep this list small; the grid multiplies by len(TEXT_SHAPES).
KNOB_BASELINE = {
    "svc_C": 1.0,
    "svd_components": None,
    "skip_calibration": False,
    "pooling": "mean",
}

KNOB_VARIANTS = [
    ("baseline", KNOB_BASELINE),
    ("no_cal", {**KNOB_BASELINE, "skip_calibration": True}),
    ("highC_nocal", {**KNOB_BASELINE, "svc_C": 100.0,
                     "skip_calibration": True}),
    ("cls_nocal", {**KNOB_BASELINE, "pooling": "cls",
                   "svc_C": 100.0, "skip_calibration": True}),
    ("svd200_nocal", {**KNOB_BASELINE, "svc_C": 100.0,
                      "svd_components": 200, "skip_calibration": True}),
    ("svd1000_nocal", {**KNOB_BASELINE, "svc_C": 100.0,
                       "svd_components": 1000, "skip_calibration": True}),
]

# Text shapes — second grid axis.  Each variant runs against both.
#   "svm"  → build_texts_and_labels (reflect_nhsvm): lean
#             `name | type | vals | siblings`
#   "rich" → build_rich_texts_and_labels (this file): full CatBoost-style
#             `ColumnFeatures.to_embedding_text()` with patterns,
#             ontology_priors, value_description, scalars, source_table
TEXT_SHAPES = ["svm", "rich"]


def _read_tfidf_reference() -> float | None:
    """Pull best top1_on_kept from the TF-IDF diagnostic's results.json
    (if present) so the report can quote the head-to-head delta."""
    tf_results = Path("build/reflect_nhsvm/results.json")
    if not tf_results.exists():
        return None
    try:
        data = json.loads(tf_results.read_text())
        return max(v["measure"]["top1_on_kept"] for v in data["variants"])
    except Exception as exc:
        log.warning("  could not parse TF-IDF reference: %s", exc)
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
    labels_canonical: list[str] | None = None
    for shape in TEXT_SHAPES:
        if shape == "svm":
            texts_s, labels_s = build_texts_and_labels(rows)
        elif shape == "rich":
            texts_s, labels_s = build_rich_texts_and_labels(rows)
        else:
            raise ValueError(f"unknown text shape: {shape}")
        text_sets[shape] = texts_s
        if labels_canonical is None:
            labels_canonical = labels_s
        else:
            assert labels_s == labels_canonical, \
                f"label mismatch between text shapes: {shape}"
        avg_len = sum(len(t) for t in texts_s) / max(len(texts_s), 1)
        log.info("  text_shape=%-4s  rows=%d  avg_chars=%.0f",
                 shape, len(texts_s), avg_len)
    labels = labels_canonical  # safe — assertion-guarded above
    code_counts = Counter(labels)
    singletons = [c for c, n in code_counts.items() if n < 2]
    log.info("  %d singleton classes (will be dropped at NHSVM fit time)",
             len(singletons))

    # Encode each (text_shape, pooling) once, cache to disk.  Only encode
    # CLS-pool if any variant in the sweep needs it.
    log.info("\n=== Phase 0b: ModernBERT encode (per text shape × pooling) ===")
    poolings_needed: set[str] = {"mean"}  # baseline always needs mean
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

    # ── Grid sweep: text_shape × knob_variant ──
    log.info("\n=== Phase 1+2: grid sweep (text_shape × knob_variant) ===")
    variants: list[dict] = []
    failures_baseline: dict | None = None  # captured from svm/baseline

    def _persist_partial():
        """Write what we have so far so a crash mid-sweep loses nothing."""
        RESULTS_JSON.write_text(json.dumps({
            "n_rows": len(rows),
            "n_singletons": len(singletons),
            "singleton_codes": sorted(singletons),
            "variants": variants,
            "baseline_failures": failures_baseline or {
                "failure_kinds": {}, "examples_top": [], "total_failures": 0,
            },
            "tfidf_reference_top1_kept": None,  # filled in at end
            "incomplete": True,
        }, indent=2))

    knobs_to_run = [KNOB_VARIANTS[0]] if args.quick else KNOB_VARIANTS
    for shape in TEXT_SHAPES:
        for name, cfg in knobs_to_run:
            full_name = f"{shape}/{name}"
            log.info("  variant: %s  (%s)", full_name, cfg)
            t0 = time.time()
            try:
                pool = cfg.get("pooling", "mean")
                X_in = encodings[(shape, pool)]
                pred_v, meta_v = fit_and_predict_dense(
                    X_in, labels, cat_set,
                    **{k: v for k, v in cfg.items() if k != "pooling"},
                )
                m_v = measure(pred_v, labels)
                log.info(
                    "    %.1fs  top-1 overall=%.4f  on-kept=%.4f",
                    time.time() - t0, m_v["overall_top1"], m_v["top1_on_kept"],
                )
                # Capture failure breakdown from the svm/baseline cell so
                # the report's residual analysis has something to anchor.
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

    tfidf_ref = _read_tfidf_reference()
    if tfidf_ref is not None:
        log.info("  TF-IDF reference (best top1_on_kept): %.4f", tfidf_ref)

    results = {
        "n_rows": len(rows),
        "n_singletons": len(singletons),
        "singleton_codes": sorted(singletons),
        "variants": variants,
        "baseline_failures": failures_baseline,
        "tfidf_reference_top1_kept": tfidf_ref,
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
    print(f"  theoretical ceiling: "
          f"{(len(rows)-len(singletons))/len(rows):.4f} "
          f"({len(singletons)} singletons dropped of {len(rows)} rows)")
    if tfidf_ref is not None:
        delta = best["measure"]["top1_on_kept"] - tfidf_ref
        sign = "+" if delta >= 0 else ""
        print(f"  TF-IDF reference: {tfidf_ref:.4f}  (Δ {sign}{delta:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
