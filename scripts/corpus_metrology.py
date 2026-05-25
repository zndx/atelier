#!/usr/bin/env python3
"""scripts/corpus_metrology.py — per-code metrology for synthetic corpus quality.

Computes diagnostic signals in the embedding space the factorized NHSVM
head actually uses, so the refinement loop and the generator-authoring
agent can reason about generator quality with quantitative feedback
instead of just "you got these wrong."

Signals computed per code:

  - fidelity_vs_reference:    1 - cos(synth_centroid, reference_centroid)
                              (low = synth shape resembles reference;
                              undefined for codes with no reference examples)
  - separability[neighbor]:   1 - cos(synth_centroid, neighbor_synth_centroid)
                              (high = clearly distinct from this neighbor)
  - spread:                   mean intra-code synth cosine / mean cos-to-nearest-neighbor
                              (< 1 = too tight, > 2 = too diffuse)
  - shape_exemplars:          up to 5 reference lean-text strings for this code
  - recommended_action:       improve_fidelity | improve_separability |
                              reduce_redundancy | abandon_structural | hold

Neighbors are sourced from per_category_accuracy.json's predictions
when available (confusion-driven); otherwise fall back to the top-K
nearest reference centroids (semantic similarity).

Output: build/svm_corpus_v2/metrology_report.json — keyed by code.

Usage:
  python scripts/corpus_metrology.py \\
      --corpus build/data/svm_training/corpus_v2 \\
      --output build/svm_corpus_v2/metrology_report.json

  # Pre-flight dry run: stdout summary only, no write
  python scripts/corpus_metrology.py --corpus ... --dry-run

  # Use validate-set predictions to source neighbors (post-Phase-D)
  python scripts/corpus_metrology.py --corpus ... \\
      --per-category-json build/reflect_nhsvm_eval_shap_v2/per_category_accuracy.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from sklearn.preprocessing import normalize as sk_normalize

from reflect_nhsvm import Row, load_rows, build_texts_and_labels
from reflect_nhsvm_eval_shap_v2 import encode_with_cache, corpus_hash
from reflect_nhsvm_eval_shap import stratified_split

log = logging.getLogger("corpus_metrology")


# Percentile-based action thresholds.  ModernBERT's general-purpose
# encoder produces high-baseline similarity (most text pairs cos > 0.85),
# so absolute centroid distances are not directly interpretable across
# codes — the load-bearing signal is comparative position within the
# population.  These percentiles were calibrated against the 2026-05-25
# corpus_v2 distribution; refinement may surface that they need
# adjustment as the encoder or corpus changes.
FIDELITY_BAD_PERCENTILE = 90  # fidelity-distance in top decile → shape mismatch
SEPARABILITY_BAD_PERCENTILE = 10  # sep_min in bottom decile → trapped near neighbor
SPREAD_HIGH_PERCENTILE = 90  # intra-code mean-sim in top decile → over-saturated
ACCURACY_TROUBLED = 0.50  # validate_accuracy below this → in-trouble code
ABANDON_STRUCTURAL_PASSES = 2  # passes_with_problem ≥ this → abandon


# ──────────────────────────────────────────────────────────────────────
# Centroid computation
# ──────────────────────────────────────────────────────────────────────

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    return sk_normalize(X, norm="l2", axis=1).astype(np.float32)


def _centroids_by_code(
    embeddings: np.ndarray, labels: list[str],
) -> dict[str, np.ndarray]:
    """Mean-pool L2-normalized embeddings per code; re-normalize the mean."""
    X = _l2_normalize(embeddings)
    by_code: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        by_code[l].append(i)
    out: dict[str, np.ndarray] = {}
    for code, idxs in by_code.items():
        c = X[idxs].mean(axis=0)
        n = np.linalg.norm(c) + 1e-12
        out[code] = (c / n).astype(np.float32)
    return out


def _intra_code_mean_sim(
    embeddings: np.ndarray, labels: list[str],
) -> dict[str, float]:
    """Mean pairwise cosine similarity within each code's synth set.

    Uses ``mean^T mean - 1/N`` identity for L2-normalized vectors:
    ``mean_i mean_j (x_i · x_j) = ||sum x_i||^2 / N^2``, less the
    diagonal contribution.  Avoids materializing the full pairwise
    matrix.
    """
    X = _l2_normalize(embeddings)
    by_code: dict[str, list[int]] = defaultdict(list)
    for i, l in enumerate(labels):
        by_code[l].append(i)
    out: dict[str, float] = {}
    for code, idxs in by_code.items():
        n = len(idxs)
        if n < 2:
            out[code] = float("nan")
            continue
        sub = X[idxs]
        sum_vec = sub.sum(axis=0)
        sum_sq = float(np.dot(sum_vec, sum_vec))
        # sum over i!=j of x_i.x_j = sum_sq - n  (since x_i.x_i = 1)
        off_diag_sum = sum_sq - n
        out[code] = off_diag_sum / (n * (n - 1))
    return out


# ──────────────────────────────────────────────────────────────────────
# Neighbor sourcing
# ──────────────────────────────────────────────────────────────────────

def _neighbors_from_predictions(
    per_category_json: Path,
) -> dict[str, list[tuple[str, int]]]:
    """For each true-code, what pred-codes did the model use, with counts?

    Returns: { true_code: [(pred_code, count), ...] }  excluding self.
    """
    if not per_category_json.exists():
        return {}
    data = json.loads(per_category_json.read_text())
    preds = data.get("predictions", [])
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in preds:
        true = rec["true"]
        pred = rec["pred"]
        if true == pred:
            continue
        confusion[true][pred] += 1
    out: dict[str, list[tuple[str, int]]] = {}
    for true, counts in confusion.items():
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])
        out[true] = ranked
    return out


def _neighbors_from_centroids(
    code: str,
    centroids: dict[str, np.ndarray],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Top-K nearest centroids in cosine space (semantic neighbors)."""
    if code not in centroids:
        return []
    target = centroids[code]
    sims: list[tuple[str, float]] = []
    for other, vec in centroids.items():
        if other == code:
            continue
        sim = float(np.dot(target, vec))
        sims.append((other, sim))
    sims.sort(key=lambda kv: -kv[1])
    return sims[:k]


# ──────────────────────────────────────────────────────────────────────
# Action derivation
# ──────────────────────────────────────────────────────────────────────

def _derive_action(
    *,
    fidelity_pct: float | None,
    separability_pct: float | None,
    spread_pct: float | None,
    accuracy: float | None,
    passes_with_problem: int,
) -> str:
    """Decision rule using percentile positions within the population.

    The metrology signals are RELATIVE: a code is "in trouble" when
    its accuracy is low AND one of its metrology signals is in the
    tail of the population distribution.  Codes that are accurate
    despite tail signals are left alone — the metrics matter for
    diagnosis, not as universal targets.

    Returns one of:
       improve_fidelity | improve_separability | reduce_redundancy |
       abandon_structural | hold
    """
    # Codes with no accuracy signal (e.g., no reference examples): hold
    # — the agent can't reason about them without confusion data.
    if accuracy is None:
        return "hold"

    in_trouble = accuracy < ACCURACY_TROUBLED
    if not in_trouble:
        return "hold"

    bad_fidelity = (fidelity_pct is not None
                    and fidelity_pct >= FIDELITY_BAD_PERCENTILE)
    bad_separability = (separability_pct is not None
                        and separability_pct <= SEPARABILITY_BAD_PERCENTILE)
    high_spread = (spread_pct is not None
                   and spread_pct >= SPREAD_HIGH_PERCENTILE)

    if bad_fidelity and bad_separability and passes_with_problem >= ABANDON_STRUCTURAL_PASSES:
        return "abandon_structural"
    if bad_separability and passes_with_problem >= ABANDON_STRUCTURAL_PASSES:
        # In-trouble + persistently inseparable from neighbors = structural
        return "abandon_structural"
    if bad_fidelity:
        return "improve_fidelity"
    if bad_separability:
        return "improve_separability"
    if high_spread:
        return "reduce_redundancy"
    # In-trouble but no metrology signal explains it — likely insufficient
    # synth volume for this code, or a structural-but-not-extreme issue.
    return "improve_separability"


def _percentile_rank(value: float, sorted_pop: list[float]) -> float:
    """Returns the percentile (0..100) of `value` within sorted_pop.
    100 = larger than everything; 0 = smaller than everything."""
    if not sorted_pop:
        return 50.0
    import bisect
    idx = bisect.bisect_left(sorted_pop, value)
    return (idx / len(sorted_pop)) * 100.0


# ──────────────────────────────────────────────────────────────────────
# Shape exemplars
# ──────────────────────────────────────────────────────────────────────

def _shape_exemplars_by_code(
    reference_rows: list[Row], reference_texts: list[str],
    max_per_code: int = 5,
) -> dict[str, list[str]]:
    """Up to N reference lean-text strings per code."""
    out: dict[str, list[str]] = defaultdict(list)
    for r, t in zip(reference_rows, reference_texts):
        if len(out[r.code]) < max_per_code:
            out[r.code].append(t)
    return dict(out)


# ──────────────────────────────────────────────────────────────────────
# Top-level
# ──────────────────────────────────────────────────────────────────────

def _phase_d_reference_split(real_rows: list[Row]) -> tuple[
    list[Row], list[str], list[Row], list[str]
]:
    """Reproduce Phase D's reference split (seed=42, 80/20 stratified)
    so we can reuse the existing aug-train cache entries that encoded
    rows in this order.  Returns (real_train_rows, real_train_labels,
    real_test_rows, real_test_labels).

    The split is used purely for cache-key alignment.  For metrology,
    we then re-concatenate train+test as the full reference set.
    """
    from collections import Counter as _Counter
    _, real_labels = build_texts_and_labels(real_rows)
    code_counts = _Counter(real_labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    trainable_mask = [l not in singletons for l in real_labels]
    trainable_rows = [r for r, m in zip(real_rows, trainable_mask) if m]
    trainable_labels = [l for l, m in zip(real_labels, trainable_mask) if m]
    train_idx, test_idx = stratified_split(
        trainable_labels, test_size=0.2, seed=42,
    )
    rt_rows = [trainable_rows[i] for i in train_idx]
    rt_labels = [trainable_labels[i] for i in train_idx]
    re_rows = [trainable_rows[i] for i in test_idx]
    re_labels = [trainable_labels[i] for i in test_idx]
    return rt_rows, rt_labels, re_rows, re_labels


def compute_metrology(
    *,
    corpus_dir: Path,
    output_path: Path,
    per_category_json: Path | None,
    abandon_history_path: Path | None,
    batch_size: int = 64,
    refresh_embeddings: bool = False,
    dry_run: bool = False,
) -> dict:
    """Compute per-code metrology against the full reference set.

    Strategy: reuse Phase D's cache keys so dry-runs against existing
    corpus_v2 hit cache cleanly (the aug-train cache contains synth +
    real_train concatenated in known order; the real-test cache holds
    the held-out 271).  Slicing reconstructs the full reference set
    (real_train ∪ real_test) without re-encoding.

    Returns the report dict.  Writes to output_path unless dry_run.
    """
    # 1. Load corpus + reference
    log.info("Loading synth corpus from %s", corpus_dir)
    synth_rows_path = corpus_dir / "synth_rows.jsonl"
    if not synth_rows_path.exists():
        raise FileNotFoundError(f"Synth corpus missing: {synth_rows_path}")
    synth_rows: list[Row] = []
    with synth_rows_path.open() as f:
        for line in f:
            d = json.loads(line)
            synth_rows.append(Row(
                table=d["table"], column=d["column"],
                column_type=d["column_type"],
                sample_values=d["sample_values"],
                siblings_full=d["siblings_full"],
                mnemonic=d["mnemonic"], code=d["code"],
            ))
    log.info("  %d synth rows loaded", len(synth_rows))

    log.info("Loading reference set (full, used as validate)...")
    real_rows = load_rows(refresh_cache=False, database="reference_corpus")
    log.info("  %d reference rows loaded", len(real_rows))

    # Reproduce Phase D's split so cache keys align
    real_train_rows, real_train_labels, real_test_rows, real_test_labels = \
        _phase_d_reference_split(real_rows)
    log.info("  reference split (Phase-D-aligned): train=%d  test=%d",
             len(real_train_rows), len(real_test_rows))

    # 2. Build lean_text for aug-train (synth + real_train) and test
    log.info("Building lean texts...")
    synth_texts, synth_labels = build_texts_and_labels(synth_rows)
    aug_rows = synth_rows + real_train_rows
    aug_texts, aug_labels = build_texts_and_labels(aug_rows)
    test_texts, _ = build_texts_and_labels(real_test_rows)

    # 3. Encode using Phase D's cache keys — this reuses existing cache
    #    cleanly when corpus_v2 hasn't changed since the last Phase D run.
    log.info("Encoding aug-train via Phase D cache key...")
    aug_emb = encode_with_cache(
        aug_texts, cache_key=f"aug-train-{corpus_hash()}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  aug-train embeddings: %s", aug_emb.shape)

    log.info("Encoding real-test via Phase D cache key...")
    test_emb = encode_with_cache(
        test_texts, cache_key="real-test",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  real-test embeddings: %s", test_emb.shape)

    # 4. Slice aug-train into synth + real_train; concat with real-test
    #    to form the FULL reference embedding (per the validate framing).
    n_synth = len(synth_rows)
    synth_emb = aug_emb[:n_synth]
    real_train_emb = aug_emb[n_synth:]
    log.info("  sliced: synth=%s  real_train=%s",
             synth_emb.shape, real_train_emb.shape)

    # Full reference = real_train + real_test (validate dataset)
    ref_emb = np.concatenate([real_train_emb, test_emb], axis=0)
    real_labels = list(real_train_labels) + list(real_test_labels)
    log.info("  full reference embeddings: %s (train %d + test %d)",
             ref_emb.shape, len(real_train_labels), len(real_test_labels))

    # 4. Centroids per code
    log.info("Computing per-code centroids...")
    synth_centroids = _centroids_by_code(synth_emb, synth_labels)
    ref_centroids = _centroids_by_code(ref_emb, real_labels)
    log.info("  synth: %d codes  reference: %d codes",
             len(synth_centroids), len(ref_centroids))

    # 5. Intra-code mean sim (for spread)
    intra_sim = _intra_code_mean_sim(synth_emb, synth_labels)

    # 6. Neighbors
    pred_neighbors = (_neighbors_from_predictions(per_category_json)
                      if per_category_json and per_category_json.exists()
                      else {})
    if pred_neighbors:
        log.info("  neighbors sourced from predictions in %s",
                 per_category_json)
    else:
        log.info("  neighbors sourced from top-K nearest reference centroids")

    # 7. Shape exemplars — built from the full reference (train ∪ test)
    full_ref_rows = list(real_train_rows) + list(real_test_rows)
    full_ref_texts = list(build_texts_and_labels(full_ref_rows)[0])
    exemplars = _shape_exemplars_by_code(
        full_ref_rows, full_ref_texts, max_per_code=5,
    )

    # 8. Abandon-history (from prior passes) — informs passes_with_problem
    abandon_history: dict[str, int] = {}
    if abandon_history_path and abandon_history_path.exists():
        abandon_history = json.loads(abandon_history_path.read_text())

    # 9. Validate-accuracy + confusion counts from per_category_json
    validate_accuracy: dict[str, float] = {}
    n_validate: dict[str, int] = {}
    if per_category_json and per_category_json.exists():
        pc_data = json.loads(per_category_json.read_text())
        for code, info in pc_data.get("per_category", {}).items():
            validate_accuracy[code] = info.get("accuracy", 0.0)
            n_validate[code] = info.get("n_test", 0)

    # 10. First pass: compute raw metrics for every code (no actions yet,
    #     because actions depend on percentile rankings across population).
    raw: dict[str, dict] = {}
    for code in sorted(synth_centroids.keys()):
        synth_c = synth_centroids[code]
        ref_c = ref_centroids.get(code)

        # Fidelity
        if ref_c is not None:
            fidelity = float(1.0 - np.dot(synth_c, ref_c))
        else:
            fidelity = None

        # Neighbors + separability
        if code in pred_neighbors:
            neigh_codes = [n for n, _ in pred_neighbors[code]
                           if n in synth_centroids][:5]
            neigh_source = "predictions"
        else:
            if ref_c is not None:
                cand = _neighbors_from_centroids(code, ref_centroids, k=5)
                neigh_codes = [n for n, _ in cand if n in synth_centroids]
                neigh_source = "reference_centroids"
            else:
                cand = _neighbors_from_centroids(code, synth_centroids, k=5)
                neigh_codes = [n for n, _ in cand]
                neigh_source = "synth_centroids"

        separability: dict[str, float] = {}
        for n in neigh_codes:
            dist = float(1.0 - np.dot(synth_c, synth_centroids[n]))
            separability[n] = dist
        sep_min = min(separability.values()) if separability else None
        sep_nearest = (min(separability.items(), key=lambda kv: kv[1])[0]
                       if separability else None)

        spread = intra_sim.get(code)
        spread_val = None if (spread is None or spread != spread) else float(spread)

        raw[code] = {
            "fidelity": fidelity,
            "separability": separability,
            "sep_min": sep_min,
            "sep_nearest": sep_nearest,
            "spread": spread_val,
            "neigh_source": neigh_source,
        }

    # 11. Build sorted populations for percentile lookup
    fid_pop = sorted([r["fidelity"] for r in raw.values()
                      if r["fidelity"] is not None])
    sep_pop = sorted([r["sep_min"] for r in raw.values()
                      if r["sep_min"] is not None])
    spread_pop = sorted([r["spread"] for r in raw.values()
                          if r["spread"] is not None])
    log.info("  population stats: fid n=%d, sep n=%d, spread n=%d",
             len(fid_pop), len(sep_pop), len(spread_pop))

    # 12. Second pass: assemble final report with percentile + action
    log.info("Computing per-code signals + actions...")
    report: dict[str, dict] = {}
    for code in sorted(synth_centroids.keys()):
        r = raw[code]
        fid_pct = (_percentile_rank(r["fidelity"], fid_pop)
                   if r["fidelity"] is not None else None)
        sep_pct = (_percentile_rank(r["sep_min"], sep_pop)
                   if r["sep_min"] is not None else None)
        spread_pct = (_percentile_rank(r["spread"], spread_pop)
                      if r["spread"] is not None else None)

        accuracy = validate_accuracy.get(code)
        passes_with_problem = int(abandon_history.get(code, 0))
        action = _derive_action(
            fidelity_pct=fid_pct,
            separability_pct=sep_pct,
            spread_pct=spread_pct,
            accuracy=accuracy,
            passes_with_problem=passes_with_problem,
        )

        report[code] = {
            "n_synth": int(sum(1 for l in synth_labels if l == code)),
            "n_reference": int(sum(1 for l in real_labels if l == code)),
            "n_validate": n_validate.get(code, 0),
            "validate_accuracy": accuracy,
            "fidelity_vs_reference": (round(r["fidelity"], 4)
                                      if r["fidelity"] is not None else None),
            "fidelity_pct": (round(fid_pct, 1) if fid_pct is not None else None),
            "separability": {n: round(d, 4) for n, d in r["separability"].items()},
            "separability_min": (round(r["sep_min"], 4)
                                  if r["sep_min"] is not None else None),
            "separability_nearest_neighbor": r["sep_nearest"],
            "separability_pct": (round(sep_pct, 1) if sep_pct is not None else None),
            "intra_code_mean_sim": (round(r["spread"], 4)
                                     if r["spread"] is not None else None),
            "spread_pct": (round(spread_pct, 1) if spread_pct is not None else None),
            "shape_exemplars_from_reference": exemplars.get(code, []),
            "neighbor_source": r["neigh_source"],
            "passes_with_problem": passes_with_problem,
            "recommended_action": action,
        }

    # 10. Summary
    by_action: dict[str, int] = defaultdict(int)
    for entry in report.values():
        by_action[entry["recommended_action"]] += 1
    summary = {
        "n_codes_synth": len(synth_centroids),
        "n_codes_reference": len(ref_centroids),
        "n_codes_with_fidelity": sum(
            1 for e in report.values()
            if e["fidelity_vs_reference"] is not None
        ),
        "action_counts": dict(by_action),
        "neighbor_source_counts": dict(defaultdict(int, {
            "predictions": sum(1 for e in report.values()
                                if e["neighbor_source"] == "predictions"),
            "reference_centroids": sum(1 for e in report.values()
                                        if e["neighbor_source"] == "reference_centroids"),
            "synth_centroids": sum(1 for e in report.values()
                                    if e["neighbor_source"] == "synth_centroids"),
        })),
    }

    out = {
        "summary": summary,
        "thresholds": {
            "fidelity_bad_percentile": FIDELITY_BAD_PERCENTILE,
            "separability_bad_percentile": SEPARABILITY_BAD_PERCENTILE,
            "spread_high_percentile": SPREAD_HIGH_PERCENTILE,
            "accuracy_troubled": ACCURACY_TROUBLED,
            "abandon_structural_passes": ABANDON_STRUCTURAL_PASSES,
        },
        "per_code": report,
    }

    if dry_run:
        log.info("Dry-run summary:")
        log.info("  %s", json.dumps(summary, indent=2))
        # Also surface a few specific codes the manual diagnostic flagged
        chronic = ["1.1.1.4.2.1.1", "1.1.1.4.2.2.1", "1.1.1.4.2.3.1",
                   "0.2", "1.1.2", "1.1.1.6", "1.2.6.1", "1.1.1.8.6"]
        log.info("Chronic-code spot-check (pct = percentile in population):")
        log.info("  %-20s  %-12s  %-12s  %-12s  %-6s  %-6s  %s",
                 "code", "fid (pct)", "sep (pct)", "spread(pct)",
                 "n_ref", "acc", "action")
        for c in chronic:
            if c in report:
                e = report[c]
                fid_str = (f"{e['fidelity_vs_reference']:.3f}({e['fidelity_pct']:.0f})"
                           if e['fidelity_vs_reference'] is not None else "n/a")
                sep_str = (f"{e['separability_min']:.3f}({e['separability_pct']:.0f})"
                           if e['separability_min'] is not None else "n/a")
                sp_str  = (f"{e['intra_code_mean_sim']:.3f}({e['spread_pct']:.0f})"
                           if e['intra_code_mean_sim'] is not None else "n/a")
                acc_str = (f"{e['validate_accuracy']:.2f}"
                           if e['validate_accuracy'] is not None else "n/a")
                log.info("  %-20s  %-12s  %-12s  %-12s  %-6d  %-6s  %s",
                         c, fid_str, sep_str, sp_str,
                         e['n_reference'], acc_str, e['recommended_action'])
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(out, indent=2))
        log.info("Wrote %s", output_path)

    return out


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus", type=Path,
                    default=Path("build/data/svm_training/corpus_v2"),
                    help="Corpus directory containing synth_rows.jsonl")
    ap.add_argument("--output", type=Path,
                    default=Path("build/svm_corpus_v2/metrology_report.json"),
                    help="Output JSON path")
    ap.add_argument("--per-category-json", type=Path, default=None,
                    help="per_category_accuracy.json for neighbor sourcing "
                         "from validate-set predictions (optional)")
    ap.add_argument("--abandon-history", type=Path, default=None,
                    help="JSON {code: passes_with_problem} for abandon "
                         "decision (optional)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print summary, do not write output JSON")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    compute_metrology(
        corpus_dir=args.corpus,
        output_path=args.output,
        per_category_json=args.per_category_json,
        abandon_history_path=args.abandon_history,
        batch_size=args.batch_size,
        refresh_embeddings=args.refresh_embeddings,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
