#!/usr/bin/env python3
"""scripts/svm_target_health.py — Post-integration diagnostic on hive-poc.

**Not the deployment gate.**  The deployment gate is
``scripts/svm_cosine_uplift_gate.py`` (cosine ⊕ SVM mutual-affirmation
on the full reference set).  This script runs AFTER the gate passes
and the SVM channel is wired into the runtime Atelier classification
pipeline; it surfaces concerns the gate didn't catch by sampling
broader hive-poc target entities — most of which are NOT in the
reference set, so accuracy can only be spot-checked via reference-
overlap rows.

What this script measures (all informational, no gating semantics):

  - **Code coverage**: distinct top-1 codes predicted across target
    entities, compared against the reference set's code distribution.
    Collapse to a handful or scatter across most of the taxonomy are
    both concerning.
  - **Confidence distribution**: histogram of (top1_score - top2_score)
    margin per target entity.  Bimodal (clearly-confident + clearly-
    uncertain bins) is healthy; uniform low margin is unconfident
    everywhere.
  - **Per-table consistency**: for each table, fraction of column
    predictions sharing the same root-level code.  Tables with
    semantic cohesion (an addresses table, a transactions table)
    should cluster.
  - **Mass concentration**: top1_score / (top1+top2+top3) per entity.
    Signal for DST fusion sanity.
  - **Reference-overlap sanity**: for target entities IN the reference
    set, fraction where SVM top-1 matches the label.  Should
    approximately equal the validate accuracy from Phase D.

Usage:
  python scripts/svm_target_health.py
  python scripts/svm_target_health.py --connection hive-poc --database reference_corpus
  python scripts/svm_target_health.py --n-tables 50 --n-cols-per-table 20
  python scripts/svm_target_health.py --full
  python scripts/svm_target_health.py --corpus-dir build/data/svm_training/corpus_v3
"""
from __future__ import annotations

import argparse
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
from sklearn.preprocessing import normalize as sk_normalize

from reflect_nhsvm import (
    Row, build_texts_and_labels, build_category_set, _pull_hive_table,
    AGENT_MEDIATED,
)
from reflect_nhsvm_eval_shap_v2 import (
    encode_with_cache, corpus_hash, load_synth_rows, REPORT_DIR,
)
from reflect_nhsvm_eval_shap import BEST_KNOB
from atelier.classify.factorized_nhsvm import fit_factorized_nhsvm

log = logging.getLogger("svm_target_health")

DEFAULT_CONNECTION = "hive-poc"
DEFAULT_DATABASE = "reference_corpus"
DEFAULT_CORPUS_DIR = Path("build/data/svm_training/corpus_v2")
OUTPUT_PATH = REPORT_DIR / "target_health_report.json"
TARGET_CACHE = Path("build/svm_target_health/target_cache.json")

# Reference-overlap watermark: informational expectation for what
# validate accuracy should look like on the labeled subset of hive-poc.
# Surfaces a soft "concern" flag when overlap accuracy diverges
# substantially from the prior Phase D number — it does NOT gate
# deployment.  The gate is svm_cosine_uplift_gate.py.
REFERENCE_OVERLAP_WATERMARK = 0.50  # below this, surface as concern


# ──────────────────────────────────────────────────────────────────────
# Target-data acquisition
# ──────────────────────────────────────────────────────────────────────

def _list_tables(conn, database: str) -> list[str]:
    """Enumerate tables in the target database."""
    df = conn.get_pandas_dataframe(f"show tables in {database}")
    if df.empty:
        return []
    # Column name varies by Hive variant; take the first column
    col = df.columns[0]
    return sorted(df[col].astype(str).tolist())


def _pull_target_columns(
    *,
    connection_name: str,
    database: str,
    n_tables: int | None,
    n_cols_per_table: int | None,
    full: bool,
    refresh_cache: bool,
) -> list[Row]:
    """Pull target columns from the hive-poc database, possibly cached."""
    if TARGET_CACHE.exists() and not refresh_cache:
        cache = json.loads(TARGET_CACHE.read_text())
        if (cache.get("connection") == connection_name
                and cache.get("database") == database
                and cache.get("n_tables") == n_tables
                and cache.get("n_cols_per_table") == n_cols_per_table
                and cache.get("full") == full):
            log.info("  reusing cache %s (matches scope)", TARGET_CACHE)
            return [Row(**r) for r in cache["rows"]]

    import cml.data_v1 as cmldata
    conn = cmldata.get_connection(connection_name)

    log.info("Listing tables in %s.%s ...", connection_name, database)
    all_tables = _list_tables(conn, database)
    log.info("  %d tables found", len(all_tables))

    if full:
        tables = all_tables
    elif n_tables is None:
        tables = all_tables
    else:
        # Stratified-by-name (alphabetical) sampling for determinism
        if len(all_tables) <= n_tables:
            tables = all_tables
        else:
            stride = len(all_tables) / n_tables
            tables = [all_tables[int(i * stride)] for i in range(n_tables)]
    log.info("  pulling %d tables", len(tables))

    rows: list[Row] = []
    t0 = time.time()
    for i, t in enumerate(tables, 1):
        try:
            tdata = _pull_hive_table(conn, database, t)
        except Exception as exc:  # noqa: BLE001
            log.warning("  [%d/%d] %s failed: %s", i, len(tables), t, exc)
            continue
        cols = tdata["columns"]
        if n_cols_per_table is not None:
            cols = cols[:n_cols_per_table]
        for c in cols:
            rows.append(Row(
                table=t, column=c,
                column_type=tdata["types"].get(c, ""),
                sample_values=tdata["samples"].get(c, []),
                siblings_full=tdata["columns"],  # full list for centered window
                mnemonic="",  # unknown for target columns
                code="",      # unknown — this is the prediction target
            ))
        if i % 10 == 0:
            log.info("  [%d/%d] tables done  %d cols  %.1fs",
                     i, len(tables), len(rows), time.time() - t0)

    log.info("Pulled %d target columns from %d tables in %.1fs",
             len(rows), len(tables), time.time() - t0)

    # Cache
    TARGET_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TARGET_CACHE.write_text(json.dumps({
        "connection": connection_name,
        "database": database,
        "n_tables": n_tables,
        "n_cols_per_table": n_cols_per_table,
        "full": full,
        "rows": [{
            "table": r.table, "column": r.column,
            "column_type": r.column_type, "sample_values": r.sample_values,
            "siblings_full": r.siblings_full, "mnemonic": r.mnemonic,
            "code": r.code,
        } for r in rows],
    }, indent=2))
    return rows


# ──────────────────────────────────────────────────────────────────────
# Reference-overlap sanity
# ──────────────────────────────────────────────────────────────────────

def _load_reference_labels() -> dict[str, str]:
    """Load {table.column: code} from the agent-mediated reference."""
    am = json.loads(AGENT_MEDIATED.read_text())
    out: dict[str, str] = {}
    for key, entry in am.items():
        if not isinstance(entry, dict):
            continue
        out[key] = entry["code"]
    return out


# ──────────────────────────────────────────────────────────────────────
# Health metrics
# ──────────────────────────────────────────────────────────────────────

def _confidence_distribution(
    pred_top1: list[str], top1_score: np.ndarray, top2_score: np.ndarray,
) -> dict:
    """Top1-top2 margin distribution."""
    margins = (top1_score - top2_score).tolist()
    bins = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, float("inf")]
    labels = ["<0.01", "0.01-0.05", "0.05-0.1", "0.1-0.2",
              "0.2-0.5", "0.5-1.0", ">1.0"]
    counts = [0] * (len(bins) - 1)
    for m in margins:
        for i in range(len(bins) - 1):
            if bins[i] <= m < bins[i + 1]:
                counts[i] += 1
                break
    return {
        "margin_mean": float(np.mean(margins)) if margins else 0.0,
        "margin_median": float(np.median(margins)) if margins else 0.0,
        "histogram": [{"bin": labels[i], "count": counts[i]}
                       for i in range(len(labels))],
    }


def _per_table_root_consistency(
    rows: list[Row], pred_top1: list[str],
) -> dict:
    """For each table, fraction of predictions sharing the most-common
    root-level code (top-level dotted segment).
    """
    by_table: dict[str, list[str]] = defaultdict(list)
    for r, p in zip(rows, pred_top1):
        by_table[r.table].append(p.split(".", 1)[0])
    per_table: dict[str, dict] = {}
    consistencies: list[float] = []
    for table, roots in by_table.items():
        if not roots:
            continue
        c = Counter(roots)
        most_common, n_most = c.most_common(1)[0]
        consistency = n_most / len(roots)
        per_table[table] = {
            "n_cols": len(roots),
            "most_common_root": most_common,
            "consistency": round(consistency, 4),
        }
        consistencies.append(consistency)
    return {
        "mean_consistency": (round(float(np.mean(consistencies)), 4)
                              if consistencies else 0.0),
        "median_consistency": (round(float(np.median(consistencies)), 4)
                                if consistencies else 0.0),
        "per_table": per_table,
    }


def _reference_overlap_sanity(
    rows: list[Row], pred_top1: list[str], ref_labels: dict[str, str],
) -> dict:
    """For target columns that exist in the reference, fraction where
    SVM top-1 matches the reference code.
    """
    n_overlap = 0
    n_match = 0
    misses: list[dict] = []
    for r, p in zip(rows, pred_top1):
        key = f"{r.table}.{r.column}"
        if key not in ref_labels:
            continue
        n_overlap += 1
        true_code = ref_labels[key]
        if p == true_code:
            n_match += 1
        elif len(misses) < 30:
            misses.append({"key": key, "true": true_code, "pred": p})
    return {
        "n_target_in_reference": n_overlap,
        "n_match": n_match,
        "match_rate": (round(n_match / n_overlap, 4)
                       if n_overlap else None),
        "sample_misses": misses,
    }


def _code_coverage(
    pred_top1: list[str], ref_labels: dict[str, str], cat_set,
) -> dict:
    """Distinct code coverage vs reference distribution."""
    pred_counts = Counter(pred_top1)
    ref_counts = Counter(ref_labels.values())
    total_codes = len(list(cat_set.all_categories))
    return {
        "n_distinct_predicted": len(pred_counts),
        "n_distinct_reference": len(ref_counts),
        "n_taxonomy_total": total_codes,
        "predicted_top_10": pred_counts.most_common(10),
        "reference_top_10": ref_counts.most_common(10),
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--connection", default=DEFAULT_CONNECTION)
    ap.add_argument("--database", default=DEFAULT_DATABASE)
    ap.add_argument("--n-tables", type=int, default=None,
                    help="Number of tables to sample (default: all)")
    ap.add_argument("--n-cols-per-table", type=int, default=None,
                    help="Cap columns per table (default: all)")
    ap.add_argument("--full", action="store_true",
                    help="Enumerate every table in the database")
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Synthetic corpus to train on (best-pass corpus)")
    ap.add_argument("--refresh-target-cache", action="store_true")
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    ap.add_argument("--overlap-watermark", type=float,
                    default=REFERENCE_OVERLAP_WATERMARK,
                    help=f"Informational watermark for reference-overlap "
                         f"accuracy (default {REFERENCE_OVERLAP_WATERMARK:.2f}); "
                         f"below this, a 'concern' flag surfaces in the "
                         f"report.  Does NOT gate deployment (the gate is "
                         f"scripts/svm_cosine_uplift_gate.py).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    log.info("=== svm_target_health (test stage) ===")

    # 1. Load synth corpus + train factorized head (replicates Phase D's
    #    training; head isn't persisted to disk by Phase D, so we re-run)
    log.info("Training factorized NHSVM head on %s ...", args.corpus_dir)
    cat_set = build_category_set()
    synth_rows = load_synth_rows(args.corpus_dir)
    synth_labels = [r.code for r in synth_rows]
    synth_counts = Counter(synth_labels)
    keep = [i for i, l in enumerate(synth_labels) if synth_counts[l] >= 2]
    train_rows = [synth_rows[i] for i in keep]
    train_labels = [synth_labels[i] for i in keep]
    log.info("  train: %d synth rows  %d distinct codes",
             len(train_rows), len(set(train_labels)))

    train_texts, _ = build_texts_and_labels(train_rows)
    log.info("Encoding training set...")
    X_train = encode_with_cache(
        train_texts, cache_key=f"synth-only-{corpus_hash(args.corpus_dir)}",
        refresh=args.refresh_embeddings, batch_size=args.batch_size,
    )

    log.info("Training factorized head (this is the best-pass training)...")
    head, train_result = fit_factorized_nhsvm(
        X_train, train_labels, cat_set,
        **BEST_KNOB, verbose=True, eval_every=50,
    )
    log.info("  trained: fit-acc=%.4f", train_result.final_train_acc)

    # 2. Pull target columns
    target_rows = _pull_target_columns(
        connection_name=args.connection,
        database=args.database,
        n_tables=args.n_tables,
        n_cols_per_table=args.n_cols_per_table,
        full=args.full,
        refresh_cache=args.refresh_target_cache,
    )
    if not target_rows:
        log.error("No target columns pulled — aborting")
        return 1

    # 3. Encode target columns
    log.info("Building lean texts for target columns...")
    target_texts, _ = build_texts_and_labels(target_rows)
    log.info("Encoding target columns...")
    X_target = encode_with_cache(
        target_texts,
        cache_key=f"target-{args.connection}-{args.database}-{len(target_rows)}",
        refresh=args.refresh_embeddings, batch_size=args.batch_size,
    )

    # 4. Predict + extract top-K scores
    log.info("Predicting on target columns...")
    X_target_norm = sk_normalize(X_target, norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    with torch.no_grad():
        X_t = torch.tensor(X_target_norm, device=device)
        scores = head(X_t)  # (N, n_nodes)
        # Top-3 per row
        top3 = torch.topk(scores, k=3, dim=1)
        top_idx = top3.indices.cpu().numpy()
        top_val = top3.values.cpu().numpy()
    pred_top1 = [head.codes[i] for i in top_idx[:, 0]]
    top1_score = top_val[:, 0]
    top2_score = top_val[:, 1]
    top3_score = top_val[:, 2]

    # 5. Health metrics
    log.info("Computing health metrics...")
    ref_labels = _load_reference_labels()

    coverage = _code_coverage(pred_top1, ref_labels, cat_set)
    confidence = _confidence_distribution(pred_top1, top1_score, top2_score)
    per_table = _per_table_root_consistency(target_rows, pred_top1)
    overlap = _reference_overlap_sanity(target_rows, pred_top1, ref_labels)

    # Mass concentration: top1 / sum(top3) per row
    mass_conc = (top1_score / (top1_score + top2_score + top3_score
                                 + 1e-12)).tolist()
    mass_summary = {
        "mean": round(float(np.mean(mass_conc)), 4),
        "median": round(float(np.median(mass_conc)), 4),
        "p25": round(float(np.percentile(mass_conc, 25)), 4),
        "p75": round(float(np.percentile(mass_conc, 75)), 4),
    }

    # Informational watermark check — NOT a deployment gate.
    overlap_rate = overlap.get("match_rate")
    if overlap_rate is None:
        watermark_status = "indeterminate"
        watermark_reason = ("no overlap between target entities and "
                             "reference set; can't measure accuracy")
    elif overlap_rate >= args.overlap_watermark:
        watermark_status = "above_watermark"
        watermark_reason = (f"reference-overlap accuracy {overlap_rate:.4f} "
                             f"≥ {args.overlap_watermark:.2f} watermark")
    else:
        watermark_status = "concern_below_watermark"
        watermark_reason = (f"reference-overlap accuracy {overlap_rate:.4f} "
                             f"< {args.overlap_watermark:.2f} — concern flag; "
                             f"deployment gate (cosine-SVM uplift) is the "
                             f"binding decision, but this divergence from "
                             f"the expected validate baseline is worth "
                             f"investigating")

    report = {
        "connection": args.connection,
        "database": args.database,
        "corpus_dir": str(args.corpus_dir),
        "corpus_hash": corpus_hash(args.corpus_dir),
        "n_target_columns": len(target_rows),
        "n_target_tables": len({r.table for r in target_rows}),
        "train": {
            "n_synth": len(train_rows),
            "n_distinct_train_codes": len(set(train_labels)),
            "fit_acc": round(train_result.final_train_acc, 4),
        },
        "code_coverage": coverage,
        "confidence_distribution": confidence,
        "per_table_consistency": per_table,
        "mass_concentration": mass_summary,
        "reference_overlap_sanity": overlap,
        "overlap_watermark_check": {
            "watermark": args.overlap_watermark,
            "observed_overlap_accuracy": overlap_rate,
            "status": watermark_status,
            "reason": watermark_reason,
            "note": ("This is informational only — the deployment gate "
                      "is scripts/svm_cosine_uplift_gate.py, not this "
                      "watermark."),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", args.output)

    # Console summary
    print()
    print("=== Target-data-source health (test stage) ===")
    print(f"  target columns:        {report['n_target_columns']} across "
          f"{report['n_target_tables']} tables")
    print(f"  code coverage:         {coverage['n_distinct_predicted']} distinct "
          f"predicted (taxonomy size {coverage['n_taxonomy_total']}, "
          f"reference has {coverage['n_distinct_reference']})")
    print(f"  confidence median margin: {confidence['margin_median']:.4f}")
    print(f"  per-table root consistency (median): "
          f"{per_table['median_consistency']:.4f}")
    print(f"  mass concentration (median top1/sum-top3): "
          f"{mass_summary['median']:.4f}")
    if overlap['n_target_in_reference']:
        print(f"  reference-overlap sanity: {overlap['match_rate']:.4f} "
              f"({overlap['n_match']}/{overlap['n_target_in_reference']})")
    else:
        print(f"  reference-overlap sanity: no overlap to check")
    print()
    badge = {
        "above_watermark":           "  above watermark",
        "concern_below_watermark":   "⚠  below watermark — concern flag",
        "indeterminate":             "?  indeterminate (no overlap)",
    }[watermark_status]
    print(f"  {badge}")
    print(f"  {watermark_reason}")
    print()
    print("  Note: this script is a POST-INTEGRATION DIAGNOSTIC.")
    print("  The deployment gate is scripts/svm_cosine_uplift_gate.py.")
    # Always return 0 — this script is informational, not gating.
    return 0


if __name__ == "__main__":
    sys.exit(main())
