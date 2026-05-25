#!/usr/bin/env python3
"""scripts/svm_cosine_uplift_gate.py — Gate B of the SVM stage's dual-gate exit.

`just optimize svm` exits when BOTH gates pass:
  Gate A (quantitative steering): refinement loop's full-validate top-1
         reaches TARGET_ACCURACY (default 0.95) OR honest plateau.
         Enforced in scripts/refine_generators_from_failures.py.
  Gate B (architectural correctness): this script.  Tests whether
         the trained factorized NHSVM channel is a verified mutually-
         affirming addition to the cosine channel that `just optimize
         cosine` previously established.

Gate B is NOT a substitute for Gate A.  A high-accuracy SVM that fails
this gate is redundant or correlated with cosine and shouldn't ship;
a Gate-B-passing SVM that hasn't reached Gate A is correct-but-weak
and shouldn't ship either.  Both required.  See
.claude/projects/-home-cdsw/memory/feedback_deployment_ready_is_mutually_affirming.md
for the dual-gate rationale.

This gate is the pairwise Dempster fusion of (cosine ⊕ SVM) compared
against cosine alone on the full agent-mediated reference set
(1126 rows after singleton-class filter).  Four checks:

  1. **Global uplift**:  A_fused − A_cos ≥ ε (default 0.01).  Mandatory.
  2. **Per-code regression band**: of rows where cos_top1 == true,
     fraction where fused_top1 != true must stay ≤ 0.05.  Mandatory.
  3. **Confidence-conditional do-no-harm**: among rows where cosine
     emits ≥ 0.7 mass on its top-1 singleton AND cos_top1 == true,
     100% must still have fused_top1 == true.  No overturns of
     confident-and-right cosine calls.  Mandatory.
  4. **Conflict K bound** (diagnostic): median K across rows ≤ 0.5.
     Logged but not fail-default.

A failing gate run still produces the full report so the operator can
see WHICH check failed and WHICH codes regressed — the actionable
output of refinement.

Preconditions:
  - Qdrant running with the current ColBERT collection populated by
    `just optimize cosine`.  The bridge raises LateInteractionUnavailable
    if not — that's a precondition failure, not a gate failure.
  - ATELIER_DB_URL set so the bridge can resolve the active taxonomy.
  - Synth corpus present at the configured corpus directory (defaults
    to build/data/svm_training/corpus_v2).
  - Cached synth embeddings at build/reflect_nhsvm_eval_shap_v2/
    embeddings_cache.npz under `synth-only-{corpus_hash}` key
    (populated by a prior Phase D run on the same corpus).

Outputs:
  build/svm_uplift_gate/uplift_report.json — per-code table + per-row
                                              records + 4-check status
  build/svm_uplift_gate/uplift_report.md  — operator-readable summary

Exit code: 0 if gate passes, non-zero otherwise.

Usage:
  python scripts/svm_cosine_uplift_gate.py
  python scripts/svm_cosine_uplift_gate.py --corpus-dir build/data/svm_training/corpus_v3
  python scripts/svm_cosine_uplift_gate.py --uplift-threshold 0.02
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

from atelier.config import load_config
from atelier.classify.belief import (
    BeliefAssignment, FocalElement, FrameOfDiscernment, dempster_combine,
)
from atelier.classify.factorized_nhsvm import fit_factorized_nhsvm
from atelier.classify.features import extract_features
from atelier.classify.late_interaction_bridge import (
    LateInteractionUnavailable, try_compute_cosine_mass,
)
from atelier.classify.mass_functions import nhsvm_to_mass
from reflect_nhsvm import (
    Row, build_category_set, build_texts_and_labels, load_rows,
)
from reflect_nhsvm_eval_shap import BEST_KNOB
from reflect_nhsvm_eval_shap_v2 import (
    REPORT_DIR as PHASE_D_REPORT_DIR,
    corpus_hash, encode_with_cache, load_synth_rows,
)

log = logging.getLogger("svm_cosine_uplift_gate")


# Default thresholds — operator-tunable via CLI
DEFAULT_UPLIFT_EPS = 0.01           # check 1: A_fused - A_cos minimum
DEFAULT_REGRESSION_BAND = 0.05      # check 2: max regression rate on cos-right rows
DEFAULT_CONFIDENT_COS_MASS = 0.7    # check 3: cosine top-1 singleton mass threshold
DEFAULT_K_MEDIAN_BOUND = 0.5        # check 4: median conflict K bound (diagnostic)

DEFAULT_CORPUS_DIR = Path("build/data/svm_training/corpus_v2")
OUTPUT_DIR = Path("build/svm_uplift_gate")


# ──────────────────────────────────────────────────────────────────────
# SVM training (mirrors reflect_nhsvm_eval_shap_v2)
# ──────────────────────────────────────────────────────────────────────

def _train_factorized_head_on_synth(
    corpus_dir: Path, batch_size: int, refresh_embeddings: bool,
):
    """Train the factorized NHSVM head on the synth corpus.

    Mirrors Phase D's training procedure — uses the same cache key
    (`synth-only-{corpus_hash}`) so re-running this script after Phase D
    on the same corpus is a cache hit, not a re-encode.
    """
    log.info("Loading synth corpus from %s", corpus_dir)
    synth_rows = load_synth_rows(corpus_dir)
    synth_labels = [r.code for r in synth_rows]

    code_counts = Counter(synth_labels)
    keep = [i for i, l in enumerate(synth_labels) if code_counts[l] >= 2]
    train_rows = [synth_rows[i] for i in keep]
    train_labels = [synth_labels[i] for i in keep]
    log.info("  train (synth, post-singleton-filter): %d rows  %d codes",
             len(train_rows), len(set(train_labels)))

    train_texts, _ = build_texts_and_labels(train_rows)
    ch = corpus_hash(corpus_dir)
    log.info("Encoding (or loading cached) synth train embeddings...")
    X_train = encode_with_cache(
        train_texts, cache_key=f"synth-only-{ch}",
        refresh=refresh_embeddings, batch_size=batch_size,
    )
    log.info("  X_train shape=%s", X_train.shape)

    log.info("Training factorized NHSVM head...")
    cat_set = build_category_set()
    t0 = time.time()
    head, train_result = fit_factorized_nhsvm(
        X_train, train_labels, cat_set,
        **BEST_KNOB, verbose=True, eval_every=50,
    )
    log.info("  trained: fit-acc=%.4f  elapsed=%.1fs",
             train_result.final_train_acc, time.time() - t0)
    return head, cat_set


# ──────────────────────────────────────────────────────────────────────
# Reference set (validate dataset — full agent-mediated reference)
# ──────────────────────────────────────────────────────────────────────

def _load_validate_rows() -> list[Row]:
    """Load the full agent-mediated reference set, drop singleton classes.

    Matches Phase D's validate set exactly so cosine and SVM evaluate on
    the same 1126 rows.
    """
    real_rows = load_rows(refresh_cache=False, database="reference_corpus")
    _, real_labels = build_texts_and_labels(real_rows)
    code_counts = Counter(real_labels)
    singletons = {c for c, n in code_counts.items() if n < 2}
    validate_rows = [
        r for r, l in zip(real_rows, real_labels) if l not in singletons
    ]
    log.info("  validate (full reference): %d rows  (%d singleton-class rows dropped)",
             len(validate_rows), len(real_rows) - len(validate_rows))
    return validate_rows


# ──────────────────────────────────────────────────────────────────────
# Per-row mass computation
# ──────────────────────────────────────────────────────────────────────

def _column_features_for_row(row: Row):
    """Build a ColumnFeatures using the same code path the production
    pipeline uses for cosine input.  Ensures cosine sees the exact text
    it would see in production.
    """
    return extract_features(
        column_name=row.column,
        column_type=row.column_type,
        values=row.sample_values,
        siblings=row.siblings_full,
        source_table=row.table,
    )


def _svm_proba_dict(head, X_row: np.ndarray) -> dict[str, float]:
    """Run the factorized head on a single row, return {code: prob}.

    `head.predict_proba` returns (1, n_nodes) softmax probabilities;
    we dict-ify against the head's code ordering.
    """
    X_norm = sk_normalize(X_row.reshape(1, -1), norm="l2", axis=1).astype(np.float32)
    device = next(head.parameters()).device
    X_t = torch.tensor(X_norm, device=device)
    proba = head.predict_proba(X_t)[0]  # (n_nodes,)
    return {head.codes[i]: float(proba[i]) for i in range(len(head.codes))}


# ──────────────────────────────────────────────────────────────────────
# Mass extraction helpers
# ──────────────────────────────────────────────────────────────────────

def _top1_singleton(ba: BeliefAssignment, frame: FrameOfDiscernment) -> tuple[str | None, float]:
    """Return (code, mass) of the highest-mass SINGLETON focal element.

    Mirrors pipeline.py's most_committed_singleton().  Returns
    (None, 0.0) if the assignment has no singleton mass (only
    internal-node or Θ mass).
    """
    top = ba.most_committed_singleton()
    return (top[0], top[1]) if top is not None else (None, 0.0)


# ──────────────────────────────────────────────────────────────────────
# Per-code aggregation
# ──────────────────────────────────────────────────────────────────────

def _aggregate_per_code(records: list[dict]) -> list[dict]:
    """Build per-code accuracy table from per-row records."""
    by_code: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "cos_correct": 0, "svm_correct": 0,
                  "fused_correct": 0, "n_regressions": 0}
    )
    for r in records:
        true = r["true"]
        entry = by_code[true]
        entry["n"] += 1
        if r["cos_top1"] == true:
            entry["cos_correct"] += 1
            if r["fused_top1"] != true:
                entry["n_regressions"] += 1
        if r["svm_top1"] == true:
            entry["svm_correct"] += 1
        if r["fused_top1"] == true:
            entry["fused_correct"] += 1
    out = []
    for code in sorted(by_code.keys()):
        e = by_code[code]
        n = e["n"]
        a_cos = e["cos_correct"] / n
        a_svm = e["svm_correct"] / n
        a_fused = e["fused_correct"] / n
        out.append({
            "code": code,
            "n": n,
            "A_cos": round(a_cos, 4),
            "A_svm": round(a_svm, 4),
            "A_fused": round(a_fused, 4),
            "delta": round(a_fused - a_cos, 4),
            "n_regressions": e["n_regressions"],
            "regression_rate": (round(e["n_regressions"] / e["cos_correct"], 4)
                                 if e["cos_correct"] else 0.0),
        })
    return out


# ──────────────────────────────────────────────────────────────────────
# Gate decision
# ──────────────────────────────────────────────────────────────────────

def _evaluate_gate(
    records: list[dict],
    *,
    uplift_eps: float,
    regression_band: float,
    confident_cos_mass: float,
    k_median_bound: float,
) -> dict:
    """Evaluate the four gate checks on per-row records."""
    n = len(records)
    a_cos = sum(1 for r in records if r["cos_top1"] == r["true"]) / max(n, 1)
    a_svm = sum(1 for r in records if r["svm_top1"] == r["true"]) / max(n, 1)
    a_fused = sum(1 for r in records if r["fused_top1"] == r["true"]) / max(n, 1)
    uplift = a_fused - a_cos

    # Check 1: global uplift
    check1 = {
        "name": "global_uplift",
        "pass": bool(uplift >= uplift_eps),
        "observed": round(uplift, 4),
        "threshold": uplift_eps,
        "mandatory": True,
    }

    # Check 2: per-code regression band — fraction of rows where
    # cosine was right but fused is wrong, over all cosine-correct rows.
    cos_right = [r for r in records if r["cos_top1"] == r["true"]]
    regressions = [r for r in cos_right if r["fused_top1"] != r["true"]]
    reg_rate = len(regressions) / max(len(cos_right), 1)
    check2 = {
        "name": "regression_band",
        "pass": bool(reg_rate <= regression_band),
        "observed": round(reg_rate, 4),
        "n_cos_right": len(cos_right),
        "n_regressions": len(regressions),
        "threshold": regression_band,
        "mandatory": True,
    }

    # Check 3: confidence-conditional do-no-harm
    confident_cos_right = [
        r for r in records
        if r["cos_top1"] == r["true"] and r["cos_top1_mass"] >= confident_cos_mass
    ]
    confident_violations = [
        r for r in confident_cos_right if r["fused_top1"] != r["true"]
    ]
    check3 = {
        "name": "confident_cosine_preservation",
        "pass": bool(len(confident_violations) == 0),
        "observed": len(confident_violations),
        "n_confident_cos_right": len(confident_cos_right),
        "confident_mass_threshold": confident_cos_mass,
        "threshold": 0,
        "mandatory": True,
    }

    # Check 4: median K (diagnostic)
    ks = sorted(r["K"] for r in records if r.get("K") is not None)
    median_k = ks[len(ks) // 2] if ks else 0.0
    check4 = {
        "name": "median_K",
        "pass": bool(median_k <= k_median_bound),
        "observed": round(median_k, 4),
        "threshold": k_median_bound,
        "mandatory": False,
    }

    checks = [check1, check2, check3, check4]
    mandatory_pass = all(c["pass"] for c in checks if c["mandatory"])

    return {
        "global": {
            "A_cos": round(a_cos, 4),
            "A_svm": round(a_svm, 4),
            "A_fused": round(a_fused, 4),
            "uplift": round(uplift, 4),
        },
        "checks": {c["name"]: c for c in checks},
        "deployment_ready": mandatory_pass,
        "confident_violations_sample": [
            {"key": v["key"], "true": v["true"],
             "cos_top1": v["cos_top1"], "fused_top1": v["fused_top1"]}
            for v in confident_violations[:30]
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Report rendering
# ──────────────────────────────────────────────────────────────────────

def _render_report_md(report: dict) -> str:
    g = report["global"]
    checks = report["checks"]
    ready = report["deployment_ready"]
    lines: list[str] = []
    lines.append("# cosine-SVM mutual-affirmation gate report")
    lines.append("")
    lines.append(f"**DEPLOYMENT_READY: {'✓ TRUE' if ready else '✗ FALSE'}**")
    lines.append("")
    lines.append(f"- corpus_hash: `{report['corpus_hash']}`")
    lines.append(f"- qdrant_collection: `{report['qdrant_collection']}`")
    lines.append(f"- n_validate: {report['n_validate']} "
                 f"(rows with cosine mass: {report['n_validate_with_cosine_mass']})")
    lines.append("")
    lines.append("## Global accuracies")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| A_cos (cosine alone) | **{g['A_cos']:.4f}** |")
    lines.append(f"| A_svm (SVM alone) | {g['A_svm']:.4f} |")
    lines.append(f"| A_fused (cos ⊕ svm) | **{g['A_fused']:.4f}** |")
    sign = "+" if g["uplift"] >= 0 else ""
    lines.append(f"| uplift (A_fused − A_cos) | **{sign}{g['uplift']:.4f}** |")
    lines.append("")
    lines.append("## Gate checks")
    lines.append("")
    lines.append("| # | check | mandatory | observed | threshold | pass |")
    lines.append("|---|---|---|---:|---:|---|")
    for i, (name, info) in enumerate(checks.items(), 1):
        check_pass = "✓" if info["pass"] else "✗"
        mand = "yes" if info["mandatory"] else "no"
        lines.append(f"| {i} | {name} | {mand} | "
                     f"{info['observed']} | {info['threshold']} | {check_pass} |")
    lines.append("")
    if not ready:
        failed = [c for c in checks.values() if c["mandatory"] and not c["pass"]]
        lines.append("## What blocked deployment-readiness")
        lines.append("")
        for c in failed:
            lines.append(f"- **{c['name']}**: observed {c['observed']} "
                         f"vs threshold {c['threshold']}")
        lines.append("")

    # Per-code table — top 20 deltas in each direction
    pcs = report["per_code"]
    improved = sorted([p for p in pcs if p["delta"] > 0],
                      key=lambda x: -x["delta"])[:20]
    regressed = sorted([p for p in pcs if p["delta"] < 0],
                      key=lambda x: x["delta"])[:20]

    if improved:
        lines.append("## Top per-code improvements")
        lines.append("")
        lines.append("| code | n | A_cos | A_svm | A_fused | Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for p in improved:
            lines.append(f"| `{p['code']}` | {p['n']} | "
                         f"{p['A_cos']:.3f} | {p['A_svm']:.3f} | "
                         f"{p['A_fused']:.3f} | +{p['delta']:.3f} |")
        lines.append("")

    if regressed:
        lines.append("## Top per-code regressions")
        lines.append("")
        lines.append("| code | n | A_cos | A_svm | A_fused | Δ |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for p in regressed:
            lines.append(f"| `{p['code']}` | {p['n']} | "
                         f"{p['A_cos']:.3f} | {p['A_svm']:.3f} | "
                         f"{p['A_fused']:.3f} | {p['delta']:.3f} |")
        lines.append("")

    if report["confident_violations_sample"]:
        lines.append("## Confident-cosine violations (do-no-harm check failed)")
        lines.append("")
        lines.append("Sample of rows where cosine had ≥ "
                     f"{checks['confident_cosine_preservation']['confident_mass_threshold']} "
                     "mass on its top-1 AND cosine was right, but the fused "
                     "prediction was wrong.  These are the cases the SVM "
                     "channel actively dragged cosine off-target:")
        lines.append("")
        lines.append("| row | true | cos_top1 | fused_top1 |")
        lines.append("|---|---|---|---|")
        for v in report["confident_violations_sample"][:15]:
            lines.append(f"| `{v['key']}` | `{v['true']}` | `{v['cos_top1']}` | `{v['fused_top1']}` |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if ready:
        lines.append(
            f"All mandatory checks passed.  The SVM channel adds {sign}"
            f"{g['uplift']:.4f} on full-reference accuracy when paired with "
            f"cosine; no significant regressions on cosine's correct calls; "
            f"zero overturns of confident-and-right cosine predictions.  "
            f"The channel is DEPLOYMENT_READY for integration into the "
            f"Atelier classification pipeline."
        )
    else:
        lines.append(
            f"At least one mandatory check failed.  The SVM channel is "
            f"NOT deployment-ready.  Run another corpus refinement cycle "
            f"targeting the failed-check direction (e.g., codes in the "
            f"regression table above for check 2)."
        )
    lines.append("")
    lines.append(
        "Note: this gate is INDEPENDENT of TARGET_ACCURACY.  "
        f"TARGET_ACCURACY (the runtime pipeline's bar, typically 0.95) "
        f"is measured by running the full Atelier classification pipeline "
        f"against hive-poc and counting `matches_reference == true` in "
        f"classifications.json.  That's a separate downstream check."
    )
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR,
                    help="Synth corpus directory (default: corpus_v2)")
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR / "uplift_report.json",
                    help="Output JSON path")
    ap.add_argument("--output-md", type=Path,
                    default=OUTPUT_DIR / "uplift_report.md",
                    help="Output Markdown path")
    ap.add_argument("--uplift-threshold", type=float, default=DEFAULT_UPLIFT_EPS,
                    help=f"Check 1: minimum global uplift (default {DEFAULT_UPLIFT_EPS})")
    ap.add_argument("--regression-band", type=float, default=DEFAULT_REGRESSION_BAND,
                    help=f"Check 2: max regression rate on cos-right rows (default {DEFAULT_REGRESSION_BAND})")
    ap.add_argument("--confident-cos-mass", type=float, default=DEFAULT_CONFIDENT_COS_MASS,
                    help=f"Check 3: confident-cosine mass threshold (default {DEFAULT_CONFIDENT_COS_MASS})")
    ap.add_argument("--k-median-bound", type=float, default=DEFAULT_K_MEDIAN_BOUND,
                    help=f"Check 4: median K bound, diagnostic only (default {DEFAULT_K_MEDIAN_BOUND})")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--refresh-embeddings", action="store_true")
    ap.add_argument("--qdrant-url", default=None,
                    help="Override Qdrant URL (bypasses taxonomy_registry DAO lookup)")
    ap.add_argument("--qdrant-collection", default=None,
                    help="Override Qdrant collection name (bypasses taxonomy_registry DAO lookup)")
    args = ap.parse_args()

    # Optional override of taxonomy-registry-based collection resolution.
    # Useful when running outside the live App pod (no PGlite) but with
    # a known Qdrant collection.
    if args.qdrant_url and args.qdrant_collection:
        import atelier.classify.late_interaction_bridge as _bridge
        _bridge._resolve_qdrant_collection = (
            lambda cfg, _url=args.qdrant_url, _col=args.qdrant_collection:
            (_url, _col)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)

    log_path = args.output.parent / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w"),
        ],
    )

    log.info("=== cosine-SVM mutual-affirmation gate ===")

    # Config + frame
    cfg = load_config()
    log.info("Loaded config; ColBERT model: %s",
             getattr(cfg, "classify_colbert_model", "<default>"))

    head, cat_set = _train_factorized_head_on_synth(
        args.corpus_dir, batch_size=args.batch_size,
        refresh_embeddings=args.refresh_embeddings,
    )
    frame = FrameOfDiscernment(cat_set, confusable_pairs=[])
    alphas = cat_set.compute_nhsvm_alphas()
    log.info("Frame built: %d singletons, %d internal nodes",
             len(frame.singletons), len(frame.internal_nodes))

    # Validate rows
    validate_rows = _load_validate_rows()
    validate_texts, validate_labels = build_texts_and_labels(validate_rows)
    log.info("Encoding validate rows for SVM inference...")
    X_validate = encode_with_cache(
        validate_texts, cache_key="full-reference",
        refresh=args.refresh_embeddings, batch_size=args.batch_size,
    )

    # Resolve Qdrant collection (for the report header)
    from atelier.classify.late_interaction_bridge import _resolve_qdrant_collection
    resolved = _resolve_qdrant_collection(cfg)
    if resolved is None:
        log.error("No 'current' Qdrant collection registered.  "
                  "Run scripts/enrich_annotations.py or `just optimize cosine` first.")
        return 2
    qdrant_url, collection_name = resolved
    log.info("Qdrant collection: %s @ %s", collection_name, qdrant_url)

    # Per-row pairwise fusion
    log.info("Computing per-row cosine + SVM mass + pairwise fusion (%d rows)...",
             len(validate_rows))
    records: list[dict] = []
    n_cosine_failed = 0
    t0 = time.time()
    for i, (row, X_row, label) in enumerate(zip(validate_rows, X_validate, validate_labels)):
        # Cosine mass via the production bridge
        cf = _column_features_for_row(row)
        try:
            m_cos, status, attr = try_compute_cosine_mass(
                cfg=cfg, column_features=cf,
                column_name=row.column, table_name=row.table,
                samples=row.sample_values,
                neighbor_column_names=row.siblings_full,
                pattern_summary=None, frame=frame,
            )
        except LateInteractionUnavailable as exc:
            log.error("Late-interaction unavailable for %s.%s: %s",
                      row.table, row.column, exc)
            return 3
        if m_cos is None:
            log.warning("Cosine returned None (status=%s) for %s.%s — "
                        "skipping row", status, row.table, row.column)
            n_cosine_failed += 1
            continue

        # SVM mass via factorized head proba → nhsvm_to_mass
        svm_proba = _svm_proba_dict(head, X_row)
        m_svm = nhsvm_to_mass(
            svm_proba, frame, cat_set, alphas, discount=0.20,
        )

        # Pairwise fusion
        try:
            m_fused, K = dempster_combine(m_cos, m_svm)
        except ValueError as exc:
            # Total conflict (K=1) — record as conflict-dominated row,
            # use vacuous fused (treated as no prediction).
            log.warning("Total Dempster conflict for %s.%s: %s",
                        row.table, row.column, exc)
            m_fused = frame.vacuous()
            K = 1.0

        # Extract top-1
        cos_top1, cos_top1_mass = _top1_singleton(m_cos, frame)
        svm_top1, _ = _top1_singleton(m_svm, frame)
        fused_top1, _ = _top1_singleton(m_fused, frame)

        records.append({
            "key": f"{row.table}.{row.column}",
            "true": label,
            "cos_top1": cos_top1,
            "cos_top1_mass": round(cos_top1_mass, 4),
            "svm_top1": svm_top1,
            "fused_top1": fused_top1,
            "K": round(K, 4),
        })

        if (i + 1) % 100 == 0:
            log.info("  %d/%d rows  %.1fs", i + 1, len(validate_rows),
                     time.time() - t0)

    log.info("All rows processed in %.1fs", time.time() - t0)
    if n_cosine_failed:
        log.warning("%d rows had no cosine mass (skipped from gate)",
                    n_cosine_failed)

    # Gate evaluation
    gate = _evaluate_gate(
        records,
        uplift_eps=args.uplift_threshold,
        regression_band=args.regression_band,
        confident_cos_mass=args.confident_cos_mass,
        k_median_bound=args.k_median_bound,
    )

    # Per-code aggregation
    per_code = _aggregate_per_code(records)

    # Assemble report
    report = {
        "corpus_dir": str(args.corpus_dir),
        "corpus_hash": corpus_hash(args.corpus_dir),
        "qdrant_collection": collection_name,
        "qdrant_url": qdrant_url,
        "n_validate": len(validate_rows),
        "n_validate_with_cosine_mass": len(records),
        "thresholds": {
            "uplift_threshold": args.uplift_threshold,
            "regression_band": args.regression_band,
            "confident_cos_mass": args.confident_cos_mass,
            "k_median_bound": args.k_median_bound,
        },
        **gate,
        "per_code": per_code,
        "per_row": records,
    }
    args.output.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", args.output)
    args.output_md.write_text(_render_report_md(report))
    log.info("Wrote %s", args.output_md)

    # Console summary
    print()
    print("=== Gate result ===")
    g = report["global"]
    print(f"  A_cos:    {g['A_cos']:.4f}")
    print(f"  A_svm:    {g['A_svm']:.4f}")
    print(f"  A_fused:  {g['A_fused']:.4f}")
    sign = "+" if g["uplift"] >= 0 else ""
    print(f"  uplift:   {sign}{g['uplift']:.4f}")
    print()
    for check_name, info in report["checks"].items():
        mark = "✓" if info["pass"] else "✗"
        mand = " (mandatory)" if info["mandatory"] else " (diagnostic)"
        print(f"  {mark} {check_name}{mand}: "
              f"observed={info['observed']}  threshold={info['threshold']}")
    print()
    ready = report["deployment_ready"]
    badge = "✓ DEPLOYMENT_READY" if ready else "✗ NOT DEPLOYMENT_READY"
    print(f"  {badge}")
    print(f"  detail: {args.output_md}")
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
