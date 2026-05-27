#!/usr/bin/env python
"""Phase 1 calibration re-sweep against the new synth-primary head.

The existing 5ef4868c/classifications.json carries SVM mass from the
OLD c3cf4fce head (uncalibrated, mass_mean=0.029).  This script:

1. Re-fetches each scored column's samples from Hive
2. Extracts ColumnFeatures with siblings (same as pipeline runtime)
3. Encodes through the NEW adapter (b60d5a4e28ccee25, T_baked=0.322)
   to get fresh, calibrated SVM probabilities
4. Substitutes the new SVM evidence into a working copy of
   classifications.json
5. Runs the Phase 1 calibration sweep against the shadow → new
   operating point with the calibrated SVM channel

Output: build/runs/calibration/resweep_b60d5a4e/
  - cls_shadow.json           — classifications.json with new SVM evidence
  - svm_proba_cache.json      — per-column raw SVM probabilities (top-10)
  - new_operating_point.md    — α values + accuracy comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--classifications-path", type=Path,
                    default=Path("build/results/5ef4868c/classifications.json"))
    ap.add_argument("--adapter-dir", type=Path,
                    default=Path("build/cache/nhsvm/b60d5a4e28ccee25"))
    ap.add_argument("--database", default="reference_corpus")
    ap.add_argument("--connection", default="hive-poc")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("build/runs/calibration/resweep_b60d5a4e"))
    ap.add_argument("--features-cache", type=Path,
                    default=Path("build/runs/calibration/resweep_b60d5a4e/features_cache.pkl"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("resweep")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.features_cache.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── 1. Load classifications ──────────────────────────────────────
    log.info("loading classifications from %s", args.classifications_path)
    cls = json.loads(args.classifications_path.read_text())
    scored = [c for c in cls if c.get("reference_code")]
    log.info("  %d scored columns", len(scored))

    # Group by table for batched Hive fetch
    by_table: dict[str, list[dict]] = defaultdict(list)
    for c in scored:
        by_table[c["table_name"]].append(c)
    log.info("  spread across %d tables", len(by_table))

    # ── 2. Fetch + extract features (cached) ─────────────────────────
    import pickle
    if args.features_cache.exists():
        log.info("loading features cache from %s", args.features_cache)
        with open(args.features_cache, "rb") as f:
            features_by_key = pickle.load(f)
        log.info("  cache hit: %d features", len(features_by_key))
    else:
        log.info("fetching samples + extracting features (db=%s)", args.database)
        import cml.data_v1 as cmldata
        from atelier.classify.features import extract_features
        conn = cmldata.get_connection(args.connection)
        features_by_key: dict[str, "Any"] = {}
        for ti, (tbl, cols) in enumerate(sorted(by_table.items())):
            try:
                df = conn.get_pandas_dataframe(
                    f"SELECT * FROM {args.database}.{tbl} LIMIT 100"
                )
            except Exception as e:
                log.warning("  [%d/%d] %s: SKIP (%s)", ti+1, len(by_table),
                            tbl, type(e).__name__)
                continue
            # Hive returns table-qualified col names
            unqualified = {col.split(".")[-1]: col for col in df.columns}
            siblings = list(unqualified.keys())
            total = len(df)
            for c in cols:
                col_name = c["column_name"]
                if col_name not in unqualified:
                    continue
                ser = df[unqualified[col_name]]
                non_null = ser.dropna()
                values = [str(v)[:120] for v in non_null.head(20).tolist()]
                null_count = int(ser.isna().sum())
                try:
                    feats = extract_features(
                        column_name=col_name,
                        column_type=c.get("column_type", "string"),
                        values=values,
                        siblings=[s for s in siblings if s != col_name],
                        null_count=null_count,
                        total_count=total,
                        source_table=tbl,
                        distinct_count=int(non_null.nunique()),
                    )
                except Exception:
                    continue
                key = f"{tbl}.{col_name}"
                features_by_key[key] = feats
            if (ti + 1) % 15 == 0:
                log.info("  [%d/%d] elapsed %.0fs (%d features so far)",
                         ti+1, len(by_table), time.time()-t_start,
                         len(features_by_key))
        log.info("extracted %d features in %.0fs",
                 len(features_by_key), time.time()-t_start)
        with open(args.features_cache, "wb") as f:
            pickle.dump(features_by_key, f)
        log.info("  cached → %s", args.features_cache)

    # ── 3. Load new adapter + predict ────────────────────────────────
    log.info("loading new adapter %s", args.adapter_dir)
    from atelier.classify.factorized_nhsvm import NHSVMHeadAdapter
    adapter = NHSVMHeadAdapter.load(args.adapter_dir)
    log.info("  encoder=%s codes=%d T_baked=%s",
             adapter.encoder_id, len(adapter.head.codes),
             adapter.training_metadata.get("softmax_temperature"))

    log.info("predicting on %d columns via new head's predict_proba_features",
             len(features_by_key))
    t_pred = time.time()
    proba_by_key: dict[str, dict[str, float]] = {}
    skipped = 0
    for i, (key, feats) in enumerate(features_by_key.items()):
        try:
            proba = adapter.predict_proba_features(feats)
        except Exception as e:
            skipped += 1
            continue
        proba_by_key[key] = proba
        if (i + 1) % 200 == 0:
            log.info("  [%d/%d] elapsed %.0fs", i+1,
                     len(features_by_key), time.time()-t_pred)
    log.info("predicted %d (skipped %d) in %.0fs",
             len(proba_by_key), skipped, time.time()-t_pred)

    # Save raw probas for diagnostics (top-10 per column)
    cache_path = args.output_dir / "svm_proba_cache.json"
    proba_top10 = {
        k: dict(sorted(p.items(), key=lambda kv: -kv[1])[:10])
        for k, p in proba_by_key.items()
    }
    cache_path.write_text(json.dumps(proba_top10, indent=1))
    log.info("saved top-10 probas → %s", cache_path)

    # ── 4. Substitute SVM evidence ────────────────────────────────────
    # Build mass via nhsvm_to_mass (same path the pipeline uses), then
    # take top-3 like the original evidence_sources storage format.
    log.info("rebuilding SVM evidence via nhsvm_to_mass")
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.mass_functions import nhsvm_to_mass
    from atelier.optimize.svm.reflect import build_category_set
    cs = build_category_set()
    frame = FrameOfDiscernment(cs)
    nhsvm_alphas = cs.compute_nhsvm_alphas()
    # Use default discount + mass_alpha=1.0 so the sweep applies its
    # own α post-scaling
    new_svm_evidence: dict[str, dict[str, float]] = {}
    for key, proba in proba_by_key.items():
        try:
            mass = nhsvm_to_mass(proba, frame, cs, nhsvm_alphas,
                                  discount=0.20, temperature=1.0)
        except Exception as e:
            continue
        # Serialize like evidence_sources["svm"]: top-3 by mass
        masses_dict = {}
        for fe, m in mass.masses.items():
            if fe == frame.theta:
                continue
            # Build code string with * suffix for internal nodes
            codes = sorted(fe.codes)
            if len(codes) == 1:
                code_str = codes[0]
            else:
                # Find which internal-node code maps to this FE
                inv = {tuple(sorted(v.codes)): k for k, v in frame.internal_nodes.items()}
                int_code = inv.get(tuple(codes))
                code_str = f"{int_code}*" if int_code else f"|{len(codes)}|"
            masses_dict[code_str] = float(m)
        top3 = dict(sorted(masses_dict.items(), key=lambda kv: -kv[1])[:3])
        new_svm_evidence[key] = top3

    log.info("built new SVM evidence for %d cols", len(new_svm_evidence))

    # ── 5. Write shadow classifications.json ─────────────────────────
    shadow = []
    for c in cls:
        c2 = dict(c)
        key = f"{c['table_name']}.{c['column_name']}"
        if key in new_svm_evidence:
            c2["evidence_sources"] = dict(c["evidence_sources"])
            c2["evidence_sources"]["svm"] = new_svm_evidence[key]
        shadow.append(c2)

    shadow_path = args.output_dir / "cls_shadow.json"
    shadow_path.write_text(json.dumps(shadow, indent=1))
    log.info("wrote shadow classifications → %s", shadow_path)

    # ── 6. Quick mass distribution diff (OLD vs NEW SVM evidence) ────
    import statistics
    old_top1 = []
    new_top1 = []
    for c in scored:
        key = f"{c['table_name']}.{c['column_name']}"
        old_svm = c.get("evidence_sources", {}).get("svm", {})
        new_svm = new_svm_evidence.get(key, {})
        if old_svm:
            old_top1.append(max(old_svm.values()))
        if new_svm:
            new_top1.append(max(new_svm.values()))

    log.info("=== SVM mass distribution: OLD c3cf4fce vs NEW b60d5a4e28ccee25 ===")
    if old_top1:
        log.info("  OLD top1 mass: mean=%.4f p50=%.4f p95=%.4f max=%.4f",
                 statistics.mean(old_top1),
                 sorted(old_top1)[len(old_top1)//2],
                 sorted(old_top1)[int(0.95*len(old_top1))],
                 max(old_top1))
    if new_top1:
        log.info("  NEW top1 mass: mean=%.4f p50=%.4f p95=%.4f max=%.4f",
                 statistics.mean(new_top1),
                 sorted(new_top1)[len(new_top1)//2],
                 sorted(new_top1)[int(0.95*len(new_top1))],
                 max(new_top1))
        if old_top1:
            log.info("  ratio mean: %.1fx", statistics.mean(new_top1) / max(statistics.mean(old_top1), 1e-6))

    log.info("total wall: %.0fs", time.time() - t_start)
    log.info("next step: run sweep against %s", shadow_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
