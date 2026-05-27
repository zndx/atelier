#!/usr/bin/env python
"""CatBoost depth=6 vs depth=8 comparison on 5ef4868c's LLM labels.

Re-fetches samples from Hive (reference_corpus), extracts features,
trains two CatBoosts (depth=6 = current default, depth=8 = overwatch
recommendation), compares mass-distribution sharpness.

Question: does depth=8 break the flat 0.45 ± 0.036 rubber-stamp
pattern that limits CatBoost's contribution to fusion?
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    t0 = time.time()
    print("loading classifications + Hive connection...")
    cls = json.loads(
        Path("build/results/5ef4868c/classifications.json").read_text()
    )
    scored = [c for c in cls if c.get("reference_code") and c.get("llm_code")]
    print(f"  {len(scored)} columns with LLM labels")

    import cml.data_v1 as cmldata
    from atelier.classify.features import extract_features

    # Group columns by table
    by_table: dict[str, list[dict]] = defaultdict(list)
    for c in scored:
        by_table[c["table_name"]].append(c)
    print(f"  spread across {len(by_table)} tables")

    conn = cmldata.get_connection("hive-poc")
    db = "reference_corpus"

    features_list = []
    codes = []
    print(f"\nfetching samples (db={db})...")
    for ti, (tbl, cols) in enumerate(sorted(by_table.items())):
        try:
            df = conn.get_pandas_dataframe(
                f"SELECT * FROM {db}.{tbl} LIMIT 100"
            )
        except Exception as e:
            print(f"  [{ti+1}/{len(by_table)}] {tbl}: SKIP ({type(e).__name__})")
            continue
        # Hive returns table-qualified names (e.g. "academic_records.row_id")
        unqualified = {col.split(".")[-1]: col for col in df.columns}
        siblings = list(unqualified.keys())
        total = len(df)
        usable = 0
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
            features_list.append(feats)
            codes.append(c["llm_code"])
            usable += 1
        if (ti + 1) % 10 == 0:
            print(f"  [{ti+1}/{len(by_table)}] {tbl}: {usable} cols  "
                  f"(elapsed {time.time()-t0:.0f}s)")

    print(f"\ntotal usable (raw): {len(features_list)} cols, "
          f"{len(set(codes))} distinct LLM codes "
          f"(elapsed {time.time()-t0:.0f}s)")

    # Label normalization: resolve mnemonics → hierarchical ids,
    # drop unresolvable hallucinations. Mirrors what _resolve_to_focal_element
    # does at fusion time but applied to training labels.
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.optimize.svm.reflect import build_category_set
    cs = build_category_set()
    frame = FrameOfDiscernment(cs)
    print(f"frame: {len(frame.singletons)} singletons, "
          f"{len(frame.internal_nodes)} internal nodes")

    ann_df = conn.get_pandas_dataframe(
        "select id, annotation from default.annotations "
        "where deprecated != 'yes'"
    )
    mnemonic_to_id = {
        str(a): str(i) for a, i in zip(ann_df['annotation'], ann_df['id'])
    }
    print(f"mnemonic→id map: {len(mnemonic_to_id)} entries")

    norm_features, norm_codes = [], []
    drop_unresolvable = drop_deprecated_aliased = 0
    in_frame = lambda c: c in frame.singletons or c in frame.internal_nodes
    for feat, code in zip(features_list, codes):
        if in_frame(code):
            norm_features.append(feat)
            norm_codes.append(code)
        elif code in mnemonic_to_id:
            canonical = mnemonic_to_id[code]
            if in_frame(canonical):
                norm_features.append(feat)
                norm_codes.append(canonical)
            else:
                drop_deprecated_aliased += 1
        else:
            drop_unresolvable += 1

    features_list, codes = norm_features, norm_codes
    print(f"normalized: {len(features_list)} cols, "
          f"{len(set(codes))} distinct codes "
          f"(dropped {drop_unresolvable} unresolvable + "
          f"{drop_deprecated_aliased} deprecated-aliased)")

    # Train both depths
    from atelier.classify.ml_train import fit_catboost_to_llm_labels

    results = {}
    for depth in (6, 8):
        print(f"\n=== training depth={depth} (200 iter, normalized labels) ===",
              flush=True)
        t_train = time.time()
        cb = fit_catboost_to_llm_labels(
            features_list, codes,
            iterations=200, depth=depth, learning_rate=0.10,
        )
        if cb is None:
            print(f"  fit failed at depth={depth}")
            continue
        print(f"  trained in {time.time()-t_train:.1f}s")
        # Predict on the same data to characterize mass shape
        probas = cb.predict_proba(features_list)
        top1_masses = []
        top1_correct = 0
        for proba, true_code in zip(probas, codes):
            if not proba:
                continue
            top_code = max(proba, key=proba.get)
            top_mass = proba[top_code]
            top1_masses.append(top_mass)
            if top_code == true_code:
                top1_correct += 1
        results[depth] = {
            "top1_masses": top1_masses,
            "top1_acc": top1_correct / len(top1_masses) if top1_masses else 0,
        }

    print(f"\n{'='*78}")
    print(f"{'depth':>6}  {'fit-acc':>8}  {'top1 mean':>10}  {'p10':>6}  {'p50':>6}  {'p90':>6}  {'max':>6}  {'std':>6}")
    print(f"{'-'*78}")
    for depth, r in sorted(results.items()):
        masses = r["top1_masses"]
        if not masses:
            continue
        ms = sorted(masses)
        n = len(ms)
        print(
            f"  {depth:>4}  {100*r['top1_acc']:>6.2f}%  "
            f"{statistics.mean(masses):>10.4f}  "
            f"{ms[int(0.10*n)]:>6.3f}  {ms[int(0.50*n)]:>6.3f}  "
            f"{ms[int(0.90*n)]:>6.3f}  {max(masses):>6.3f}  "
            f"{statistics.stdev(masses):>6.4f}"
        )

    # Quantify the rubber-stamp pattern: how many columns fall in
    # [mean-0.05, mean+0.05]? (Tighter = more rubber-stamp)
    print(f"\nrubber-stamp tightness (% within ±0.05 of mean):")
    for depth, r in sorted(results.items()):
        masses = r["top1_masses"]
        if not masses:
            continue
        mn = statistics.mean(masses)
        in_band = sum(1 for m in masses if abs(m - mn) < 0.05)
        print(f"  depth={depth}: {100*in_band/len(masses):.1f}% within ±0.05")

    # Save snapshot
    snap = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_run": "5ef4868c",
        "n_train": len(features_list),
        "n_distinct_labels": len(set(codes)),
        "depths": {
            str(d): {
                "fit_acc": r["top1_acc"],
                "top1_mass_mean": statistics.mean(r["top1_masses"]),
                "top1_mass_std": statistics.stdev(r["top1_masses"]),
                "top1_mass_max": max(r["top1_masses"]),
                "top1_mass_min": min(r["top1_masses"]),
            }
            for d, r in results.items()
        },
    }
    Path("build/runs/calibration/catboost_depth_experiment.json").write_text(
        json.dumps(snap, indent=2)
    )
    print(f"\nsnapshot → build/runs/calibration/catboost_depth_experiment.json")
    print(f"total wall: {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
