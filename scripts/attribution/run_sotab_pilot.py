#!/usr/bin/env python
"""SOTAB Schema.org pilot — Path 1 feature-attribution dataset generator.

Runs on ``/raid/datasets/sotab/CTA_training_schemaorg.zip``'s
``sotab_v2_cta_training_set_small.csv`` (27,693 rows across 82
Schema.org column-type labels).

Pipeline (Path 1):

    stratified sample (n_classes × per_class columns)
      → extract features (12-feature ColumnFeatures.to_embedding_text)
      → LLM classify in batches (same llm_backend the main pipeline uses)
      → CatBoost fit to LLM labels (not to published GT)
      → SHAP per row + SAGE corpus-wide
      → attribution JSONL + corpus-wide SAGE JSON + fidelity metrics

Outputs at ``build/sotab_pilot/run_{YYYYMMDD_HHMMSS}/``:

    metadata.json           pilot settings + fidelity metrics
    columns.parquet         per-column records
    sage_importance.json    corpus-wide per-feature SAGE Shapley
    feature_attributions.jsonl  the novel research artifact —
                                one JSON record per column with
                                SHAP contributions + published GT
                                + LLM label + CatBoost label.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import random
import re
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path


SOTAB_ZIP = Path("/raid/datasets/sotab/CTA_training_schemaorg.zip")
SOTAB_LABELS_XLSX = Path("/raid/datasets/sotab/CTA_CPA_label_set_schemaorg.xlsx")
GT_CSV_IN_ZIP = "sotab_v2_cta_training_set_small.csv"


log = logging.getLogger("sotab_pilot")


def _load_gt(zip_path: Path) -> list[tuple[str, int, str]]:
    """Parse the packaged GT CSV into (table_name, col_idx, label) triples."""
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(GT_CSV_IN_ZIP) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            return [
                (r["table_name"], int(r["column_index"]), r["label"])
                for r in reader
            ]


def _stratified_sample(
    rows: list[tuple[str, int, str]],
    n_classes: int,
    per_class: int,
    seed: int,
) -> list[tuple[str, int, str]]:
    """Sample top-N-by-frequency classes, per_class columns each."""
    by_class: dict[str, list[tuple[str, int, str]]] = defaultdict(list)
    for r in rows:
        by_class[r[2]].append(r)
    # Top-N most-populated classes (so we get representative signal).
    top = sorted(by_class.keys(), key=lambda c: -len(by_class[c]))[:n_classes]
    rng = random.Random(seed)
    out: list[tuple[str, int, str]] = []
    for c in top:
        pool = by_class[c]
        rng.shuffle(pool)
        out.extend(pool[:per_class])
    log.info(
        "stratified sample: %d rows across %d classes (top-N by frequency)",
        len(out), len(top),
    )
    return out


def _load_table_columns(
    zip_path: Path,
    requests: list[tuple[str, int, str]],
) -> dict[tuple[str, int], tuple[list[str], list[int], str]]:
    """Return {(table, col_idx): (values, other_col_indices, published_label)}.

    Reads each referenced JSON.GZ once; extracts only the columns
    actually requested plus the list of all column-indices (for the
    `siblings` feature).
    """
    tables_needed: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for table, col, label in requests:
        tables_needed[table].append((col, label))

    out: dict[tuple[str, int], tuple[list[str], list[int], str]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for table, col_specs in tables_needed.items():
            member = f"Train/{table}"
            try:
                data = zf.read(member)
            except KeyError:
                log.warning("table missing in zip: %s", table)
                continue
            # JSONL inside gzip — each line is a dict with stringified
            # integer keys ({"0": val, "1": val, …}).
            rows: list[dict] = []
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(data)) as gz:
                    for line in io.TextIOWrapper(gz, encoding="utf-8"):
                        line = line.strip()
                        if not line:
                            continue
                        rows.append(json.loads(line))
            except Exception as exc:
                log.warning("failed to parse %s: %s", table, exc)
                continue
            if not rows or not isinstance(rows[0], dict):
                continue
            ncols = max(int(k) for k in rows[0].keys()) + 1
            sibling_idxs = list(range(ncols))
            for col_idx, label in col_specs:
                key = str(col_idx)
                values = []
                for r in rows:
                    v = r.get(key)
                    if v is None:
                        continue
                    s = str(v).strip()
                    if s:
                        values.append(s)
                if values:
                    out[(table, col_idx)] = (values, sibling_idxs, label)
    log.info("loaded %d column value lists", len(out))
    return out


def _build_category_set(labels: list[str]):
    """Flat CategorySet over Schema.org SOTAB labels.

    Each label is its own code (no hierarchy in SOTAB CTA).  The
    ``embedding_text`` feeds the cosine / SAGE donors; we use the
    label's humanized form.
    """
    from atelier.classify.taxonomy import ReferenceCategory, HierarchicalCategorySet

    def _humanize(lbl: str) -> str:
        # "Person/name" → "Person name"; CamelCase → "Camel Case"
        s = lbl.replace("/", " ")
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        return s

    cats = [
        ReferenceCategory(
            code=lbl,
            label=_humanize(lbl),
            embedding_text=_humanize(lbl),
            abbrev="",
            description=f"Schema.org type: {lbl}",
            common_names="",
            notation="",
            taxonomy="sotab_schemaorg",
            parent_code=None,
        )
        for lbl in labels
    ]
    return HierarchicalCategorySet(
        name="sotab_schemaorg_cta",
        categories=cats,
        all_categories=cats,
    )


def _build_column_samples(
    requests: list[tuple[str, int, str]],
    loaded: dict[tuple[str, int], tuple[list[str], list[int], str]],
):
    """Build ColumnSamples compatible with llm_backend.classify_batch."""
    from atelier.classify.sampler import ColumnSample
    samples = []
    for table, col_idx, label in requests:
        hit = loaded.get((table, col_idx))
        if hit is None:
            continue
        values, sibling_idxs, _ = hit
        siblings = [f"col_{i}" for i in sibling_idxs if i != col_idx]
        sample_vals = values[:5]
        samples.append(ColumnSample(
            name=f"col_{col_idx}",
            column_type="object",
            values=sample_vals,
            all_values=values,
            total_count=len(values),
            null_count=0,
            table_name=table.replace("_CTA.json.gz", ""),
            database="sotab",
            siblings=siblings,
            ground_truth=label,  # published SOTAB GT — retained for fidelity scoring only
            distinct_count=len(set(values)),
        ))
    return samples


def _extract_features(samples):
    from atelier.classify.features import extract_features
    feats = []
    for s in samples:
        f = extract_features(
            column_name=s.name,
            column_type=s.column_type,
            values=s.values,
            siblings=s.siblings,
            null_count=s.null_count,
            total_count=s.total_count,
            source_table=s.table_name,
            distinct_count=getattr(s, "distinct_count", 0),
        )
        feats.append(f)
    return feats


def _llm_classify(samples, category_set, batch_size: int):
    """Pass-1 LLM classification + capture reasoning traces per batch.

    Returns ``(predictions, reasoning_traces)`` where ``reasoning_traces``
    is a list of per-batch dicts recording the thinking trace GLM-4.7
    emits alongside each JSON answer, plus the column names that batch
    covers.  Captured as a research artifact — these traces are the
    per-batch "how did the LLM arrive at these labels" record, and are
    memorization-safe to share since they are signal-level artifacts
    about features rather than published labels.
    """
    from atelier.classify.llm_backend import (
        create_backend_from_cfg, build_system_prompt, build_category_table,
    )
    from atelier.config import load_config
    cfg = load_config()
    backend = create_backend_from_cfg(cfg)
    table = build_category_table(category_set)
    prompt = build_system_prompt(table, category_set=category_set)

    predictions: list[str] = [""] * len(samples)
    reasoning_traces: list[dict] = []
    t0 = time.time()
    total_batches = (len(samples) + batch_size - 1) // batch_size
    for b in range(0, len(samples), batch_size):
        chunk = samples[b : b + batch_size]
        bnum = b // batch_size + 1
        log.info("LLM batch %d/%d (%d columns)", bnum, total_batches, len(chunk))
        try:
            resp = backend.classify_batch(
                chunk, prompt, revisit_context=None,
                table_name=None,
            )
        except Exception as exc:
            log.warning("batch %d failed: %s", bnum, exc)
            continue
        for i, c in enumerate(resp.classifications):
            predictions[b + i] = c.category_code or ""
        reasoning_traces.append({
            "batch_id": bnum,
            "column_ids": [f"{s.table_name}:{s.name}" for s in chunk],
            "predicted_codes": [c.category_code or "" for c in resp.classifications],
            "reasoning_text": resp.reasoning_text,
            "reasoning_tokens": resp.reasoning_tokens,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "pass": "pass1",
        })
    log.info("LLM classify done in %.1fs", time.time() - t0)
    return predictions, reasoning_traces


def _train_catboost(feats, llm_labels):
    """Fit CatBoost on (embedding(text), llm_label) pairs.

    Mirrors the fit-to-LLM path so SHAP/SAGE attributions of this model
    are interpretations of its >= 95% fidelity reproduction of the
    LLM's decision boundary — not of the LLM directly.
    """
    from atelier.classify.embedding import embed_texts
    from atelier.classify.catboost_classifier import CatBoostColumnClassifier
    import numpy as np

    pairs = [(f, y) for f, y in zip(feats, llm_labels) if y]
    texts = [f.to_embedding_text() for f, _ in pairs]
    labels = [y for _, y in pairs]
    if len(set(labels)) < 2:
        raise RuntimeError(f"insufficient class variety: {set(labels)}")
    log.info("embedding %d texts", len(texts))
    X = np.asarray(embed_texts(texts))
    clf = CatBoostColumnClassifier()
    clf.fit(X, labels)
    return clf, X, labels


def _run_shap(clf, X, preds):
    from atelier.classify.shap_explanations import run_catboost_shap
    import numpy as np
    # predicted class indices
    classes = list(clf._classes)
    pred_idx = np.array([classes.index(p) if p in classes else 0 for p in preds])
    return run_catboost_shap(clf._model, X, pred_idx)


def _run_sage(feats, llm_labels, category_set):
    from atelier.classify.sage import run_sage_analysis
    import numpy as np
    codes = [c.code for c in category_set.categories]
    code_to_idx = {c: i for i, c in enumerate(codes)}
    filtered = [(f, y) for f, y in zip(feats, llm_labels) if y in code_to_idx]
    if not filtered:
        return None
    f_filtered = [f for f, _ in filtered]
    y_idx = np.array([code_to_idx[y] for _, y in filtered])
    return run_sage_analysis(
        f_filtered, y_idx, category_set,
        n_permutations=256, detect_convergence=True,
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-classes", type=int, default=10)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SOTAB_ZIP.is_file():
        log.error("missing %s", SOTAB_ZIP); return 1

    gt_rows = _load_gt(SOTAB_ZIP)
    log.info("SOTAB GT rows: %d across %d labels",
             len(gt_rows), len({r[2] for r in gt_rows}))

    sample = _stratified_sample(gt_rows, args.n_classes, args.per_class, args.seed)
    loaded = _load_table_columns(SOTAB_ZIP, sample)
    samples = _build_column_samples(sample, loaded)
    log.info("column samples: %d", len(samples))

    all_labels = sorted({r[2] for r in gt_rows})
    category_set = _build_category_set(all_labels)
    feats = _extract_features(samples)
    llm_labels, reasoning_traces = _llm_classify(
        samples, category_set, batch_size=args.batch_size,
    )

    # Fidelity vs published GT — the memorization-safe check.
    published = [s.ground_truth for s in samples]
    fidelity_exact = sum(
        1 for p, g in zip(llm_labels, published) if p and p == g
    ) / len(samples) if samples else 0.0
    log.info("LLM vs SOTAB published GT fidelity: %.4f", fidelity_exact)

    # CatBoost fit-to-LLM + attributions.
    # Train on the subset with LLM labels, then predict on ALL 400
    # columns — including those where the LLM abstained on pass 1.
    # For an unlabeled column, the CatBoost prediction is NEW
    # independent information (the model's extrapolation from the
    # labeled subset) that the LLM can consume on a revisit pass.
    clf, X_train, labels = _train_catboost(feats, llm_labels)
    # Embed ALL features (including those from unlabeled rows)
    from atelier.classify.embedding import embed_texts
    import numpy as np
    all_texts = [f.to_embedding_text() for f in feats]
    X_all = np.asarray(embed_texts(all_texts))
    proba_all = clf.predict_proba(X_all)
    cb_preds = [
        max(p.items(), key=lambda kv: kv[1])[0] if p else ""
        for p in proba_all
    ]
    # Top-3 per row for the downstream revisit pass
    cb_top3 = [
        sorted(p.items(), key=lambda kv: -kv[1])[:3] for p in proba_all
    ]
    # Fit-to-LLM fidelity measured only where a pass-1 label exists
    cb_train_preds = [cb_preds[i] for i, y in enumerate(llm_labels) if y]
    cb_fidelity = sum(
        1 for a, b in zip(cb_train_preds, labels) if a == b
    ) / len(labels) if labels else 0.0
    log.info("CatBoost fit-to-LLM fidelity (labeled subset): %.4f", cb_fidelity)
    log.info("LLM-abstained rows rescued by CatBoost: %d (predicted labels available)",
             sum(1 for y, c in zip(llm_labels, cb_preds) if not y and c))

    # SKIP_HEAVY_GPU disables SAGE (the most expensive step) when set.
    # Set via ATELIER_ATTRIB_SKIP_SAGE=1 in the environment.  Useful
    # when the local GPUs are pinned by a different job.
    import os as _os
    skip_sage = _os.environ.get("ATELIER_ATTRIB_SKIP_SAGE", "").lower() in ("1", "true", "yes")

    shap = _run_shap(clf, X_all, cb_preds)
    if skip_sage:
        log.info("SAGE skipped via ATELIER_ATTRIB_SKIP_SAGE=1")
        sage = None
    else:
        sage = _run_sage(feats, llm_labels, category_set)

    # ── Emit artifacts ─────────────────────────────────────────
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path("build/sotab_pilot") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-column records
    # IMPORTANT: retain ALL samples including those where the LLM
    # abstained on pass 1 (y == "").  CatBoost trained on the labeled
    # subset will still produce predictions for those rows at inference
    # time — that extrapolated prediction is NEW independent signal for
    # the LLM on a subsequent revisit pass.  Dropping unlabeled rows
    # here would discard exactly the rows the revisit loop is designed
    # to help.
    import pandas as pd
    shap_records = shap.to_records(k=5) if shap else [{} for _ in samples]
    filtered_samples = list(zip(samples, feats, llm_labels))
    per_col: list[dict] = []
    attrib_jsonl: list[dict] = []
    for i, (s, f, y) in enumerate(filtered_samples):
        sh = shap_records[i] if i < len(shap_records) else {}
        rec_col = {
            "table": s.table_name,
            "column_idx": s.name,
            "published_label": s.ground_truth,
            "llm_label": y,
            "catboost_label": cb_preds[i] if i < len(cb_preds) else "",
            "llm_matches_published": y == s.ground_truth,
            "catboost_matches_llm": (
                cb_preds[i] == y if i < len(cb_preds) else False
            ),
            "embedding_text": f.to_embedding_text(),
            "cardinality": f.cardinality,
            "numeric_ratio": f.numeric_ratio,
            "avg_len": f.avg_value_length,
            "entropy": f.value_entropy,
            "shap_top": json.dumps(sh.get("features", []) if isinstance(sh, dict) else []),
        }
        per_col.append(rec_col)
        attrib_jsonl.append({
            "corpus": "sotab_schemaorg_cta",
            "table_id": s.table_name,
            "column_id": s.name,
            "published_label": s.ground_truth,
            "llm_label": y,
            "catboost_label": cb_preds[i] if i < len(cb_preds) else "",
            "catboost_top3": (
                [(lbl, round(p, 4)) for lbl, p in cb_top3[i]]
                if i < len(cb_top3) else []
            ),
            "llm_matches_published": y == s.ground_truth,
            "llm_abstained_pass1": not y,
            "catboost_matches_llm": (
                cb_preds[i] == y if i < len(cb_preds) and y else False
            ),
            "feature_snapshot": {
                "cardinality": f.cardinality,
                "numeric_ratio": f.numeric_ratio,
                "avg_len": f.avg_value_length,
                "entropy": f.value_entropy,
                "column_type": s.column_type,
                "value_description": f.value_description,
                "pattern_signals": list(f.pattern_signals.keys()) if f.pattern_signals else [],
            },
            "shap_local": (
                sh.get("features", []) if isinstance(sh, dict) else []
            ),
        })

    pd.DataFrame(per_col).to_parquet(out_dir / "columns.parquet")
    with open(out_dir / "feature_attributions.jsonl", "w") as f:
        for rec in attrib_jsonl:
            f.write(json.dumps(rec) + "\n")

    # Reasoning traces — the long-term research artifact we're building
    # while we work.  One record per LLM batch; each carries the
    # thinking text alongside the columns it covers.  Stored as JSONL
    # so appending pass-2 reasoning later is trivial.
    with open(out_dir / "reasoning_traces.jsonl", "w") as f:
        for tr in reasoning_traces:
            f.write(json.dumps(tr) + "\n")
    total_reasoning_chars = sum(len(tr["reasoning_text"]) for tr in reasoning_traces)
    total_reasoning_tokens = sum(tr["reasoning_tokens"] for tr in reasoning_traces)
    log.info(
        "reasoning-trace artifact: %d batches, %d chars, %d tokens",
        len(reasoning_traces), total_reasoning_chars, total_reasoning_tokens,
    )
    if sage:
        (out_dir / "sage_importance.json").write_text(
            json.dumps(sage.to_dict(), indent=2)
        )
    (out_dir / "metadata.json").write_text(json.dumps({
        "run_id": run_id,
        "sotab_zip": str(SOTAB_ZIP),
        "gt_csv": GT_CSV_IN_ZIP,
        "n_classes_sampled": args.n_classes,
        "per_class_target": args.per_class,
        "total_columns": len(samples),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "llm_vs_published_fidelity": round(fidelity_exact, 4),
        "catboost_fit_to_llm_fidelity": round(cb_fidelity, 4),
        "timestamp": run_id,
    }, indent=2))

    print(f"\n=== SOTAB pilot ({run_id}) ===")
    print(f"  columns processed:    {len(samples)}")
    print(f"  LLM vs published GT:  {fidelity_exact:.4f}")
    print(f"  CatBoost vs LLM:      {cb_fidelity:.4f}")
    print(f"  artifacts:            {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
