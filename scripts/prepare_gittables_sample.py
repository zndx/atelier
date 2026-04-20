#!/usr/bin/env python3
"""Prepare a GitTables visualization-ready parquet for the Embeddings.

Reads the signals pipeline's gittables_eval.parquet (which already contains
classification results, DST evidence fusion, and SAGE importance), computes
sentence-transformer embeddings + UMAP projection, and outputs a parquet
compatible with embedding-atlas.

By default (``--classify``), runs atelier's own ML classification on each
column to produce honest ``predicted_label`` values.  This is the seed
dataset's label source — it shows what atelier's evidence fusion produces
without an LLM, so users see real baseline quality.  The full pipeline
(with LLM convergence) can then improve these results.

Usage:
    # From signals eval output (recommended)
    uv run python scripts/prepare_gittables_sample.py \
        --input ~/local/src/cldr/signals/build/gittables_eval.parquet

    # Skip classification (just re-project embeddings)
    uv run python scripts/prepare_gittables_sample.py \
        --input ~/local/src/cldr/signals/build/gittables_eval.parquet \
        --no-classify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _classify_seed_columns(table, columns: set[str]) -> list[str]:
    """Run atelier's ML-only classification on each row to produce labels.

    This is an offline, build-time step that calls ``_classify_column()``
    with no LLM evidence (llm_code=None) — the one case where that is
    legitimate.  The labels represent what atelier's evidence fusion
    (name matching, pattern detection, cosine similarity, CatBoost, SVM)
    produces on its own.
    """
    from atelier.classify.belief import FrameOfDiscernment
    from atelier.classify.features import extract_features
    from atelier.classify.mass_functions import DiscountConfig
    from atelier.classify.pipeline import _classify_column
    from atelier.classify.sampler import ColumnSample
    from atelier.classify.taxonomy import (
        HierarchicalCategorySet,
        load_universal_vocabulary,
    )

    # Load vocabulary
    category_set = load_universal_vocabulary(hierarchical=True)
    if not isinstance(category_set, HierarchicalCategorySet):
        raise RuntimeError("Expected HierarchicalCategorySet")
    frame = FrameOfDiscernment(category_set)
    discounts = DiscountConfig()

    # Configure ML model paths (use defaults from build/)
    from atelier.classify import ml_inference
    ml_inference.configure_paths()

    # Build ColumnSample objects from parquet rows
    col_names = table.column("column_name").to_pylist()
    sample_vals_raw = table.column("sample_values").to_pylist() if "sample_values" in columns else [None] * table.num_rows
    col_types = table.column("column_type").to_pylist() if "column_type" in columns else ["string"] * table.num_rows
    source_tables = table.column("source_table").to_pylist() if "source_table" in columns else ["unknown"] * table.num_rows
    gt_codes = table.column("gt_code").to_pylist() if "gt_code" in columns else [None] * table.num_rows

    labels = []
    n_classified = 0
    for i in range(table.num_rows):
        sv = sample_vals_raw[i]
        values = json.loads(sv) if sv else [] if isinstance(sv, str) else []

        col = ColumnSample(
            name=col_names[i],
            table_name=source_tables[i] or "unknown",
            column_type=col_types[i] or "string",
            values=values[:50],
            siblings=[],
            total_count=len(values),
            null_count=0,
            reference_code=gt_codes[i],
        )

        result = _classify_column(
            col, category_set, frame,
            use_cosine=True,
            discounts=discounts,
        )

        label = result.get("predicted_label") or result.get("predicted_code") or ""
        labels.append(label)
        if label:
            n_classified += 1

    print(f"  Classified {n_classified}/{table.num_rows} columns")
    return labels


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Prepare GitTables visualization parquet for Embeddings",
    )
    p.add_argument(
        "--input", required=True,
        help="Input parquet (signals gittables_eval.parquet or gittables_columns.parquet)",
    )
    p.add_argument(
        "--output", default="data/gittables_sample.parquet",
        help="Output parquet path (default: data/gittables_sample.parquet)",
    )
    p.add_argument(
        "--model", default="all-MiniLM-L6-v2",
        help="Sentence transformer model (default: all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--classify", dest="classify", action="store_true", default=True,
        help="Run atelier classification to produce predicted_label (default)",
    )
    p.add_argument(
        "--no-classify", dest="classify", action="store_false",
        help="Skip classification; copy label from tag_label/gt_code",
    )
    args = p.parse_args(argv)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: {input_path} not found", file=sys.stderr)
        print("Run the signals pipeline first, or use --input to point to "
              "an existing gittables parquet.", file=sys.stderr)
        return 1

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    # ── Load input ───────────────────────────────────────────────
    print(f"Reading {input_path}...")
    table = pq.read_table(str(input_path))
    columns = set(table.column_names)
    n_rows = table.num_rows
    print(f"  {n_rows} rows, {len(columns)} columns")

    # Detect input type: eval parquet (has belief/plausibility) or raw columns
    is_eval = "belief" in columns and "tag_label" in columns

    # ── Build embedding texts ────────────────────────────────────
    if "embedding_text" in columns:
        texts = table.column("embedding_text").to_pylist()
        print("  Using existing embedding_text column")
    else:
        col_names = table.column("column_name").to_pylist()
        samples = table.column("sample_values").to_pylist()
        texts = []
        for name, sv in zip(col_names, samples):
            vals = json.loads(sv) if sv else []
            parts = [f"column: {name}"]
            if vals:
                parts.append(f"values: {', '.join(vals[:3])}")
            texts.append(" | ".join(parts))
        print("  Built embedding text from column_name + sample_values")

    # ── Compute embeddings ───────────────────────────────────────
    print(f"Computing embeddings with {args.model}...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
    print(f"  Shape: {embeddings.shape}")

    # ── UMAP projection ──────────────────────────────────────────
    print("Running UMAP projection...")
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    projection = reducer.fit_transform(embeddings)
    x = projection[:, 0].astype(np.float32)
    y = projection[:, 1].astype(np.float32)

    # ── Build output table ───────────────────────────────────────
    print(f"Writing {output_path}...")

    out_columns = {
        "id": pa.array([f"col_{i:05d}" for i in range(n_rows)], type=pa.string()),
        "x": pa.array(x, type=pa.float32()),
        "y": pa.array(y, type=pa.float32()),
        "text": pa.array(texts, type=pa.string()),
    }

    # predicted_label: produced by atelier's own evidence fusion (ML-only,
    # no LLM).  This is the honest baseline — shows what the pipeline
    # produces before the user runs the full LLM convergence loop.
    if args.classify:
        print("Running atelier classification...")
        labels = _classify_seed_columns(table, columns)
        out_columns["predicted_label"] = pa.array(labels, type=pa.string())
    elif "tag_label" in columns:
        out_columns["predicted_label"] = table.column("tag_label")
    elif "gt_code" in columns:
        out_columns["predicted_label"] = table.column("gt_code")
    elif "column_type" in columns:
        out_columns["predicted_label"] = table.column("column_type")

    # Carry forward useful columns from the eval parquet
    for col in ["source_table", "column_name", "sample_values", "column_kind"]:
        if col in columns:
            out_columns[col] = table.column(col)

    # DST evidence fusion columns (the rich classification data)
    if is_eval:
        for col in ["tag_label", "confidence", "belief", "plausibility",
                     "uncertainty_gap", "conflict", "needs_clarification",
                     "ml_tag_label", "ml_confidence", "correct", "ml_correct"]:
            if col in columns:
                out_columns[col] = table.column(col)

    out_table = pa.table(out_columns)
    pq.write_table(out_table, str(output_path))

    # ── Summary ──────────────────────────────────────────────────
    if "predicted_label" in out_columns:
        cats = out_columns["predicted_label"].to_pylist()
        unique_cats = set(c for c in cats if c)
        from collections import Counter
        print(f"\nWrote {output_path}")
        print(f"  Rows: {n_rows}")
        print(f"  Unique labels: {len(unique_cats)}")
        print(f"  Top labels:")
        for cat, count in Counter(cats).most_common(10):
            print(f"    {cat or '(empty)'}: {count}")
    else:
        print(f"\nWrote {output_path} ({n_rows} rows)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
