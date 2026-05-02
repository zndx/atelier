#!/usr/bin/env python
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Produce a clean meta-tagging dataset ready for manual Hive import.

The UAT snapshot at ``build/meta-tagging/`` uses a paired-column
convention: each natural-named column (``first_name``) is followed by
a *reference column* (``attr_1_1_1_9_2_1``) whose name encodes the
natural column's reference code in its suffix.  Reference columns
are answer keys — not production schema — and they shouldn't appear
in a dataset anyone runs a classifier against:

  * they leak the reference label directly into the column name;
  * they're trivial to "classify" (the name *is* the answer), so
    including them inflates accuracy metrics vacuously.

This script emits a cleaned copy suitable for Hive CREATE TABLE /
INSERT — reference columns dropped, table-name prefix stripped from
headers (``business_data.first_name`` → ``first_name``).

Output:

    build/meta-tagging-clean/
      README.md                        (provenance + cleaning rules)
      annotations.csv                  (vocabulary, verbatim from UAT)
      curated_reference.csv            (generator-derived + spot-checked
                                       per-column reference; built by
                                       build_curated_reference.py)
      <table>.csv × 8                  (natural-named columns only,
                                       stripped headers)
    build/meta-tagging-clean.zip       (same, zipped for hand-off)

Re-run idempotently.  Nothing leaves ``build/`` so annotations.csv,
the labeled tables, and the curated reference stay out of git.
"""

from __future__ import annotations

import csv
import logging
import shutil
import sys
import zipfile
from pathlib import Path

# Canonical reference-column regex lives in meta_tagging_source —
# importing it here keeps this script in sync with the rest of the
# pipeline rather than duplicating the pattern.
from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE


log = logging.getLogger("clean_hive_dataset")


def _strip_table_prefix(header: list[str], table_name: str) -> list[str]:
    """Drop a uniform ``<table>.`` prefix on every column header."""
    prefix = f"{table_name}."
    if all(h.startswith(prefix) for h in header if h):
        return [h[len(prefix):] for h in header]
    return header


def _clean_table(src: Path, dst: Path) -> tuple[int, int]:
    """Copy one CSV with reference columns dropped and headers stripped.

    Returns (kept_columns, dropped_columns).
    """
    with open(src, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if not header:
        return (0, 0)

    stripped = _strip_table_prefix(header, src.stem)
    keep_idx = [i for i, name in enumerate(stripped)
                if not _REFERENCE_COL_RE.match(name)]
    dropped = len(stripped) - len(keep_idx)

    with open(dst, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([stripped[i] for i in keep_idx])
        for row in rows:
            writer.writerow([row[i] if i < len(row) else "" for i in keep_idx])

    return (len(keep_idx), dropped)


def _build_predictions_parquet(out_dir: Path, *, run_id: str) -> dict[str, int]:
    """Produce ``atelier_predictions.parquet`` next to the curated reference.

    Reads the source run's parquet (``build/results/{run_id}/``), drops
    the legacy ``ground_truth`` / ``is_correct`` columns, and attaches
    three authoritative columns sourced from ``curated_reference.csv``:
    ``reference_code``, ``reference_label``, ``matches_reference =
    predicted_code == reference_code``.  Returns the accuracy counts
    so the README can quote them verbatim from the actual artifact.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    src_parquet = Path(f"build/results/{run_id}/atelier_embeddings.parquet")
    ref_csv = out_dir / "curated_reference.csv"
    dst = out_dir / "atelier_predictions.parquet"
    if not src_parquet.is_file():
        raise FileNotFoundError(f"source parquet not found: {src_parquet}")
    if not ref_csv.is_file():
        raise FileNotFoundError(f"curated reference not found: {ref_csv}")

    # Load curated reference; skip unresolved rows so parquet rows with
    # no curated code get reference_code = "" + matches_reference = None.
    ref_map: dict[tuple[str, str], tuple[str, str]] = {}
    with open(ref_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r["derivation"] == "unresolved":
                continue
            ref_map[(r["table"], r["column"])] = (
                r["reference_code"],
                r.get("reference_label", "") or "",
            )

    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE

    t = pq.read_table(src_parquet)
    keep = [n for n in t.column_names if n not in ("ground_truth", "is_correct")]
    t = t.select(keep)

    # Drop any reference-column rows that leaked through an older
    # pre-invariant loader build.  The natural parquet is defined as
    # "one row per natural-named column" — answer keys belong in
    # atelier_predictions_all_columns.parquet, not here.  Using a mask
    # via pyarrow so we don't round-trip through pandas on the big table.
    leaked_mask = [
        not bool(_REFERENCE_COL_RE.match(str(n))) for n in t["column_name"].to_pylist()
    ]
    if not all(leaked_mask):
        pre_n = t.num_rows
        t = t.filter(pa.array(leaked_mask, type=pa.bool_()))
        log.info(
            "  dropped %d leaked reference-column rows from source parquet "
            "(older loader build); %d → %d natural rows",
            pre_n - t.num_rows, pre_n, t.num_rows,
        )

    tables = t["table_name"].to_pylist()
    cols = t["column_name"].to_pylist()
    preds = [str(x or "").strip() for x in t["predicted_code"].to_pylist()]

    ref_codes: list[str] = []
    ref_labels: list[str] = []
    matches: list[object] = []
    for tn, cn, pred in zip(tables, cols, preds):
        rc, rl = ref_map.get((tn, cn), ("", ""))
        ref_codes.append(rc)
        ref_labels.append(rl)
        matches.append((pred == rc) if rc else None)

    new_cols: list[str] = []
    new_arrs: list = []
    for name, col in zip(t.column_names, t.columns):
        new_cols.append(name); new_arrs.append(col)
        if name == "evidence":
            new_cols.extend(["reference_code", "reference_label", "matches_reference"])
            new_arrs.extend([
                pa.array(ref_codes, type=pa.string()),
                pa.array(ref_labels, type=pa.string()),
                pa.array(matches, type=pa.bool_()),
            ])

    t2 = pa.table(dict(zip(new_cols, new_arrs)))
    pq.write_table(t2, dst)

    # Compute the audit counts the README quotes, reading them back
    # from the parquet we just wrote (so the numbers and the file
    # disagree only via a disk-level corruption).
    df = t2.to_pandas()
    scored = int(df["matches_reference"].notna().sum())
    exact = int(df["matches_reference"].sum(skipna=True))
    hier = 0
    for _, r in df.iterrows():
        rc = str(r["reference_code"]).strip()
        if not rc:
            continue
        p = str(r["predicted_code"]).strip()
        if p == rc or (p and rc.startswith(p + ".")) or (rc and p.startswith(rc + ".")):
            hier += 1
    log.info(
        "  atelier_predictions.parquet: %d rows, %d scorable, "
        "exact=%d/%d=%.4f, hier=%d/%d=%.4f",
        len(df), scored, exact, scored, exact / scored if scored else 0.0,
        hier, scored, hier / scored if scored else 0.0,
    )
    if scored and exact / scored < 0.945:
        raise RuntimeError(
            f"parquet exact accuracy {exact}/{scored} = "
            f"{exact/scored:.4f} below the 94.5% parity floor"
        )

    # Also produce the all-columns parquet — includes every source
    # column, synth-generator answer keys included, so UAT's row-count
    # check-every-column scoring scripts see a complete corpus.
    full_stats = _build_all_columns_parquet(out_dir, natural_parquet=t2)
    return {
        "rows": len(df), "scored": scored, "exact": exact, "hier": hier,
        "full_rows": full_stats["rows"],
        "full_scored": full_stats["scored"],
        "full_exact": full_stats["exact"],
        "full_hier": full_stats["hier"],
    }


def _build_all_columns_parquet(
    out_dir: Path, *, natural_parquet,
) -> dict[str, int]:
    """Extend the natural-only parquet with synth reference-column rows.

    The pipeline intentionally excludes reference columns (name pattern
    ``^(attr|code|col|data|field|item|key|ref|val|var)_\\d+(_\\d+)*$``)
    from classification because they are synth-generator answer keys —
    the numeric suffix literally IS the code.  UAT's post-run scoring
    script iterates the source corpus and flags any column missing
    from the output parquet, so this helper emits a sibling parquet
    that adds one deterministic row per reference column, preserving
    the honest-evaluation semantics of ``atelier_predictions.parquet``
    while letting UAT's row-count check find every source column.

    Reference rows carry NO prediction and NO reference_code.  Decoding
    the column-name suffix to a code would be name-parse cheating — if
    a reviewer renames ``attr_1_1_1_9_2_1`` to ``xyz_9876`` for a
    validation test, any pipeline that name-parses silently "succeeds"
    for the wrong reason.  The honest behavior: leave these rows
    unscored; a reviewer who wants predictions on reference columns
    runs the pipeline with
    ``classify_exclude_reference_columns=false`` and lets the LLM +
    ML classifiers classify them from values alone, exactly as they
    would on a renamed validation set.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE

    src_dir = Path("build/meta-tagging")

    # Natural parquet rows we've already emitted — dedup key (table, col)
    existing: set[tuple[str, str]] = set()
    nat_df = natural_parquet.to_pandas()
    for _, row in nat_df.iterrows():
        existing.add((str(row["table_name"]).strip(),
                      str(row["column_name"]).strip()))

    # Walk source CSV headers and emit ONE row per reference column not
    # already in the natural parquet — predictions deliberately empty.
    # This gives UAT's row-count script every source column without
    # silently declaring "correct by name parse" on synth answer keys.
    new_rows: list[dict] = []
    for csv_path in sorted(src_dir.glob("*.csv")):
        if csv_path.name == "annotations.csv":
            continue
        table_name = csv_path.stem
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f), None)
        if not header:
            continue
        prefix = f"{table_name}."
        names = [h[len(prefix):] if h.startswith(prefix) else h for h in header]
        for col_name in names:
            if not _REFERENCE_COL_RE.match(col_name):
                continue
            if (table_name, col_name) in existing:
                continue
            new_rows.append({
                "table_name": table_name,
                "column_name": col_name,
                # predictions and reference intentionally empty — see
                # docstring; matches_reference=None so they don't enter
                # any accuracy numerator or denominator.
                "predicted_code": "",
                "predicted_label": "",
                "predicted_annotation": "",
                "reference_code": "",
                "reference_label": "",
                "matches_reference": None,
                "confidence": 0.0,
                "belief": 0.0,
                "plausibility": 1.0,
                "uncertainty": 1.0,
                "conflict": 0.0,
                "llm_code": "",
                "llm_confidence": 0.0,
                "needs_clarification": False,
                "evidence": (
                    "synth reference column — excluded from "
                    "classification in the default configuration "
                    "(classify.exclude_reference_columns=true). Re-run "
                    "with the toggle flipped on the Status page to "
                    "have the LLM + ML classifiers predict this column "
                    "from its values alone (no name parse)."
                ),
                "cautious_code": "",
                "column_type": "object",
                "text": col_name,
                "embedding_text": "",
                "pattern_signals": "",
                "dst_belief_path": "[]",
                "x": 0.0,
                "y": 0.0,
                "shap_top1_name": "",
                "shap_top1_value": 0.0,
                "shap_top2_name": "",
                "shap_top2_value": 0.0,
                "shap_top3_name": "",
                "shap_top3_value": 0.0,
            })

    # Compose: keep the natural parquet's columns order, fill any new
    # row dict into matching types.  Use the natural parquet's schema
    # as the authority so both files have identical column layouts.
    schema = natural_parquet.schema
    col_names = [f.name for f in schema]
    appended = {name: [] for name in col_names}
    for row in new_rows:
        for name in col_names:
            appended[name].append(row.get(name, None))
    # Coerce to pyarrow columns matching the natural-parquet schema.
    arrs = []
    for f in schema:
        vals = appended[f.name]
        if pa.types.is_boolean(f.type):
            arr = pa.array(
                [None if v is None else bool(v) for v in vals], type=pa.bool_(),
            )
        elif pa.types.is_floating(f.type):
            arr = pa.array([float(v) if v is not None else 0.0 for v in vals], type=f.type)
        elif pa.types.is_integer(f.type):
            arr = pa.array([int(v) if v is not None else 0 for v in vals], type=f.type)
        else:
            arr = pa.array([("" if v is None else str(v)) for v in vals], type=f.type)
        arrs.append(arr)
    appended_table = pa.table(arrs, names=col_names)
    combined = pa.concat_tables([natural_parquet, appended_table])

    dst = out_dir / "atelier_predictions_all_columns.parquet"
    pq.write_table(combined, dst)

    df = combined.to_pandas()
    scored = int(df["matches_reference"].notna().sum())
    exact = int(df["matches_reference"].sum(skipna=True))
    hier = 0
    for _, r in df.iterrows():
        rc = str(r["reference_code"]).strip()
        if not rc:
            continue
        p = str(r["predicted_code"]).strip()
        if p == rc or (p and rc.startswith(p + ".")) or (rc and p.startswith(rc + ".")):
            hier += 1

    log.info(
        "  atelier_predictions_all_columns.parquet: %d rows "
        "(%d appended reference cols), %d scorable, "
        "exact=%d/%d=%.4f, hier=%d/%d=%.4f",
        len(df), len(new_rows), scored, exact, scored,
        exact / scored if scored else 0.0,
        hier, scored, hier / scored if scored else 0.0,
    )
    return {"rows": len(df), "scored": scored, "exact": exact, "hier": hier}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    src_dir = Path("build/meta-tagging").resolve()
    if not (src_dir / "annotations.csv").is_file():
        log.error("missing %s/annotations.csv", src_dir)
        return 1

    # Preserve the curated-reference CSV across the rmtree + rebuild.
    # It's produced by build_curated_reference.py separately.  The
    # Atelier predictions parquet is re-derived below from the source
    # run parquet so it can't drift from the curated reference.
    out_dir = Path("build/meta-tagging-clean").resolve()
    preserved: dict[str, bytes] = {}
    if out_dir.exists():
        for fname in ("curated_reference.csv", "curated_reference_summary.json"):
            p = out_dir / fname
            if p.is_file():
                preserved[fname] = p.read_bytes()
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for fname, data in preserved.items():
        (out_dir / fname).write_bytes(data)

    # 1. annotations.csv — verbatim.  (The Hive export already has
    # ``annotations.<col>`` headers which Hive imports unchanged;
    # stripping the prefix here would desync from the CREATE TABLE
    # statement UAT already uses for the vocabulary table.)
    shutil.copyfile(src_dir / "annotations.csv", out_dir / "annotations.csv")

    # 1b. Reconciliation doc (if present) — copy from build/results/parity/
    # so the handoff bundle includes the apples-to-apples comparison
    # alongside the data.  Produced by
    # scripts/parity/reconcile_baseline_coverage.py.
    parity_dir = Path("build/results/parity")
    for fname in ("reconciliation.md", "reconciliation.json"):
        src_recon = parity_dir / fname
        if src_recon.is_file():
            shutil.copyfile(src_recon, out_dir / fname)
            log.info("  included %s", fname)

    # 1c. Atelier predictions parquet — built from the source run parquet
    # with reference_code / reference_label / matches_reference
    # authoritatively sourced from curated_reference.csv.  Keeping the
    # build here (rather than preserving a hand-written parquet across
    # rmtrees) guarantees the parquet can't drift from the curated
    # reference between UAT handoffs.
    parquet_stats = _build_predictions_parquet(out_dir, run_id="323cfbbc")

    # 2. data CSVs — drop reference cols, strip table-name prefix.
    table_csvs = sorted(p for p in src_dir.glob("*.csv")
                        if p.name != "annotations.csv")
    totals: list[tuple[str, int, int]] = []
    for src in table_csvs:
        dst = out_dir / src.name
        kept, dropped = _clean_table(src, dst)
        totals.append((src.name, kept, dropped))
        log.info("  %s: kept=%d dropped=%d", src.name, kept, dropped)

    # 3. README.md with provenance.
    readme_lines = [
        "# Meta-tagging UAT corpus — leak-sanitized for LLM/ML comparison",
        "",
        "Revision 4 (2026-04-20). Ships two Atelier prediction parquets "
        "so UAT's scoring and row-audit flows both have the shape they "
        "expect:\n\n"
        "- `atelier_predictions.parquet` — the honest-evaluation "
        "artifact: one row per natural-named column (no synth-generator "
        "reference columns). This is the file `reconciliation.md` "
        "summarizes.\n"
        "- `atelier_predictions_all_columns.parquet` — every column in "
        "the source corpus (natural + synth reference-column "
        "answer keys). Reference-column rows are trivially correct by "
        "name-parse; flagged via `evidence` so reviewers can spot them.",
        "",
        "Generated by ``scripts/parity/make_clean_hive_dataset.py``. Source "
        "is the UAT snapshot at ``build/meta-tagging/`` (2026-04-18). "
        "This cleaned copy is drop-in for Hive CREATE TABLE / INSERT — "
        "header table-name prefixes stripped, synthetic reference "
        "columns removed.",
        "",
        "## What changed vs. the UAT source",
        "",
        "For every data CSV (all eight tables):",
        "",
        "1. **Stripped** the ``<table_name>.`` prefix from column headers "
        "so Hive CREATE TABLE statements are clean (``first_name`` rather "
        "than ``business_data.first_name``).",
        "",
        "2. **Dropped reference columns** — columns whose names match the "
        "paired pattern "
        "``^(attr|code|col|data|field|item|key|ref|val|var)_\\d+(_\\d+)*$``. "
        "These are *answer keys* that encode the reference code of the "
        "natural-named column immediately preceding them in schema order. "
        "Reference columns are a synth-generator artifact, not production "
        "schema, and classifying them is trivial by construction — the "
        "name IS the answer.  Every dropped reference column is paired "
        "with a preceding natural-named column that carries identical "
        "values; only the column name differs.",
        "",
        "The ``annotations.csv`` vocabulary is unchanged from UAT.",
        "",
        "## Per-table cleanup counts",
        "",
        "| Table | Columns kept | Reference columns dropped |",
        "|---|---|---|",
    ]
    for name, kept, dropped in totals:
        readme_lines.append(f"| {name} | {kept} | {dropped} |")
    readme_lines += [
        "",
        "## Curated reference labels",
        "",
        "``curated_reference.csv`` in this bundle is a generator-derived, "
        "spot-checked per-column reference produced by "
        "``scripts/parity/build_curated_reference.py``.  For the "
        "UAT synthetic corpus, 193 / 246 rows come from direct "
        "column-pair evidence (the synth generator encodes each code "
        "in a paired reference-column twin, making that row's label "
        "deterministic by design); the remainder use name-index "
        "lookup with Ontology > Annotation > Common Names priority "
        "and depth-winning tie-breaking.  External, human-curated "
        "benchmarks (e.g. SOTAB's published labels) are out of scope "
        "for this corpus; the file in this bundle is a **curated "
        "reference**, not a human-curated benchmark.",
        "",
        "UAT's own classification outputs (e.g. "
        "``Atelier_Results_Default_DB_4-16.xlsx``) are **provisional** — "
        "they're classifier *predictions*, not curated labels.",
        "",
        "## Reconciliation (``reconciliation.md``)",
        "",
        "Both pipelines are scored against the curated reference in this "
        "bundle. The reconciliation separates three distinct questions the "
        "first bundle did not cleanly disentangle:",
        "",
        "1. **Coverage** — of the 239 resolvable curated-reference columns, "
        "177 appear in the baseline's input, 62 do not. Comparing the two "
        "pipelines on the full 239 conflates accuracy with coverage.",
        "2. **Shared-intersection accuracy** — on the 177-column overlap, "
        "baseline is 88.14% exact (99.36% when it commits a tag); Atelier "
        "is 94.35% exact with 100% coverage.",
        "3. **Extension** — Atelier also tags 20 columns baseline saw but "
        "left blank (65.00% exact) and 62 columns absent from baseline's "
        "input (95.16% exact).",
        "",
        "See ``reconciliation.md`` for per-table breakdowns and "
        "``reconciliation.json`` for machine-readable totals.",
        "",
        "## Atelier predictions (``atelier_predictions.parquet``)",
        "",
        f"{parquet_stats['rows']}-row parquet — one row per natural-named "
        "column from the UAT corpus after reference-column exclusion — "
        "with the full prediction surface the reconciliation summarizes. "
        "Source run: ``323cfbbc``.",
        "",
        "### Accuracy (audited at bundle-build time)",
        "",
        f"- **Exact accuracy**: {parquet_stats['exact']} / "
        f"{parquet_stats['scored']} = "
        f"**{parquet_stats['exact']/parquet_stats['scored']:.2%}** "
        "(``matches_reference == True`` on rows where "
        "``reference_code`` is populated)",
        f"- **Hierarchical accuracy**: {parquet_stats['hier']} / "
        f"{parquet_stats['scored']} = "
        f"**{parquet_stats['hier']/parquet_stats['scored']:.2%}** "
        "(``predicted_code`` equals ``reference_code`` or is an "
        "ancestor / descendant in the ICE hierarchy)",
        "",
        "Reproduction:",
        "",
        "```python",
        "import pyarrow.parquet as pq",
        "t = pq.read_table(\"atelier_predictions.parquet\").to_pandas()",
        "scored = t[\"matches_reference\"].notna().sum()",
        "exact  = t[\"matches_reference\"].sum(skipna=True)",
        "print(f\"{exact}/{scored} = {exact/scored:.2%}\")",
        "```",
        "",
        "``reference_code`` / ``reference_label`` in the parquet are "
        "joined from ``curated_reference.csv`` at bundle-build time, "
        "so the parquet's ``matches_reference`` flag is the same check "
        "``reconciliation.md`` reports.",
        "",
        "### Schema",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| ``table_name``, ``column_name`` | identifier (join key) |",
        "| ``column_type`` | inferred dtype |",
        "| ``predicted_code``, ``predicted_label``, ``predicted_annotation`` | DST-fused prediction |",
        "| ``llm_code``, ``llm_confidence`` | raw LLM sweep output (pass-1 / revisit) |",
        "| ``confidence``, ``belief``, ``plausibility``, ``uncertainty``, ``conflict`` | DST fusion metrics |",
        "| ``needs_clarification`` | flagged by bootstrap convergence |",
        "| ``evidence`` | per-source mass contributions (string) |",
        "| ``reference_code``, ``reference_label`` | value from ``curated_reference.csv`` (join key) |",
        "| ``matches_reference`` | ``predicted_code == reference_code`` (None when no reference) |",
        "| ``embedding_text`` | the 12-feature text used by the embedding model |",
        "| ``pattern_signals`` | pattern-evidence hits |",
        "| ``dst_belief_path`` | JSON belief path leaf → root |",
        "| ``cautious_code`` | deepest code with Bel ≥ 0.7 |",
        "| ``shap_top{1,2,3}_{name,value}`` | per-column SHAP attributions |",
        "| ``text``, ``x``, ``y`` | atlas-compatible tooltip + UMAP projection |",
        "",
        "The parquet is a reproducible artifact — the same LLM batch + "
        "CatBoost + incremental SVM install it rests on can be re-run from "
        "this repo without retraining anything.  Every reconciliation "
        "number in this bundle comes from joining this parquet's "
        "``predicted_code`` against ``curated_reference.csv``.",
        "",
        "## Atelier predictions — all columns (``atelier_predictions_all_columns.parquet``)",
        "",
        f"{parquet_stats['full_rows']}-row parquet — every column in "
        "the source corpus, including synth-generator answer-key "
        "columns (the paired ``attr_*``, ``code_*``, ``item_*``, "
        "``val_*`` etc. twins that the default configuration drops "
        "from the evaluation artifact).  Provided so UAT's scoring "
        "scripts can iterate every source column without flagging any "
        "as missing.",
        "",
        "### How reference-column rows are populated — read this carefully",
        "",
        "The 246 natural-named rows are byte-identical to "
        "``atelier_predictions.parquet``.  The 213 reference-column "
        "rows are **deliberately unscored**:",
        "",
        "- ``predicted_code``, ``predicted_label``, ``reference_code``, "
        "``reference_label`` are all empty.",
        "- ``matches_reference`` is ``None`` so these rows enter "
        "neither numerator nor denominator of any accuracy metric.",
        "- ``evidence`` explains why: the column was excluded from "
        "classification in the default configuration.",
        "",
        "**We intentionally do NOT decode the column-name suffix and "
        "claim it as a prediction.**  If we did, a reviewer who "
        "renamed ``attr_1_1_1_9_2_1`` to ``xyz_9876`` on a validation "
        "test would break the silent name-parse and our accuracy "
        "would collapse.  Any pipeline that name-parses in its "
        "prediction path is cheating against the corpus rather than "
        "classifying it.",
        "",
        "### To get predictions on reference columns, flip the toggle",
        "",
        "On the Status page, turn **Reference Column Handling** from "
        "*Exclude* to *Include*.  The next pipeline run will push "
        "every column — reference twins included — through the LLM "
        "sweep and ML classifiers.  The LLM sees the column name as "
        "an opaque string plus the column's values (identical to the "
        "paired natural column by synth-generator construction) and "
        "classifies from value evidence.  That's the honest test of "
        "whether classification holds up when names carry no signal.",
        "",
        "### Natural-named accuracy (the headline)",
        "",
        f"- rows: **{parquet_stats['full_rows']}** (246 natural + 213 reference)",
        f"- scorable: **{parquet_stats['full_scored']}** (natural-named "
        "only; reference rows deliberately unscored)",
        f"- exact: **{parquet_stats['full_exact']} / "
        f"{parquet_stats['full_scored']} = "
        f"{parquet_stats['full_exact']/parquet_stats['full_scored']:.2%}**",
        f"- hierarchical: **{parquet_stats['full_hier']} / "
        f"{parquet_stats['full_scored']} = "
        f"{parquet_stats['full_hier']/parquet_stats['full_scored']:.2%}**",
        "",
        "These numbers match ``atelier_predictions.parquet`` exactly "
        "— the 213 unscored reference rows are present for row-count "
        "coverage only and do not inflate the accuracy fraction.",
        "",
        "## Revision history",
        "",
        "- rev 1 (2026-04-19, 16:55 UTC) — initial cleaned corpus + "
        "curated_reference.csv shipped.",
        "- rev 2 (2026-04-19) — adds ``reconciliation.md`` + "
        "``reconciliation.json``. No data changes.",
        "- rev 3 (2026-04-20) — adds ``atelier_predictions.parquet`` "
        "(source for the reconciliation numbers) with columns renamed "
        "to the current ``reference_code`` / ``matches_reference`` "
        "convention.  No changes to data CSVs, annotations, curated "
        "reference, or reconciliation.",
        "- rev 4 (2026-04-20, this bundle) — adds "
        "``atelier_predictions_all_columns.parquet`` for UAT's "
        "every-column row audit; drops 17 reference-column rows that "
        "leaked into ``atelier_predictions.parquet`` from an older "
        "loader build (net effect: same 226/239 exact-accuracy "
        "numerator/denominator, cleaner 246-row shape).",
    ]
    (out_dir / "README.md").write_text("\n".join(readme_lines) + "\n")

    # 4. Zip for hand-off.
    zip_path = Path("build/meta-tagging-clean.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=p.relative_to(out_dir.parent))

    # 5. Summary
    total_kept = sum(k for _, k, _ in totals)
    total_dropped = sum(d for _, _, d in totals)
    log.info("output   : %s", out_dir)
    log.info("zip      : %s (%d bytes)", zip_path, zip_path.stat().st_size)
    log.info("totals   : %d columns kept, %d dropped across %d tables",
             total_kept, total_dropped, len(totals))
    return 0


if __name__ == "__main__":
    sys.exit(main())
