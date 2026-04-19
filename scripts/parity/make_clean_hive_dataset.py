#!/usr/bin/env python
"""Produce a clean meta-tagging dataset ready for manual Hive import.

The UAT snapshot at ``build/meta-tagging/`` uses a paired-column
convention: each natural-named column (``first_name``) is followed by
a *reference column* (``attr_1_1_1_9_2_1``) whose name encodes the
natural column's ground-truth code in its suffix.  Reference columns
are answer keys — not production schema — and they shouldn't appear
in a dataset anyone runs a classifier against:

  * they leak the ground truth directly into the column name;
  * they're trivial to "classify" (the name *is* the answer), so
    including them inflates accuracy metrics vacuously.

This script emits a cleaned copy suitable for Hive CREATE TABLE /
INSERT — reference columns dropped, table-name prefix stripped from
headers (``business_data.first_name`` → ``first_name``).

Output:

    build/meta-tagging-clean/
      README.md                        (provenance + cleaning rules)
      annotations.csv                  (vocabulary, verbatim from UAT)
      ground_truth.csv                 (authoritative per-column GT;
                                       built by build_authoritative_ground_truth.py)
      <table>.csv × 8                  (natural-named columns only,
                                       stripped headers)
    build/meta-tagging-clean.zip       (same, zipped for hand-off)

Re-run idempotently.  Nothing leaves ``build/`` so annotations.csv,
the labeled tables, and the authoritative GT stay out of git.
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

    # Preserve the authoritative ground-truth CSV (if it exists) across
    # the rmtree + rebuild so we don't nuke it when rewriting the
    # cleaned-data directory.  It's produced by
    # build_authoritative_ground_truth.py separately.
    out_dir = Path("build/meta-tagging-clean").resolve()
    preserved_gt: bytes | None = None
    preserved_gt_summary: bytes | None = None
    if out_dir.exists():
        gt_path = out_dir / "ground_truth.csv"
        gt_summary = out_dir / "ground_truth_summary.json"
        if gt_path.is_file():
            preserved_gt = gt_path.read_bytes()
        if gt_summary.is_file():
            preserved_gt_summary = gt_summary.read_bytes()
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    if preserved_gt is not None:
        (out_dir / "ground_truth.csv").write_bytes(preserved_gt)
    if preserved_gt_summary is not None:
        (out_dir / "ground_truth_summary.json").write_bytes(preserved_gt_summary)

    # 1. annotations.csv — verbatim.  (The Hive export already has
    # ``annotations.<col>`` headers which Hive imports unchanged;
    # stripping the prefix here would desync from the CREATE TABLE
    # statement UAT already uses for the vocabulary table.)
    shutil.copyfile(src_dir / "annotations.csv", out_dir / "annotations.csv")

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
        "These are *answer keys* that encode the ground-truth code of the "
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
        "## Authoritative ground-truth labels",
        "",
        "``ground_truth.csv`` in this bundle is the authoritative "
        "per-column GT reference derived by "
        "``scripts/parity/build_authoritative_ground_truth.py``. It is "
        "built from direct column-pair evidence (reference-column codes) "
        "plus name-index lookup with "
        "Ontology > Annotation > Common Names priority and "
        "depth-winning tie-breaking. Both the Atelier DST-fused pipeline "
        "(recorded in ``build/results/323cfbbc/``) and Gopala's LLM-only "
        "Agent-Studio workflow should be scored against this reference "
        "so the comparison uses a single, honest ruler.",
        "",
        "UAT's own classification outputs (e.g. "
        "``Atelier_Results_Default_DB_4-16.xlsx``) are **provisional** — "
        "they're classifier *predictions*, not ground truth.",
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
