#!/usr/bin/env python3
"""Backfill ``predicted_label`` + ``predicted_annotation`` on completed
extend-classification runs.

Why: ``extend_pipeline.run_extend_classification`` did not load the
customer's hive ``annotations`` table for hive sources, so the
per-column ``label_lookup`` stayed empty.  The classifier output landed
with ``predicted_code`` set, ``predicted_label`` filled with the bare
code, and ``predicted_annotation`` set to the empty string (which the
embeddings UI renders as null).  The pipeline fix is in
``src/atelier/classify/extend_pipeline.py`` — this script repairs the
historical run artifacts in place.

Side-benefit: this script also normalizes the CatBoost class-space
contamination from the fit-to-LLM training step.  When a run's
``predicted_code`` is an annotation string (e.g. ``IPADDR`` or
``A_FD``) rather than a dot-code, we canonicalize via the live
hive ``annotations`` table: the dot-code lands in ``predicted_code``,
the human label in ``predicted_label``, and the annotation token in
``predicted_annotation``.

Source of truth is the live hive table via ``cml.data_v1``, not a
CSV export — Hue's CSV export quoting is unreliable for multi-line
fields.

Idempotent.  Backs each run dir up to ``<run_id>.bak/`` on first
invocation; subsequent re-runs re-resolve from the unchanged backup
sources.

Usage:
    python3 scripts/backfill_extend_annotations.py

Edit ``RUNS_TO_FIX`` / ``CONNECTION_NAME`` / ``DATABASE`` below to
target other historical runs or other source vocabularies.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Make atelier importable so we reuse the production hive-loader.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from atelier.classify.taxonomy import load_annotations_from_hive  # noqa: E402
from atelier.config import load_config  # noqa: E402

RUNS_TO_FIX: list[str] = ["0146134f"]
CONNECTION_NAME: str = "hive-poc"
# The canonical taxonomy lives at ``default.annotations``; per-database
# annotations tables (e.g. ``reference_corpus.annotations``) do not
# exist and would silently fall through to the canonical via hive
# resolution.  Pinning explicitly avoids that ambiguity.
DATABASE: str = "default"
RESULTS_ROOT: Path = Path("/home/cdsw/build/results")

DOT_CODE = re.compile(r"^\d+(\.\d+)*$")


def build_lookups_from_hive(
    connection_name: str,
    database: str,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    """Pull the annotations table from hive and build the two lookups.

    Returns:
        ``(code_to, ann_to)`` where
        - ``code_to[dot_code] = (label, annotation)``
        - ``ann_to[annotation] = (dot_code, label)``

    First-wins on annotation collisions — the canonical taxonomy is
    expected to be 1:1 but some rows carry comma-joined synonym
    tokens; the first canonical mapping wins.
    """
    cfg = load_config()
    cs = load_annotations_from_hive(
        cfg, connection_name=connection_name, database=database,
        hierarchical=True,
    )
    code_to: dict[str, tuple[str, str]] = {}
    ann_to: dict[str, tuple[str, str]] = {}
    for cat in cs.categories:
        code = (cat.code or "").strip()
        if not code or not DOT_CODE.match(code):
            continue
        label = (cat.label or "").strip()
        ann_raw = getattr(cat, "abbrev", "") or ""
        # ``abbrev`` is the annotation token; defensive first-line
        # extract in case the connector preserves embedded newlines.
        ann = str(ann_raw).split("\n")[0].strip().upper()
        if label:
            code_to[code] = (label, ann)
        if ann and ann not in ann_to:
            ann_to[ann] = (code, label)
    return code_to, ann_to


def make_resolver(
    code_to: dict[str, tuple[str, str]],
    ann_to: dict[str, tuple[str, str]],
):
    """Build the (code, label, annotation) resolver closure.

    Three cases:

    - ``predicted_code`` is a dot-code in the taxonomy → direct lookup.
    - ``predicted_code`` is an annotation string in the taxonomy →
      canonicalize: emit the dot-code as the new ``predicted_code``,
      the canonical label, and the original annotation token.
    - ``predicted_code`` is unknown (CatBoost predicted a class that
      isn't in the user's current taxonomy) → preserve as-is, leave
      annotation empty.  Surfaced in the verification output so
      operators can spot residual contamination.
    """
    def resolve(pred):
        if not pred:
            return pred, "", ""
        p = str(pred)
        if DOT_CODE.match(p):
            label, ann = code_to.get(p, (p, ""))
            return p, label, ann
        pu = p.upper()
        if pu in ann_to:
            code, label = ann_to[pu]
            return code, label, pu
        return p, p, ""
    return resolve


def patch_run(run_id: str, resolve) -> None:
    run_dir = RESULTS_ROOT / run_id
    if not run_dir.is_dir():
        print(f"SKIP {run_id}: dir not found")
        return

    bak = run_dir.with_name(run_id + ".bak")
    if not bak.exists():
        shutil.copytree(run_dir, bak)
        print(f"{run_id}: backed up to {bak.name}/")

    cls_path = run_dir / "classifications.json"
    if cls_path.exists():
        rows = json.loads(cls_path.read_text())
        for r in rows:
            c, l, a = resolve(r.get("predicted_code"))
            r["predicted_code"] = c
            r["predicted_label"] = l
            r["predicted_annotation"] = a
        cls_path.write_text(json.dumps(rows, indent=2, default=str) + "\n")
        print(f"{run_id}: classifications.json — {len(rows)} rows rewritten")

    pq_path = run_dir / "atelier_embeddings.parquet"
    if pq_path.exists():
        df = pq.read_table(pq_path).to_pandas()
        triples = df["predicted_code"].apply(resolve)
        df["predicted_code"] = [t[0] for t in triples]
        df["predicted_label"] = [t[1] for t in triples]
        df["predicted_annotation"] = [t[2] for t in triples]
        pq.write_table(
            pa.Table.from_pandas(df, preserve_index=False),
            pq_path,
        )
        print(f"{run_id}: atelier_embeddings.parquet — {len(df)} rows rewritten")


def verify(run_ids: list[str]) -> None:
    print("\n--- verification ---")
    for run_id in run_ids:
        pq_path = RESULTS_ROOT / run_id / "atelier_embeddings.parquet"
        if not pq_path.exists():
            continue
        df = pq.read_table(pq_path).to_pandas()
        ann_filled = (df["predicted_annotation"].astype(str).str.len() > 0).sum()
        code_form = df["predicted_code"].astype(str).str.match(DOT_CODE).sum()
        print(
            f"{run_id}: {ann_filled}/{len(df)} rows have predicted_annotation; "
            f"{code_form}/{len(df)} predicted_code is dot-code form"
        )


def main() -> None:
    print(
        f"Loading {DATABASE}.annotations from connection {CONNECTION_NAME!r} "
        f"via cml.data_v1..."
    )
    code_to, ann_to = build_lookups_from_hive(CONNECTION_NAME, DATABASE)
    print(
        f"Loaded {len(code_to)} dot-code mappings, {len(ann_to)} annotation "
        f"mappings from live hive table"
    )
    resolve = make_resolver(code_to, ann_to)
    for run_id in RUNS_TO_FIX:
        patch_run(run_id, resolve)
    verify(RUNS_TO_FIX)


if __name__ == "__main__":
    main()
