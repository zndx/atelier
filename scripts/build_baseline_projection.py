#!/usr/bin/env python3
"""Build a structural-baseline embedding-atlas projection from the Ægir release.

This is a **golden path with no pipeline**: it writes a
``build/results/<run_id>/`` directory exactly as ``run_classification_pipeline``
would, but with every prediction set to its KNOWN reference code — the
manually-curated correspondence already held in the release's
``reference.parquet`` (146 ``source_table`` ↔ ``SDG.*`` code ↔ template, all
1:1, each matched to an Atlas glossary term + footprint table).

Why: it gives a structural baseline to *visually* establish that the plumbing
is wired correctly — inputs (the Ægir DDL relational corpus) → the Atelier
results artifact → the embedding-atlas UI → and the Atlas correspondence —
*before* the real classification stack is stood up.  Points cluster by code
(a deterministic per-code layout, since a golden run has uniform belief), so a
glance at the viewer confirms the associations are tracked end to end.  When
the real pipeline later runs over the same release, its
``atelier_embeddings.parquet`` is directly comparable to this one (same keys,
same schema) — divergence is then signal, not noise.

The output schema is produced by the pipeline's own ``_write_parquet`` (with a
precomputed code-clustered layout), so it cannot drift from the real artifact.

Usage::

    uv run python scripts/build_baseline_projection.py
    uv run python scripts/build_baseline_projection.py --per-template 4
    uv run python scripts/build_baseline_projection.py \\
        --atlas-url http://localhost:21000 --verify-atlas

Then point the embedding-atlas viewer (or the Atelier Embeddings page) at
``build/results/<run_id>/atelier_embeddings.parquet``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEFAULT_RELEASE = "/raid/checkpoints/aegir-artifacts/atelier_release_v0_3"
ATLAS_CLUSTER = "aegir"
ATLAS_DB = "footprint"
GLOSSARY = "Aegir Ontology"


def _humanize(token: str) -> str:
    """``hipaa_safeguard_admin`` → ``Hipaa Safeguard Admin``."""
    return " ".join(w.capitalize() for w in token.replace("-", "_").split("_") if w)


def _code_leaf(code: str) -> str:
    """``SDG.ICE.DIRECTIVE.HIPAA_SAFEGUARD_ADMIN`` → ``HIPAA_SAFEGUARD_ADMIN``."""
    return code.rsplit(".", 1)[-1] if code else ""


def _det_unit(seed: str, salt: str) -> float:
    """Deterministic float in [-1, 1) from a string seed (no RNG state)."""
    h = hashlib.md5(f"{seed}|{salt}".encode()).digest()
    n = int.from_bytes(h[:4], "big")
    return (n / 0xFFFFFFFF) * 2.0 - 1.0


def _cluster_layout(
    codes_in_order: list[str],
    column_keys: list[tuple[str, str]],
    *,
    cluster_scale: float = 10.0,
    jitter: float = 0.9,
) -> tuple[list[float], list[float]]:
    """Deterministic 2D layout: one center per code (golden-angle spiral) plus
    a per-column jitter — so same-code columns form a visible cluster."""
    uniq = sorted(set(codes_in_order))
    golden = math.pi * (3.0 - math.sqrt(5.0))  # ~2.399963 rad
    centers: dict[str, tuple[float, float]] = {}
    for i, code in enumerate(uniq):
        r = math.sqrt(i + 1) * cluster_scale
        a = i * golden
        centers[code] = (r * math.cos(a), r * math.sin(a))

    xs: list[float] = []
    ys: list[float] = []
    for code, (table_id, column_id) in zip(codes_in_order, column_keys):
        cx, cy = centers[code]
        xs.append(cx + jitter * _det_unit(column_id, "x"))
        ys.append(cy + jitter * _det_unit(column_id, "y"))
    return xs, ys


def _load_glossary_labels(atlas_url, user, password) -> dict[str, str]:
    """template_id -> short definition, read from the live Atlas glossary."""
    from atelier.governance.atlas import AtlasClient
    from atelier.governance.client import ClientConfig
    from atelier.governance.atlas_source import read_glossary

    atlas = AtlasClient(ClientConfig(url=atlas_url, username=user, password=password))
    return {t.name: (t.definition or "") for t in read_glossary(atlas, glossary_name=GLOSSARY)}


def build(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq
    from atelier.classify.aegir_release import resolve_reference_path
    from atelier.classify.pipeline import _write_parquet

    release = Path(args.release_dir)
    corpus_rows = pq.read_table(release / "corpus_columns.parquet").to_pylist()
    # Key-separated layout (<release>.key/) first, legacy in-dir fallback.
    ref_rows = pq.read_table(resolve_reference_path(release)).to_pylist()

    # (table_id, column_id) -> reference row (the curated correspondence)
    ref_by_col = {(r["table_id"], r["column_id"]): r for r in ref_rows}

    glossary_def: dict[str, str] = {}
    if args.atlas_url:
        glossary_def = _load_glossary_labels(
            args.atlas_url, args.atlas_user, args.atlas_password
        )
        print(f"glossary labels from Atlas: {len(glossary_def)} terms")

    # Optional per-template sampling: keep at most N instance tables per code.
    tables_by_template: dict[str, list[str]] = defaultdict(list)
    seen_tables: set[str] = set()
    for r in corpus_rows:
        tid = r["table_id"]
        if tid in seen_tables:
            continue
        seen_tables.add(tid)
        ref = ref_by_col.get((tid, r["column_id"]))
        if ref:
            tables_by_template[ref["template_id"]].append(tid)
    keep_tables: set[str] = set()
    for tmpl, tids in tables_by_template.items():
        keep_tables.update(tids if args.per_template <= 0 else tids[: args.per_template])

    classifications: list[dict] = []
    codes_in_order: list[str] = []
    column_keys: list[tuple[str, str]] = []
    correspondence: dict[str, dict] = {}  # source_table -> atlas/glossary refs

    for r in corpus_rows:
        tid, cid = r["table_id"], r["column_id"]
        if tid not in keep_tables:
            continue
        ref = ref_by_col.get((tid, cid))
        if not ref:
            continue
        code = ref["reference_code"]
        template_id = ref["template_id"]
        source_table = ref["source_table"]
        label = _humanize(template_id)
        annotation = _code_leaf(code)
        definition = glossary_def.get(template_id, "")
        values = [str(v) for v in (r.get("sample_values") or [])]

        classifications.append({
            "table_name": tid,
            "column_name": r["column_name"],
            "column_type": "string",
            # golden: prediction == curated reference, full confidence
            "predicted_code": code,
            "predicted_label": label,
            "predicted_annotation": annotation,
            "confidence": 1.0,
            "belief": 1.0,
            "plausibility": 1.0,
            "uncertainty": 0.0,
            "conflict": 0.0,
            "needs_clarification": False,
            "reference_code": code,
            "reference_label": label,
            "matches_reference": True,
            "evidence": definition or "structural baseline (curated correspondence)",
            "embedding_text": ", ".join(values[:8]),
            "belief_path": [[code, 1.0, 1.0]],
            "cautious_code": code,
            # provenance back to the Ægir source + Atlas entities
            "source_table": source_table,
        })
        codes_in_order.append(code)
        column_keys.append((tid, cid))
        correspondence.setdefault(source_table, {
            "source_table": source_table,
            "template_id": template_id,
            "reference_code": code,
            "atlas_table_qn": f"{ATLAS_DB}.{source_table}@{ATLAS_CLUSTER}",
            "glossary_term": template_id,
        })

    if not classifications:
        print("ERROR: no columns matched the curated reference", file=sys.stderr)
        return 1

    xs, ys = _cluster_layout(codes_in_order, column_keys)

    run_dir = REPO / args.results_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = _write_parquet(
        classifications, run_dir / "atelier_embeddings.parquet", precomputed_xy=(xs, ys)
    )
    (run_dir / "classifications.json").write_text(
        json.dumps(classifications, indent=2, default=str) + "\n"
    )
    (run_dir / "settings_snapshot.json").write_text(
        json.dumps({
            "source_id": args.source_id,
            "baseline": True,
            "release_dir": str(release),
            "note": "structural baseline — predictions are curated references, no pipeline run",
        }, indent=2) + "\n"
    )
    # Atlas correspondence manifest (one row per source_table).
    corr_path = run_dir / "atlas_correspondence.csv"
    with corr_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["source_table", "template_id", "reference_code",
                           "atlas_table_qn", "glossary_term"]
        )
        w.writeheader()
        w.writerows(sorted(correspondence.values(), key=lambda d: d["source_table"]))

    n_codes = len(set(codes_in_order))
    n_tables = len({c["table_name"] for c in classifications})
    print(f"baseline run: {args.run_id}")
    print(f"  columns:        {len(classifications)}")
    print(f"  tables:         {n_tables}")
    print(f"  distinct codes: {n_codes}")
    print(f"  parquet:        {parquet_path}")
    print(f"  correspondence: {corr_path} ({len(correspondence)} source tables)")

    if args.verify_atlas and args.atlas_url:
        _verify_atlas(args, correspondence)
    return 0


def _verify_atlas(args: argparse.Namespace, correspondence: dict[str, dict]) -> None:
    from atelier.governance.atlas import AtlasClient
    from atelier.governance.client import ClientConfig

    atlas = AtlasClient(ClientConfig(
        url=args.atlas_url, username=args.atlas_user, password=args.atlas_password))
    sample = sorted(correspondence.values(), key=lambda d: d["source_table"])[:10]
    hits = 0
    for c in sample:
        ent = atlas.get_entity_by_qualified_name("rdbms_table", c["atlas_table_qn"])
        ok = bool(ent)
        hits += ok
        if not ok:
            print(f"  MISS: {c['atlas_table_qn']}")
    print(f"  atlas verify (sample {len(sample)}): {hits}/{len(sample)} footprint tables present")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release-dir", default=DEFAULT_RELEASE)
    ap.add_argument("--run-id", default="aegir-baseline-v0_3")
    ap.add_argument("--results-root", default="build/results")
    ap.add_argument("--source-id", default="aegir-baseline/footprint")
    ap.add_argument("--per-template", type=int, default=0,
                    help="Cap instance tables per template (0 = all).")
    ap.add_argument("--atlas-url", default=None,
                    help="Enrich labels from the live Atlas glossary (optional).")
    ap.add_argument("--atlas-user", default="admin")
    ap.add_argument("--atlas-password", default="admin")
    ap.add_argument("--verify-atlas", action="store_true",
                    help="Round-trip-check a sample of footprint tables in Atlas.")
    return build(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
