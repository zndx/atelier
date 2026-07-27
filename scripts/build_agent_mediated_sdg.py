#!/usr/bin/env python3
"""Assemble the SDG working set for agent-mediated blind curation.

This is the SINGLE INGRESS for everything the referee sees (blind-contract
mechanization, 2026-07-03 protocol note): it reads exactly two files —

    <release_dir>/corpus_columns.parquet     the blind surface
    <corpora_dir>/vocabulary/annotations.parquet   the SKOS vocabulary

— records their sha256s in the working-set metadata, and inlines every fact
the curation loop needs so the referee never touches the filesystem. It does
NO classification (mirror of ``build_agent_mediated.py``'s division of
labor: builder gathers, referee decides).

Blind-integrity is STRUCTURAL here:

* the generation-manifest pin guard runs first (unscored mode);
* a release dir that still carries ``reference.parquet`` IN-DIR (legacy
  layout) is refused — agent-facing working sets require the key-separated
  ``<release>.key/`` layout (override with ``--allow-legacy-layout`` for
  hermetic dev fixtures only);
* the emitted working set carries no reference/semantic-register fields by
  construction (the loader only sees the blind parquet).

Output: ``build/data/agent_mediated/<taxonomy_id>/working_set.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from atelier.classify.aegir_release import (  # noqa: E402
    CORPUS_FILE,
    REFERENCE_FILE,
    check_release_pin,
    load_aegir_release_samples,
    load_sdg_vocabulary,
)

_VALUES_HEAD = 15  # sample values shown to the referee per column


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_working_set(release_dir: Path, corpora_dir: Path, *,
                      max_tables: int | None = None,
                      allow_legacy_layout: bool = False) -> dict:
    in_dir_key = release_dir / REFERENCE_FILE
    if in_dir_key.exists() and not allow_legacy_layout:
        raise SystemExit(
            f"REFUSED: {in_dir_key} sits beside the blind surface (legacy "
            f"layout). Agent-facing working sets require the key-separated "
            f"<release>.key/ layout — re-cut the release, or pass "
            f"--allow-legacy-layout for hermetic dev fixtures only."
        )

    manifest = check_release_pin(release_dir, corpora_dir)

    corpus_path = release_dir / CORPUS_FILE
    annotations_path = corpora_dir / "vocabulary" / "annotations.parquet"
    ingress = {
        str(corpus_path): _sha256(corpus_path),
        str(annotations_path): _sha256(annotations_path),
    }

    cats = load_sdg_vocabulary(corpora_dir)
    # Column-name hints: the vocabulary's domain_hypernym rows ship
    # pipe-separated example column names in example_values — SHARE-surface
    # metadata designed for name→code matching. Read straight from the
    # (already-hashed) annotations ingress.
    import pyarrow.parquet as pq

    hints = {
        r["code"]: [h.strip() for h in str(r.get("example_values") or "").split("|") if h.strip()]
        for r in pq.read_table(annotations_path).to_pylist()
    }
    vocabulary = {
        c.code: {
            "label": c.label,
            "parent_code": c.parent_code,
            "description": (c.description or "")[:240],
            "common_names": c.common_names or "",
            "hints": hints.get(c.code, [])[:8],
        }
        for c in cats.categories
    }

    samples = load_aegir_release_samples(release_dir, max_tables=max_tables)
    columns: dict[str, dict] = {}
    for table in samples:
        for col in table.columns:
            columns[col.qualified_name] = {
                "table_id": table.name,
                "column_id": col.column_id,
                "column_name": col.name,
                "column_type": col.column_type,
                "values": col.all_values[:_VALUES_HEAD],
                "n_rows": col.total_count,
                "siblings": col.siblings,
                "register": col.register,
                "name_provenance": col.name_provenance,
            }

    return {
        "metadata": {
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "release_dir": str(release_dir),
            "corpora_dir": str(corpora_dir),
            "generation_manifest": manifest,
            "ingress_sha256": ingress,
            "blind": True,
            "vocabulary_size": len(vocabulary),
            "table_count": len(samples),
            "column_count": len(columns),
        },
        "vocabulary": vocabulary,
        "columns": columns,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--release-dir", default=None)
    ap.add_argument("--corpora-dir", default=None)
    ap.add_argument("--taxonomy-id", default="sdg")
    ap.add_argument("--max-tables", type=int, default=None)
    ap.add_argument("--allow-legacy-layout", action="store_true")
    ap.add_argument("--out-root", default="build/data/agent_mediated")
    a = ap.parse_args(argv)

    from atelier.config import load_config

    cfg = load_config()
    release_dir = Path(a.release_dir or cfg.aegir_release_dir).expanduser()
    if not str(release_dir).strip("."):
        ap.error("--release-dir not given and classify.aegir.release_dir is empty")
    corpora_raw = Path(a.corpora_dir or cfg.aegir_corpora_dir).expanduser()
    corpora_dir = corpora_raw if corpora_raw.is_absolute() else REPO / corpora_raw

    ws = build_working_set(release_dir, corpora_dir,
                           max_tables=a.max_tables,
                           allow_legacy_layout=a.allow_legacy_layout)

    out_dir = REPO / a.out_root / a.taxonomy_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "working_set.json"
    out_path.write_text(json.dumps(ws, indent=1) + "\n")
    m = ws["metadata"]
    print(f"working set: {m['table_count']} tables / {m['column_count']} cols / "
          f"{m['vocabulary_size']} codes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
