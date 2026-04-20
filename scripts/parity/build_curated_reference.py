#!/usr/bin/env python
"""Build the curated reference (per-column) for the synthetic corpus.

Walks every natural-named column in the UAT snapshot under
``build/meta-tagging/`` and emits exactly one reference row per column
with explicit derivation provenance.  The artifact is deliberately
named a **curated reference**: it is a generator-truth record (the
synth generator stuffed the correct code into each reference-column
twin) that we review and curate by hand as needed for quality checks.
External, human-curated benchmarks (e.g. the published SOTAB
annotations) are a distinct class of label and out of scope here.

Reference columns (paired twins) are answer keys, not inputs; they
never appear as reference rows themselves — their encoded code
becomes the reference for the natural-named column immediately
preceding them in schema order.

Output:

    build/meta-tagging-clean/curated_reference.csv
        table, column, reference_code, reference_label,
        derivation, confidence

    build/meta-tagging-clean/curated_reference_summary.json
        counts per derivation, per-class distribution, invariant
        audits (reference-column exclusion, unresolved tail).

Derivation provenance (one per natural-named column):

    reference_encoded   natural-named col paired with reference twin;
                        code from twin's suffix.  Generator-truth —
                        synth generator guaranteed the pairing.
    ontology_match      name maps to a unique Ontology field.  High.
    annotation_match    name maps to Annotation mnemonic.          High.
    common_names_match  Common Names alias; deepest code wins.     Medium.
    row_id_fallback     literal ``row_id`` → ``0.1``.               Convention.
    unresolved          no match found; excluded from scoring.       Unknown.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path


log = logging.getLogger("build_curated_reference")


def _normalize(name: str) -> str:
    import re
    s = re.sub(r"[^a-z0-9_]", "_", name.lower().strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _strip_table_prefix(header: list[str], table: str) -> list[str]:
    """Drop uniform ``<table>.`` prefix if every header shares it."""
    prefix = f"{table}."
    if all(h.startswith(prefix) for h in header if h):
        return [h[len(prefix):] for h in header]
    return header


def _decode_reference_code(ref_name: str) -> str | None:
    """Extract the encoded code from a reference-column name.

    ``attr_1_1_1_9_2_1`` → ``1.1.1.9.2.1``.  Applies the UAT root-level
    ``0.0`` → ``0`` normalization so the decoded code matches the
    vocabulary IDs in ``annotations.csv``.
    """
    from atelier.classify.meta_tagging_source import _REFERENCE_COL_RE
    m = _REFERENCE_COL_RE.match(ref_name)
    if not m:
        return None
    code = m.group(1).replace("_", ".")
    if "." in code and code.count(".") == 1 and code.endswith(".0"):
        code = code[:-2]
    return code


def _build_typed_name_index(records: list[dict]) -> dict[str, tuple[str, str]]:
    """Map normalized alias → (code, derivation_source).

    Mirrors ``meta_tagging_source._build_name_to_code_index`` (priority
    Ontology > Annotation > Common Names; depth-winning tie-break) but
    retains the winning source so GT rows can record the derivation.
    """
    import re
    SOURCE_TIER = [
        ("ontology", 0, "ontology_match"),
        ("annotation", 1, "annotation_match"),
        ("common_names", 2, "common_names_match"),
    ]
    tiered: dict[str, tuple[int, int, str, str]] = {}
    for r in records:
        code = r.get("id")
        if not code:
            continue
        depth = code.count(".")
        for key, tier, derivation in SOURCE_TIER:
            value = r.get(key)
            if not value:
                continue
            for token in re.split(r"[|,]", value):
                norm = _normalize(token)
                if not norm:
                    continue
                existing = tiered.get(norm)
                if (
                    existing is None
                    or tier < existing[0]
                    or (tier == existing[0] and depth > existing[1])
                ):
                    tiered[norm] = (tier, depth, code, derivation)
    return {n: (code, deriv) for n, (_, _, code, deriv) in tiered.items()}


def _resolve_natural_column(
    name: str, typed_index: dict, fallback: str
) -> tuple[str, str, str]:
    """Resolve (code, derivation, confidence) for one natural-named column."""
    if name == "row_id":
        return (fallback, "row_id_fallback", "authoritative")
    hit = typed_index.get(_normalize(name))
    if hit is None:
        return ("", "unresolved", "unknown")
    code, derivation = hit
    confidence = {
        "ontology_match": "high",
        "annotation_match": "high",
        "common_names_match": "medium",
    }[derivation]
    return (code, derivation, confidence)


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

    out_dir = Path("build/meta-tagging-clean").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vocabulary — records for typed name index + id → label lookup
    from atelier.classify.meta_tagging_source import (
        _read_annotation_rows, _fallback_code, _REFERENCE_COL_RE,
    )
    records = _read_annotation_rows(src_dir / "annotations.csv")
    typed_index = _build_typed_name_index(records)
    id_to_label = {r["id"]: (r.get("ontology") or "") for r in records}
    fallback = _fallback_code(records)

    rows: list[dict] = []
    per_derivation: dict[str, int] = {}
    per_table_ref_drops: dict[str, int] = {}

    for csv_path in sorted(src_dir.glob("*.csv")):
        if csv_path.name == "annotations.csv":
            continue
        table = csv_path.stem
        per_table_ref_drops.setdefault(table, 0)
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
        if not header:
            continue
        stripped = _strip_table_prefix(header, table)

        # Walk columns; a reference column contributes GT to its
        # immediately-preceding natural-named neighbor.
        for name in stripped:
            if _REFERENCE_COL_RE.match(name):
                per_table_ref_drops[table] += 1
                # attach this code to the previously-emitted row
                code = _decode_reference_code(name)
                if code and rows and rows[-1]["table"] == table:
                    prev = rows[-1]
                    # only override prior derivations — reference
                    # evidence is authoritative.
                    prev["reference_code"] = code
                    prev["reference_label"] = id_to_label.get(code, "")
                    prev["derivation"] = "reference_encoded"
                    prev["confidence"] = "authoritative"
                    # rebalance the histogram
                    old = prev.get("_prev_deriv")
                    if old and old != "reference_encoded":
                        per_derivation[old] = per_derivation.get(old, 1) - 1
                        if per_derivation[old] <= 0:
                            del per_derivation[old]
                    per_derivation["reference_encoded"] = (
                        per_derivation.get("reference_encoded", 0) + 1
                    )
                    prev["_prev_deriv"] = "reference_encoded"
                continue

            code, derivation, confidence = _resolve_natural_column(
                name, typed_index, fallback,
            )
            row = {
                "table": table,
                "column": name,
                "reference_code": code,
                "reference_label": id_to_label.get(code, "") if code else "",
                "derivation": derivation,
                "confidence": confidence,
                "_prev_deriv": derivation,
            }
            rows.append(row)
            per_derivation[derivation] = per_derivation.get(derivation, 0) + 1

    # Drop the bookkeeping field before writing.
    for r in rows:
        r.pop("_prev_deriv", None)

    csv_out = out_dir / "curated_reference.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "table", "column", "reference_code",
                "reference_label", "derivation", "confidence",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Per-class distribution
    per_class: dict[str, int] = {}
    for r in rows:
        c = r["reference_code"]
        if c:
            per_class[c] = per_class.get(c, 0) + 1

    summary = {
        "total_reference_rows": len(rows),
        "per_derivation": dict(sorted(per_derivation.items(), key=lambda kv: -kv[1])),
        "per_table_reference_columns_dropped": per_table_ref_drops,
        "reference_columns_dropped_total": sum(per_table_ref_drops.values()),
        "unique_classes_covered": len(per_class),
        "per_class_distribution_head": dict(
            sorted(per_class.items(), key=lambda kv: -kv[1])[:25]
        ),
        "unresolved_columns": [
            f"{r['table']}.{r['column']}"
            for r in rows if r["derivation"] == "unresolved"
        ],
    }
    summary_out = out_dir / "curated_reference_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2))

    print(f"\n=== curated reference ===")
    print(f"  rows written    : {len(rows)}  (natural-named columns only)")
    print(f"  reference drops : {summary['reference_columns_dropped_total']}")
    print(f"  derivations     : {summary['per_derivation']}")
    print(f"  unique classes  : {summary['unique_classes_covered']}")
    print(f"  unresolved      : {len(summary['unresolved_columns'])}")
    if summary["unresolved_columns"]:
        print(f"    examples: {summary['unresolved_columns'][:5]}")
    print(f"\n  csv     : {csv_out}")
    print(f"  summary : {summary_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
