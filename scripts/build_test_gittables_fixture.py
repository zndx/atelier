#!/usr/bin/env python3
# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Build the committed `test-gittables` fixture from the GitTables corpus.

Reads the raw GitTables parquet corpus (NOT committed; staged at
``/raid/datasets/gittables``), which is *self-labeling*: each parquet
carries a ``gittables`` schema-metadata blob with per-column DBpedia
semantic types (IRI + cleaned_label + description), confidence
similarities, dtypes, table license, and source csv_url.

The fixture is organized for **CCO coverage** (see
docs/src/architecture/cco-coverage.md): every leaf carries its referent
**CCO module** (Information Entity, Agent, Time, Quality, Geospatial,
Currency, Event, Artifact, Facility) and **ICE trichotomy** class
(Designative / Descriptive / Prescriptive). Atelier classifies columns,
which are always ICEs; the CCO module is the referent domain the leaf
denotes. The corpus is scanned *strided* across all ~562k files so the
selection spans domains (not just the geometry-heavy prefix).

Everything emitted is PUBLIC (DBpedia IRIs + GitTables, per the committed
PROVENANCE.md) — no customer/UAT data, by construction.

Output (committed) under src/atelier/classify/fixtures/test-gittables/:
  taxonomy.json          GT root -> CCO-module nodes -> leaves (+IRI, +ice)
  train_rows.jsonl       Row dicts for head training
  heldout_rows.jsonl     Row dicts for evaluation (+covered, +dbpedia_iri)
  enrichment_payloads.json  per-leaf label/description/prototypes/name_hints
  PROVENANCE.md          per-leaf DBpedia IRI + CCO module + per-table source

Deterministic given (corpus, --max-scan): strided file order, selection by
sorted table_id, no RNG.

Usage:
  uv run python scripts/build_test_gittables_fixture.py
  uv run python scripts/build_test_gittables_fixture.py --max-scan 16000
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from atelier.classify.cco_annotations import ground_term_annotations

# ── Configuration ────────────────────────────────────────────────────
RAID_DIR = Path("/raid/datasets/gittables")
OUT_DIR = Path("src/atelier/classify/fixtures/test-gittables")

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = ROOT / "src/atelier/classify/ontology/cco_modules.json"


def _load_cco_modules() -> dict[str, str]:
    """Canonical CCO module code -> label for ALL 11 modules, from the
    manifest. Coverage is measured against this complete set so partial
    coverage reads honestly (9/11) and Units/Relation auto-appear once an
    EAV/CPA producer contributes leaves for them."""
    m = json.loads(MANIFEST_PATH.read_text())
    return {x["code"]: x["label"] for x in m["modules"]}


def _load_cco_status() -> dict[str, str]:
    """code -> applied_status ('covered' | 'partial' | 'pending')."""
    m = json.loads(MANIFEST_PATH.read_text())
    return {x["code"]: x["applied_status"] for x in m["modules"]}


# Canonical referent modules (the grouping axis) — all 11, manifest-driven.
CCO_MODULES = _load_cco_modules()

# Curated leaf cleaned_label -> (cco_module, ice_class). ice_class:
# DES designative (names/ids/designations), DSC descriptive
# (measurements/timestamps/qualities). Only these leaves are admitted, so
# the taxonomy stays controlled and CCO-organized. QUAL is the deliberate
# maxsim-weak tail (length/width values overlap; name+siblings discriminate).
LEAF_CCO: dict[str, tuple[str, str]] = {
    # Information Entity (incl. creative works as info artifacts)
    "id": ("INFO", "DES"), "code": ("INFO", "DES"), "isbn": ("INFO", "DES"),
    "name": ("INFO", "DES"), "title": ("INFO", "DES"), "url": ("INFO", "DES"),
    "website": ("INFO", "DES"), "email": ("INFO", "DES"),
    "description": ("INFO", "DSC"), "category": ("INFO", "DSC"),
    "genre": ("INFO", "DSC"), "language": ("INFO", "DSC"),
    "book": ("INFO", "DES"), "film": ("INFO", "DES"), "album": ("INFO", "DES"),
    "song": ("INFO", "DES"),
    # Agent
    "person": ("AGENT", "DES"), "author": ("AGENT", "DES"),
    "publisher": ("AGENT", "DES"), "organisation": ("AGENT", "DES"),
    "organization": ("AGENT", "DES"), "company": ("AGENT", "DES"),
    "artist": ("AGENT", "DES"), "manufacturer": ("AGENT", "DES"),
    # Time
    "date": ("TIME", "DSC"), "datetime": ("TIME", "DSC"),
    "year": ("TIME", "DSC"), "duration": ("TIME", "DSC"), "time": ("TIME", "DSC"),
    # Quality (the value-ambiguous weak tail)
    "length": ("QUAL", "DSC"), "width": ("QUAL", "DSC"),
    "height": ("QUAL", "DSC"), "weight": ("QUAL", "DSC"),
    "mass": ("QUAL", "DSC"), "distance": ("QUAL", "DSC"),
    "depth": ("QUAL", "DSC"), "area": ("QUAL", "DSC"),
    "temperature": ("QUAL", "DSC"), "size": ("QUAL", "DSC"),
    # Geospatial
    "country": ("GEO", "DES"), "city": ("GEO", "DES"), "region": ("GEO", "DES"),
    "state": ("GEO", "DES"), "street": ("GEO", "DES"),
    "postalcode": ("GEO", "DES"), "postal code": ("GEO", "DES"),
    "address": ("GEO", "DES"), "location": ("GEO", "DES"), "place": ("GEO", "DES"),
    # Currency Unit
    "currency": ("CUR", "DES"), "price": ("CUR", "DSC"), "cost": ("CUR", "DSC"),
    # Event
    "event": ("EVENT", "DES"),
    # Artifact
    "brand": ("ARTIFACT", "DES"), "product": ("ARTIFACT", "DES"),
    "model": ("ARTIFACT", "DES"),
    # Facility
    "hotel": ("FACILITY", "DES"), "building": ("FACILITY", "DES"),
}

MAX_TYPES = 30
MIN_PER_TYPE = 11
N_TRAIN = 7
N_HELDOUT = 4
COVERED_SIM = 0.85
N_VALUES = 8
N_PROTOTYPES = 6
REF_COL_RE = re.compile(r"^(attr|code|col|data|field|item|key|ref|val|var)_\d+(_\d+)*$")


def _slug(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", label.upper())[:16] or "X"


def gittables_candidates(max_scan: int):
    """Candidate producer: GitTables CTA columns (strided parquet metadata).

    One *source producer* among several planned. Each producer yields the
    same candidate shape (dict with leaf/cco_module/ice_class/column/values/
    siblings/iri/sim/provenance), so the selection + emission below is
    source-agnostic. Future producers (additive, no rewrite):
      - sotab_cta_candidates()  — broader CTA types (value-only columns)
      - sotab_cpa_candidates()  — Extended Relation via column properties
      - eav_candidates()        — EAV tables -> Units + Extended Relation as
                                   classifiable column content (10-11/11)
    See docs/src/architecture/cco-coverage.md.
    """
    import pyarrow.parquet as pq

    all_files = sorted(RAID_DIR.glob("*.parquet"))
    stride = max(1, len(all_files) // max_scan)
    files = all_files[::stride][:max_scan]
    cands: dict[str, list[dict]] = defaultdict(list)
    scanned = 0
    for f in files:
        try:
            md = pq.ParquetFile(f).schema_arrow.metadata or {}
            blob = md.get(b"gittables")
            if not blob:
                continue
            gt = json.loads(blob)
            sem = gt.get("dbpedia_semantic_column_types") or {}
            sims = gt.get("dbpedia_semantic_similarities") or {}
            dtypes = gt.get("dtypes") or {}
            table_id = gt.get("table_id")
            siblings = list(dtypes.keys())
            for col, info in sem.items():
                lbl = (info.get("cleaned_label") or "").strip().lower()
                cco = LEAF_CCO.get(lbl)
                if cco is None or REF_COL_RE.match(col):
                    continue
                cands[f"{cco[0]}:{lbl}"].append({
                    "file": str(f), "table_id": table_id, "column": col,
                    "leaf": lbl, "cco_module": cco[0], "ice_class": cco[1],
                    "iri": info.get("id"), "description": info.get("description"),
                    "sim": float(sims.get(col, 0.0)),
                    "column_type": dtypes.get(col, "object"),
                    "siblings": [s for s in siblings if s != col],
                    "license": gt.get("license"), "csv_url": gt.get("csv_url"),
                })
            scanned += 1
        except Exception:
            continue
    return cands, scanned, len(all_files), len(files)


def read_values(file: str, column: str):
    import pyarrow.parquet as pq
    try:
        tbl = pq.read_table(file, columns=[column])
        out, seen = [], set()
        for v in tbl.column(0).to_pylist():
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.lower() in ("none", "nan") or s in seen:
                continue
            seen.add(s)
            out.append(s[:80])
            if len(out) >= N_VALUES:
                break
        return out
    except Exception:
        return []


def _cand_values(c: dict) -> list[str]:
    """Pre-loaded values (CPA / json.gz producers) or lazy parquet read
    (GitTables). Producer-agnostic so the row-building below is uniform."""
    if c.get("sample_values") is not None:
        return list(c["sample_values"])[:N_VALUES]
    return read_values(c["file"], c["column"])


# ── SOTAB CPA producer: Extended Relation (REL) data face ─────────────
SOTAB_DIR = Path("/raid/datasets/sotab")
_CPA_CSV = SOTAB_DIR / "sotab_cpa_train_dbpedia.csv"
_CPA_TABLES = SOTAB_DIR / "Train"
# Curated CPA relations -> REL leaves. Each is a DBpedia property a column
# expresses about the table's main entity — the relational data CTA cannot
# reach. (Some labels coincide with CTA types, e.g. price/author, but here
# they are RELATIONS, not entity types — hence the module-namespaced keys.)
CPA_RELATIONS = {
    "author", "publisher", "director", "artist", "price", "currency",
    "publicationDate", "releaseDate", "country", "city", "rating", "brand",
}


def _load_cpa_column(table_path: Path, col_index: int) -> tuple[list[str], str]:
    import pandas as pd
    try:
        df = pd.read_json(table_path, compression="gzip", lines=True)
    except Exception:
        return [], "object"
    if col_index >= df.shape[1]:
        return [], "object"
    out, seen = [], set()
    for v in df.iloc[:, col_index].tolist():
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() in ("none", "nan") or s in seen:
            continue
        seen.add(s)
        out.append(s[:80])
        if len(out) >= N_VALUES:
            break
    return out, str(df.dtypes.iloc[col_index])


def sotab_cpa_candidates(max_per_rel: int = 14) -> dict:
    """Producer: SOTAB CPA columns -> Extended Relation (REL) leaves.

    Each column expresses a relation (DBpedia property) to the table's main
    entity. This is the data face of the Extended Relation module — CPA, not
    CTA — that legitimately moves REL from annotation-partial to data-covered.
    """
    if not _CPA_CSV.exists():
        return {}
    by_rel: dict[str, list] = defaultdict(list)
    for line in _CPA_CSV.read_text().splitlines()[1:]:
        p = line.split(",")
        if len(p) < 4:
            continue
        table, _main_i, col_i, iri = p[0], p[1], p[2], p[3]
        rel = iri.rsplit("/", 1)[-1]
        if rel not in CPA_RELATIONS:
            continue
        try:
            by_rel[rel].append((table, int(col_i), iri))
        except ValueError:
            continue
    cands: dict[str, list[dict]] = defaultdict(list)
    for rel, entries in by_rel.items():
        entries.sort(key=lambda e: (e[0], e[1]))
        seen_t: set[str] = set()
        for table, col_i, iri in entries:
            if table in seen_t:
                continue
            seen_t.add(table)
            tpath = _CPA_TABLES / table
            if not tpath.exists():
                continue
            vals, ctype = _load_cpa_column(tpath, col_i)
            if not vals:
                continue
            entity = table.split("_", 1)[0]
            cands[f"REL:{rel}"].append({
                "code": f"SDG.REL.{_slug(rel)}", "leaf": rel,
                "cco_module": "REL", "ice_class": None, "iri": iri,
                "description": f"DBpedia relation '{rel}' to the {entity} subject.",
                "table_id": table.replace(".json.gz", ""), "column": f"c{col_i}",
                "column_type": ctype, "sample_values": vals, "siblings": [entity],
                "sim": 1.0,  # CPA labels are ground truth
                "license": "WDC SOTAB v2 (research use)",
                "csv_url": "https://webdatacommons.org/structureddata/sotab/",
            })
            if len(cands[f"REL:{rel}"]) >= max_per_rel:
                break
    return cands


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scan", type=int, default=14000)
    args = ap.parse_args()

    if not RAID_DIR.exists():
        print(f"ERROR: {RAID_DIR} not found (need the GitTables corpus on /raid)")
        return 1

    print(f"scanning ~{args.max_scan} strided parquet files in {RAID_DIR} ...")
    # Producer fan-in: GitTables CTA + SOTAB CPA (Extended Relation). Each
    # producer yields the same candidate shape keyed by "<MODULE>:<label>";
    # downstream selection/leaf-building read the module from the candidate.
    cands, scanned, total, sampled = gittables_candidates(args.max_scan)
    print(f"scanned {scanned} labeled parquets ({sampled} sampled of {total} total)")
    cpa = sotab_cpa_candidates()
    for k, v in cpa.items():
        cands[k].extend(v)
    print(f"  + CPA producer: {len(cpa)} relation leaves for the REL module")

    # Keep leaves with enough candidates; prefer CCO-module diversity:
    # round-robin one type per module before filling, so every module that
    # CAN be covered gets at least one leaf.
    eligible = {k: c for k, c in cands.items() if len(c) >= MIN_PER_TYPE}
    by_mod: dict[str, list[str]] = defaultdict(list)
    for k in eligible:
        by_mod[cands[k][0]["cco_module"]].append(k)
    for m in by_mod:
        by_mod[m].sort(key=lambda kk: -len(eligible[kk]))
    chosen: list[str] = []
    while len(chosen) < MAX_TYPES and any(by_mod.values()):
        for m in sorted(by_mod):
            if by_mod[m] and len(chosen) < MAX_TYPES:
                chosen.append(by_mod[m].pop(0))
    chosen = sorted(chosen, key=lambda kk: (cands[kk][0]["cco_module"], kk))
    mods_covered = sorted({cands[kk][0]["cco_module"] for kk in chosen})
    print(f"eligible types: {len(eligible)}; selected {len(chosen)} across "
          f"{len(mods_covered)} CCO modules {mods_covered}")
    print(f"  leaves: {chosen}")
    if len(mods_covered) < 4:
        print("Too few CCO modules — increase --max-scan.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modules_used: dict[str, str] = {}
    leaves: list[dict] = []
    train_rows: list[dict] = []
    heldout_rows: list[dict] = []
    enrich: dict[str, dict] = {}
    prov_rows: list[dict] = []

    # Verified CCO ICE-trichotomy IRIs (published develop catalog,
    # InformationEntityOntology.ttl) — NOT fabricated.
    ICE_FULL = {"DES": "DesignativeICE", "DSC": "DescriptiveICE",
                "PRE": "PrescriptiveICE"}
    ICE_IRI = {"DES": "https://www.commoncoreontologies.org/ont00000686",
               "DSC": "https://www.commoncoreontologies.org/ont00000853",
               "PRE": "https://www.commoncoreontologies.org/ont00000965"}
    sdg_requirements: list[dict] = []
    for key in chosen:
        c0 = cands[key][0]
        cco = c0["cco_module"]
        ice = c0.get("ice_class")
        lbl = c0["leaf"]
        modules_used[cco] = CCO_MODULES[cco]
        code = c0.get("code") or f"SDG.{cco}.{_slug(lbl)}"
        iri = next((c["iri"] for c in cands[key] if c.get("iri")), None)
        desc = next((c["description"] for c in cands[key] if c.get("description")), None)
        # SDG grounding: a CTA leaf is a value type (sdg:hasValueType); a CPA
        # (REL) leaf is a relation (sdg:describesProperty) — see
        # docs/src/architecture/sdg-fixture-grounding.md. Both ground in CCO.
        sdg_property = "describesProperty" if cco == "REL" else "hasValueType"
        sdg_term = f"sdg:{lbl}"
        leaves.append({"code": code, "label": lbl, "parent_code": f"SDG.{cco}",
                       "dbpedia_iri": iri, "cco_module": cco,
                       "cco_module_label": CCO_MODULES[cco],
                       "ice_class": ICE_FULL.get(ice),
                       "ice_class_iri": ICE_IRI.get(ice),
                       "sdg_property": sdg_property, "sdg_term": sdg_term,
                       # CCO ExtendedRelationOntology annotations this term
                       # satisfies (acronym, definition_source). has_token_unit
                       # stays unset (the unit is the semantic absence).
                       "cco_annotations": ground_term_annotations(
                           mnemonic=_slug(lbl), dbpedia_iri=iri),
                       "is_leaf": True})
        sdg_requirements.append({
            "sdg_term": sdg_term, "label": lbl, "sdg_property": sdg_property,
            "cco_module": cco, "cco_module_label": CCO_MODULES[cco],
            "ice_class": ICE_FULL.get(ice), "ice_class_iri": ICE_IRI.get(ice),
            "definition_source": iri, "status": "proposed-extension",
        })

        picks = sorted(cands[key], key=lambda c: (str(c["table_id"]), str(c["column"])))
        seen, uniq = set(), []
        for c in picks:
            k = (c["table_id"], c["column"])
            if k not in seen:
                seen.add(k)
                uniq.append(c)
        train_c = uniq[:N_TRAIN]
        held_c = uniq[N_TRAIN:N_TRAIN + N_HELDOUT]

        proto, hints = [], set()
        for c in train_c:
            vals = _cand_values(c)
            hints.add(c["column"])
            for v in vals[:3]:
                if v not in proto and len(proto) < N_PROTOTYPES:
                    proto.append(v)
            train_rows.append({
                "table": str(c["table_id"]), "column": c["column"],
                "column_type": c["column_type"], "sample_values": vals,
                "siblings_full": c["siblings"], "mnemonic": _slug(lbl), "code": code,
            })
            prov_rows.append({"code": code, "table_id": c["table_id"],
                              "license": c["license"], "csv_url": c["csv_url"]})
        for c in held_c:
            vals = _cand_values(c)
            hints.add(c["column"])
            heldout_rows.append({
                "table": str(c["table_id"]), "column": c["column"],
                "column_type": c["column_type"], "sample_values": vals,
                "siblings_full": c["siblings"], "mnemonic": _slug(lbl), "code": code,
                "dbpedia_iri": c["iri"], "covered": bool(c["sim"] >= COVERED_SIM),
            })

        enrich[code] = {
            "code": code, "label": lbl, "mnemonic": _slug(lbl),
            "cco_module": cco,
            "description": desc or f"DBpedia term '{lbl}' ({iri}).",
            "prototype_values": proto[:N_PROTOTYPES],
            "name_hints": sorted(hints), "value_patterns": [],
        }

    taxonomy = {
        "taxonomy_id": "test-gittables",
        # SDG-grounded root: an Information Content Entity (cco:ont00000958,
        # verified). The taxonomy is a test-scoped subset of SDG, NOT a
        # standalone namespace — see docs/src/architecture/sdg-fixture-grounding.md.
        "root": {"code": "SDG", "label": "SDG Data Element",
                 "cco_iri": "https://www.commoncoreontologies.org/ont00000958",
                 "parent_code": None},
        "internal": [{"code": f"SDG.{m}", "label": lab, "parent_code": "SDG",
                      "cco_module": lab}
                     for m, lab in sorted(modules_used.items())],
        "leaves": leaves,
    }

    (OUT_DIR / "taxonomy.json").write_text(json.dumps(taxonomy, indent=2) + "\n")
    with (OUT_DIR / "train_rows.jsonl").open("w") as fh:
        for r in train_rows:
            fh.write(json.dumps(r) + "\n")
    with (OUT_DIR / "heldout_rows.jsonl").open("w") as fh:
        for r in heldout_rows:
            fh.write(json.dumps(r) + "\n")
    (OUT_DIR / "enrichment_payloads.json").write_text(json.dumps(enrich, indent=2) + "\n")
    # The requirements artifact Aegir consumes: the value-type / relation terms
    # this fixture needs, as proposed SDG extensions grounded in verified CCO.
    (OUT_DIR / "sdg_requirements.json").write_text(json.dumps({
        "taxonomy_id": "test-gittables",
        "note": "Proposed SDG terms (value types via sdg:hasValueType, "
                "relations via sdg:describesProperty) the fixture exercises. "
                "For Aegir to adopt or refine into sdg-vocab.ttl + catalogs. "
                "CCO IRIs verified against the published develop catalog.",
        "terms": sorted(sdg_requirements, key=lambda t: (t["cco_module"], t["sdg_term"])),
    }, indent=2) + "\n")

    cov = sum(1 for r in heldout_rows if r["covered"])
    lines = [
        "<!-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved. -->",
        "", "# test-gittables Fixture Provenance", "",
        "PUBLIC, non-target-domain fixture for the SVM/maxsim critical-path",
        "test, organized for **CCO coverage** (see",
        "docs/src/architecture/cco-coverage.md). Source: the **GitTables**",
        "corpus (per-column DBpedia semantic types from each table's embedded",
        "`gittables` metadata), scanned strided across the full corpus. Types",
        "are **DBpedia ontology** classes/properties; each leaf is tagged with",
        "its referent **CCO module** and **ICE trichotomy** class. Per-table",
        "data carries its own upstream license (below). No customer/UAT data —",
        "by construction (public GitTables only; answer-key names excluded).", "",
        f"- CCO modules covered: {len(modules_used)}/{len(CCO_MODULES)} "
        f"({', '.join(sorted(modules_used.values()))})",
        f"- leaf types: {len(leaves)}",
        f"- train rows: {len(train_rows)}  |  held-out: {len(heldout_rows)} "
        f"(covered: {cov}, weak: {len(heldout_rows) - cov})", "",
        "## Leaf types → CCO module / ICE class / DBpedia IRI", "",
        "| Code | Label | CCO module | ICE class | DBpedia IRI |",
        "|---|---|---|---|---|",
    ]
    for lf in leaves:
        lines.append(f"| `{lf['code']}` | {lf['label']} | {lf['cco_module_label']} | "
                     f"{lf['ice_class']} | {lf['dbpedia_iri']} |")
    lines += ["", "## Per-table upstream source + license", "",
              "| Table ID | License | Source CSV |", "|---|---|---|"]
    seen_t = set()
    for p in prov_rows:
        if p["table_id"] in seen_t:
            continue
        seen_t.add(p["table_id"])
        lines.append(f"| {p['table_id']} | {p['license']} | {p['csv_url']} |")
    (OUT_DIR / "PROVENANCE.md").write_text("\n".join(lines) + "\n")

    status = _load_cco_status()
    not_data = set(CCO_MODULES) - set(modules_used)
    partial = sorted(c for c in not_data if status.get(c) == "partial")
    pending = sorted(c for c in not_data if status.get(c) == "pending")
    print(f"\nwrote fixture -> {OUT_DIR}")
    print(f"  CCO modules: {len(modules_used)}/{len(CCO_MODULES)} data-covered "
          f"{sorted(modules_used)}")
    print(f"    partial (annotation-grounded, no data leaves yet): {partial}")
    print(f"    pending (EAV/CPA-gated): {pending}")
    print(f"  leaves: {len(leaves)}  train: {len(train_rows)}  "
          f"heldout: {len(heldout_rows)} (covered {cov} / weak {len(heldout_rows) - cov})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
