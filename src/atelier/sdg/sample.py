"""Optimized SDG sample strategy — the first-run classification substrate.

Derives a *relationally sound* (referential-integrity-verified) and
*logically sound* (genus-bearing, root-diverse, ancestor-closed) sample
from the pinned sdg-corpora submodule, sized by hardware profile. The
sample is what every Atelier deployment classifies on first run — svelte
enough for a laptop, parameterized for larger systems.

Why collections, not the legacy DDL tables
------------------------------------------
The corpus ships two relational surfaces.  The 520-table legacy DDL
footprint (``ddl/<run>/``) is vestigial relative to the newer
topic-domain collections: it declares its relations *ontologically*
(``sdg:ForeignKeyColumn`` → ``references_class``) but does not
value-join them — measured on pin b24ef9f6: 0 of 25 sampled
cross-entity edges satisfied value inclusion, and most referenced
classes have no realizing base table.  The per-topic **collections**
(``corpus/collections/<slug>/``) are "deterministic, referentially-
intact relational schema" by construction (corpus CARD) — and measure
clean: every convention-inferred FK edge value-joins with zero
orphans.  Each collection also carries an authoritative
``manifest.json`` naming its entities and their **BFO/CCO genus
IRIs** — exactly the genus/root signal the taxonomy sample needs.

Selection model
---------------
Collections are the sampling unit (each is an internally FK-closed
bundle).  Greedy selection maximizes *new genus-anchor coverage* first
(diverse roots), then term/column richness, within the profile's
column budget.  Referential integrity is then **verified**, not
assumed: every ``<entity>_id`` column that resolves to a sibling table
must value-join with zero orphans, or the build fails loudly.
Columns that resolve to no table are recorded as designative
references — a positively-represented absence, never a silent skip.

The SKOS subset = genus anchors of every sampled entity + direct
entity→term matches + column-concept matches (``SDG.DOM.*``), closed
over ``parent_code``.  Entity names with no SKOS term are reported as
``vocabulary_gaps`` — the data-grounded upstream feedback for the next
corpus iteration.

Artifacts land under ``<artifact_root>/sdg_sample/<pin>_<profile>/``
with a manifest carrying the corpus pin, profile, per-collection
provenance, RI report, taxonomy stats, and vocab signature — aligned
to the ``sdg-corpora`` data-source id.

Usage::

    python -m atelier.sdg.sample --profile macbook
    python -m atelier.sdg.sample --profile workstation --max-terms 300
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CORPORA = _REPO_ROOT / "external" / "sdg-corpora"

SOURCE_ID = "sdg-corpora"  # the data-source id every artifact aligns to


class SdgSampleError(RuntimeError):
    """Raised for any condition that makes the sample unsound."""


# ── Profiles ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class SampleProfile:
    """Hardware-scaled sampling parameters.

    ``max_columns`` is the primary budget (LLM sweep cost is columns ×
    vocabulary); ``max_terms`` bounds the SKOS subset (enrichment is
    per-term LLM work); ``min_roots`` forces distinct BFO/CCO genus
    anchors — diverse roots by construction, not by luck.
    """

    name: str
    max_collections: int
    max_columns: int
    max_terms: int
    min_roots: int


PROFILES: dict[str, SampleProfile] = {
    # Svelte default: one enrich→classify arc on an Apple-silicon
    # laptop with a local Nemotron-class model, in one sitting.
    "macbook": SampleProfile("macbook", max_collections=3, max_columns=150,
                             max_terms=120, min_roots=4),
    "workstation": SampleProfile("workstation", max_collections=10,
                                 max_columns=500, max_terms=320, min_roots=6),
    # Wide open — multi-GPU hosts; the corpus itself is the cap.
    "cluster": SampleProfile("cluster", max_collections=453,
                             max_columns=100_000, max_terms=2000, min_roots=8),
}


# ── Genus IRI → SKOS upper anchor ────────────────────────────────
#
# Collection manifests ground each entity in a BFO/CCO genus IRI; the
# SKOS vocabulary roots at 13 upper anchors.  This is the bridge —
# unmapped IRIs land on SDG.GENERIC and are reported, never dropped.

_GENUS_TO_ANCHOR: dict[str, str] = {
    "bfo:0000015": "SDG.PROCESS",
    "bfo:0000004": "SDG.INDEPENDENT_CONTINUANT",
    "bfo:0000040": "SDG.MATERIAL_ENTITY",
    "bfo:0000031": "SDG.GDC",
    "bfo:0000019": "SDG.QUALITY",
    "bfo:0000023": "SDG.ROLE",
    "bfo:0000016": "SDG.DISPOSITION",
    "bfo:0000002": "SDG.INDEPENDENT_CONTINUANT",  # continuant (super)
    "cco:ont00000995": "SDG.ARTIFACT",
    "cco:ont00000958": "SDG.ICE",
    "cco:ont00000853": "SDG.ICE.DESCRIPTIVE",
    "cco:ont00000965": "SDG.ICE.DIRECTIVE",
    "cco:ont00000686": "SDG.ICE.DESIGNATIVE",
}
_FALLBACK_ANCHOR = "SDG.GENERIC"


# ── Corpus loading ───────────────────────────────────────────────

def _corpus_commit(corpora: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(corpora), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception as exc:
        raise SdgSampleError(
            f"Cannot resolve sdg-corpora pin via git ({exc}) — the sample "
            f"manifest requires the corpus commit for provenance."
        ) from exc


@dataclass
class Collection:
    slug: str
    path: Path
    manifest: dict
    anchors: set[str] = field(default_factory=set)
    column_count: int = 0
    unmapped_genera: list[str] = field(default_factory=list)


def _load_collections(corpora: Path) -> list[Collection]:
    base = corpora / "corpus" / "collections"
    if not base.is_dir():
        raise SdgSampleError(
            f"No collections at {base} — is the sdg-corpora submodule "
            f"initialized? (git submodule update --init)"
        )
    out: list[Collection] = []
    for d in sorted(base.iterdir()):
        mf = d / "manifest.json"
        if not d.is_dir() or not mf.is_file():
            continue
        try:
            manifest = json.loads(mf.read_text())
        except Exception as exc:
            logger.warning("Skipping %s: unreadable manifest (%s)", d.name, exc)
            continue
        coll = Collection(slug=d.name, path=d, manifest=manifest)
        terms = manifest.get("terms", [])
        if terms and isinstance(terms[0], str):
            raise SdgSampleError(
                f"Collection {d.name} uses the DDL-only manifest schema "
                f"(terms as template-id strings, no populated tables) — "
                f"this corpus pin ships no instantiated rows, so a "
                f"relationally-sound sample cannot be built from it.  "
                f"Pin sdg-corpora to a release with populated "
                f"collections (e.g. b24ef9f6) or adapt the sampler to "
                f"the blind-release emission format."
            )
        for term in terms:
            genus = term.get("genus", "")
            anchor = _GENUS_TO_ANCHOR.get(genus)
            if anchor is None:
                coll.unmapped_genera.append(genus)
                anchor = _FALLBACK_ANCHOR
            coll.anchors.add(anchor)
        coll.column_count = sum(
            len(t.get("columns", [])) for t in manifest.get("tables", []))
        out.append(coll)
    if not out:
        raise SdgSampleError(f"No readable collections under {base}")
    return out


@dataclass
class Vocab:
    rows: dict[str, dict]
    by_abbrev: dict[str, str]
    by_label: dict[str, str]
    children: dict[str, list[str]]

    def lineage(self, code: str) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        current = code
        while current and current not in seen and current in self.rows:
            out.append(current)
            seen.add(current)
            current = (self.rows[current].get("parent_code") or "").strip()
        return out

    def root_of(self, code: str) -> str:
        lin = self.lineage(code)
        return lin[-1] if lin else code


def _load_vocab(corpora: Path) -> Vocab:
    path = corpora / "vocabulary" / "annotations.csv"
    rows: dict[str, dict] = {}
    by_abbrev: dict[str, str] = {}
    by_label: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            code = (row.get("code") or "").strip()
            if not code:
                continue
            rows[code] = row
            ab = (row.get("abbrev") or "").strip()
            if ab:
                by_abbrev.setdefault(ab.upper(), code)
            label = (row.get("label") or "").strip()
            if label:
                by_label.setdefault(_screaming(label), code)
            parent = (row.get("parent_code") or "").strip()
            if parent:
                children.setdefault(parent, []).append(code)
    if not rows:
        raise SdgSampleError(f"No vocabulary rows parsed from {path}")
    return Vocab(rows=rows, by_abbrev=by_abbrev, by_label=by_label,
                 children=children)


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _screaming(name: str) -> str:
    return _CAMEL.sub("_", name).replace("-", "_").replace(" ", "_").upper()


# ── Selection ────────────────────────────────────────────────────

def select_collections(
    collections: list[Collection], profile: SampleProfile,
) -> list[Collection]:
    """Greedy: new-anchor coverage first, then term richness per column.

    A collection is admitted only when its whole column footprint fits
    the remaining budget — bundles are never split (splitting would
    break the very referential closure the sample exists to provide).
    """
    selected: list[Collection] = []
    covered: set[str] = set()
    columns_used = 0

    def gain(c: Collection) -> tuple:
        new = len(c.anchors - covered)
        richness = len(c.manifest.get("terms", []))
        return (new, richness, -c.column_count)

    remaining = [c for c in collections if c.column_count > 0]
    while len(selected) < profile.max_collections and remaining:
        remaining.sort(key=gain, reverse=True)
        placed = False
        for c in remaining:
            if columns_used + c.column_count > profile.max_columns:
                continue
            selected.append(c)
            covered |= c.anchors
            columns_used += c.column_count
            remaining.remove(c)
            placed = True
            break
        if not placed:
            break

    if len(covered) < profile.min_roots:
        raise SdgSampleError(
            f"Root diversity unsatisfiable: {len(covered)} genus anchors "
            f"covered ({sorted(covered)}); profile {profile.name!r} requires "
            f"{profile.min_roots}.  Raise max_columns/max_collections or "
            f"lower min_roots."
        )
    return selected


# ── Referential-integrity verification ───────────────────────────

def _read_tables(coll: Collection) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    for p in sorted((coll.path / "tables").glob("*.csv")):
        with open(p, newline="", encoding="utf-8") as f:
            tables[p.stem] = list(csv.DictReader(f))
    if not tables:
        raise SdgSampleError(f"Collection {coll.slug} has no tables/*.csv")
    return tables


def _key_column(stem: str, rows: list[dict]) -> str | None:
    if not rows:
        return None
    cols = rows[0].keys()
    if "id" in cols:
        return "id"
    singular = stem[:-1] if stem.endswith("s") else stem
    candidate = f"{singular}_id"
    return candidate if candidate in cols else None


def verify_collection_ri(coll: Collection,
                         tables: dict[str, list[dict]]) -> dict:
    """Value-join every convention-inferred FK edge; zero orphans or die.

    ``<x>_id`` columns resolving to a sibling table (``x``/``x+'s'``/
    ``x+'es'``) are joins and must be referentially intact.  Ones that
    resolve to nothing are *designative references* — recorded, not
    checked, never silently treated as sound.
    """
    edges: list[dict] = []
    designative: list[str] = []
    orphan_total = 0
    for stem, rows in tables.items():
        if not rows:
            continue
        for col in rows[0].keys():
            m = re.match(r"(.+)_id$", col)
            if not m:
                continue
            target = None
            for cand in (m.group(1), m.group(1) + "s", m.group(1) + "es"):
                if cand in tables and cand != stem:
                    target = cand
                    break
            if target is None:
                key = _key_column(stem, rows)
                if col != key:  # own key column is not a reference
                    designative.append(f"{coll.slug}:{stem}.{col}")
                continue
            key = _key_column(target, tables[target])
            if key is None:
                designative.append(f"{coll.slug}:{stem}.{col}(→{target}, no key)")
                continue
            valid = {r.get(key) for r in tables[target]}
            vals = [r[col] for r in rows if r.get(col)]
            orphans = [v for v in vals if v not in valid]
            orphan_total += len(orphans)
            edges.append({
                "collection": coll.slug, "table": stem, "column": col,
                "references": target, "key": key,
                "checked": len(vals), "orphans": len(orphans),
            })

    if orphan_total:
        bad = [e for e in edges if e["orphans"]]
        raise SdgSampleError(
            f"Referential integrity FAILED in collection {coll.slug}: "
            f"{orphan_total} orphan value(s) across {len(bad)} edge(s).  "
            f"First: {bad[0]}.  Refusing to write an unsound sample."
        )
    return {"edges": edges, "designative_references": designative,
            "orphan_total": orphan_total}


# ── Taxonomy subset ──────────────────────────────────────────────

def select_terms(
    selected: list[Collection],
    tables_by_coll: dict[str, dict[str, list[dict]]],
    vocab: Vocab,
    profile: SampleProfile,
) -> tuple[list[str], dict[str, int], list[str], list[str]]:
    """Genus anchors + entity matches + column-concept matches, closed.

    Returns (codes, support, vocabulary_gaps, matched_examples).
    Entity names without a SKOS term are the ``vocabulary_gaps`` the
    next corpus iteration should close — reported, never invented.
    """
    support: dict[str, int] = {}
    gaps: list[str] = []
    matched: list[str] = []

    def _include(code: str, weight: int = 1) -> None:
        for ancestor in vocab.lineage(code):
            support[ancestor] = support.get(ancestor, 0)
        support[code] = support.get(code, 0) + weight

    def _lookup(name: str) -> str | None:
        key = _screaming(name)
        return vocab.by_abbrev.get(key) or vocab.by_label.get(key)

    data_matched: set[str] = set()
    column_gaps: set[str] = set()

    for coll in selected:
        for term in coll.manifest.get("terms", []):
            name = term.get("name", "")
            anchor = _GENUS_TO_ANCHOR.get(term.get("genus", ""), _FALLBACK_ANCHOR)
            _include(anchor, weight=1)
            entity_code = _lookup(name)
            if entity_code:
                _include(entity_code, weight=3)
                data_matched.add(entity_code)
                matched.append(f"{name}→{entity_code}")
            else:
                gaps.append(f"{coll.slug}:{name}")
        for stem, rows in tables_by_coll[coll.slug].items():
            if not rows:
                continue
            for col in rows[0].keys():
                code = _lookup(col)
                if code:
                    _include(code, weight=1)
                    data_matched.add(code)
                    matched.append(f"{stem}.{col}→{code}")
                elif not col.endswith("_id") and col != "id":
                    column_gaps.add(col)

    gaps.extend(sorted(f"column:{c}" for c in column_gaps))

    # Activated-genus expansion: a genus whose member matched *data*
    # (an entity or column reference — not an anchor included for
    # coverage) is *effective*; admit its whole family so
    # classification has discriminative choices within
    # evidence-activated families.  Data-driven, never curated; the
    # term cap below still bounds the total.
    for code in data_matched:
        lineage = vocab.lineage(code)
        genus = lineage[1] if len(lineage) > 1 else None
        if genus:
            for sibling in vocab.children.get(genus, []):
                _include(sibling, weight=0)

    codes = set(support)
    while len(codes) > profile.max_terms:
        prunable = [c for c in codes
                    if not any(ch in codes for ch in vocab.children.get(c, []))]
        prunable.sort(key=lambda c: (support.get(c, 0), -len(vocab.lineage(c))))
        if not prunable:
            break
        codes.discard(prunable[0])

    return sorted(codes), support, gaps, matched


# ── Assembly ─────────────────────────────────────────────────────

def build_sample(
    profile: SampleProfile,
    *,
    artifact_root: Path | None = None,
    corpora: Path | None = None,
) -> Path:
    """Run the full strategy; returns the sample directory."""
    from atelier.classify.artifact_set import compute_vocab_signature
    from atelier.config import load_config

    cfg = load_config()
    corpora = corpora or _CORPORA
    artifact_root = artifact_root or (_REPO_ROOT / cfg.artifact_root)

    pin = _corpus_commit(corpora)
    collections = _load_collections(corpora)
    vocab = _load_vocab(corpora)

    selected = select_collections(collections, profile)
    tables_by_coll = {c.slug: _read_tables(c) for c in selected}

    ri_reports = {c.slug: verify_collection_ri(c, tables_by_coll[c.slug])
                  for c in selected}
    codes, support, gaps, matched = select_terms(
        selected, tables_by_coll, vocab, profile)

    out_dir = artifact_root / "sdg_sample" / f"{pin[:12]}_{profile.name}"
    out_tables = out_dir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    # Flatten tables — prefix with collection topic on stem collision.
    written: dict[str, str] = {}
    table_rows: dict[str, int] = {}
    for coll in selected:
        topic = re.sub(r"-[0-9a-f]{8}$", "", coll.slug)
        for stem, rows in tables_by_coll[coll.slug].items():
            name = stem if stem not in written else f"{topic}__{stem}"
            if name in written:
                raise SdgSampleError(
                    f"Table name collision even after prefixing: {name}")
            written[name] = coll.slug
            table_rows[name] = len(rows)
            cols = list(rows[0].keys()) if rows else []
            with open(out_tables / f"{name}.csv", "w", newline="",
                      encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(rows)

    # Taxonomy subset — upstream CSV schema preserved verbatim.
    src_csv = corpora / "vocabulary" / "annotations.csv"
    with open(src_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        keep = [r for r in reader if (r.get("code") or "").strip() in set(codes)]
    with open(out_dir / "annotations.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(keep)

    code_set = set(codes)
    roots = sorted({vocab.root_of(c) for c in codes})
    genus_count = sum(
        1 for c in codes if any(ch in code_set for ch in vocab.children.get(c, [])))
    depth_hist: dict[str, int] = {}
    for c in codes:
        d = str(len(vocab.lineage(c)) - 1)
        depth_hist[d] = depth_hist.get(d, 0) + 1
    column_total = sum(
        len(rows[0].keys()) if rows else 0
        for t in tables_by_coll.values() for rows in t.values())

    manifest = {
        "source_id": SOURCE_ID,
        "corpus_commit": pin,
        "profile": asdict(profile),
        "collections": [
            {"slug": c.slug,
             "construct_id": c.manifest.get("construct_id"),
             "entities": c.manifest.get("entities", []),
             "anchors": sorted(c.anchors),
             "tables": len(tables_by_coll[c.slug]),
             "columns": c.column_count,
             "unmapped_genera": c.unmapped_genera}
            for c in selected
        ],
        "table_count": len(written),
        "tables": table_rows,
        "column_count": column_total,
        "referential_integrity": {
            "verified": True,
            "fk_edges_checked": sum(len(r["edges"]) for r in ri_reports.values()),
            "orphans": 0,
            "designative_references": sorted(
                ref for r in ri_reports.values()
                for ref in r["designative_references"]),
            "edges": [e for r in ri_reports.values() for e in r["edges"]],
        },
        "taxonomy": {
            "term_count": len(codes),
            "roots": roots,
            "root_count": len(roots),
            "genus_terms": genus_count,
            "leaf_terms": len(codes) - genus_count,
            "depth_histogram": dict(sorted(depth_hist.items())),
            "matched_references": matched,
            "vocabulary_gaps": gaps,
            "support_top": dict(
                sorted(support.items(), key=lambda kv: -kv[1])[:15]),
        },
        "vocab_sig": compute_vocab_signature(codes),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Pointer file (not a symlink — survives object-storage backends).
    pointer = artifact_root / "sdg_sample" / "current.json"
    pointer.write_text(json.dumps({
        "path": str(out_dir), "source_id": SOURCE_ID,
        "corpus_commit": pin, "profile": profile.name,
        "vocab_sig": manifest["vocab_sig"],
    }, indent=2))

    logger.info(
        "SDG sample: %d collections → %d tables / %d columns / %d terms "
        "(%d roots, %d genus) — RI verified over %d edges, %d vocab gaps "
        "→ %s",
        len(selected), len(written), column_total, len(codes), len(roots),
        genus_count, manifest["referential_integrity"]["fk_edges_checked"],
        len(gaps), out_dir,
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--profile", default="macbook", choices=sorted(PROFILES))
    parser.add_argument("--max-collections", type=int)
    parser.add_argument("--max-columns", type=int)
    parser.add_argument("--max-terms", type=int)
    parser.add_argument("--min-roots", type=int)
    parser.add_argument("--artifact-root", help="override cfg.artifact_root")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    base = PROFILES[args.profile]
    profile = SampleProfile(
        name=base.name,
        max_collections=args.max_collections or base.max_collections,
        max_columns=args.max_columns or base.max_columns,
        max_terms=args.max_terms or base.max_terms,
        min_roots=args.min_roots or base.min_roots,
    )
    try:
        out = build_sample(
            profile,
            artifact_root=Path(args.artifact_root) if args.artifact_root else None,
        )
    except SdgSampleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    summary = json.loads((out / "manifest.json").read_text())
    summary.pop("referential_integrity", None)
    summary["taxonomy"].pop("matched_references", None)
    print(json.dumps(summary | {"sample_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
