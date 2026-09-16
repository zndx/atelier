"""Target/reference sample pairs for holdout NHSVM.

Train a fresh NHSVM on a *reference* sample whose BFO/CCO genus IRIs
cover the *target* sample, with **no collection overlap**. The target
is the holdout Atelier classifies; the reference is the training
substrate (synth from its SKOS + enrichment).

Ægir should emit the reference from finepdfs + SchemaPile so IRI
coverage is sufficient for NHSVM to label every SKOS-grounded column
on the target. Until that emission exists, Atelier can *select*
disjoint collections from the same sdg-corpora pin as a partial
reference (genus layer). Remaining SKOS-term gaps are reported as
``skos_missing`` — those are Ægir's to close.

    source-id = target sample (holdout)
    reference-id = reference sample (train, disjoint)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from atelier.sdg.sample import (
    Collection,
    SampleProfile,
    SdgSampleError,
    is_sample_source_id,
    sample_dir_from_source_id,
)

GURU_OVERLAP = "#AT.00000020.PAIROVERLAP"
GURU_IRI = "#AT.00000021.PAIRIRI"
GURU_SKOS = "#AT.00000022.PAIRSKOS"
GURU_SHAPE = "#AT.00000023.PAIRSHAPE"


class PairError(SdgSampleError):
    """Pair is unsound: overlap, missing IRI cover, or bad ids."""


def genus_iris(coll: Collection) -> set[str]:
    out: set[str] = set(getattr(coll, "genera", None) or ())
    for term in coll.manifest.get("terms") or []:
        if isinstance(term, dict) and term.get("genus"):
            out.add(str(term["genus"]))
    out.update(g for g in (coll.unmapped_genera or []) if g)
    return out


def _slugs(sample_dir: Path) -> set[str]:
    manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    return {c["slug"] for c in (manifest.get("collections") or []) if c.get("slug")}


def _skos_codes(sample_dir: Path) -> set[str]:
    path = sample_dir / "annotations.csv"
    if not path.is_file():
        return set()
    codes: set[str] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            c = (row.get("code") or "").strip()
            if c:
                codes.add(c)
    return codes


def _genera_from_manifest(sample_dir: Path) -> set[str]:
    manifest = json.loads((sample_dir / "manifest.json").read_text(encoding="utf-8"))
    out: set[str] = set()
    for c in manifest.get("collections") or []:
        out.update(c.get("unmapped_genera") or [])
        # anchors are SKOS, not IRIs; genera live on corpus terms. Best-effort:
        for g in c.get("genera") or []:
            out.add(str(g))
    return out


@dataclass
class PairReport:
    target_id: str
    reference_id: str
    target_slugs: tuple[str, ...]
    reference_slugs: tuple[str, ...]
    required_iris: tuple[str, ...]
    covered_iris: tuple[str, ...]
    missing_iris: tuple[str, ...]
    skos_missing: tuple[str, ...]
    overlap: tuple[str, ...]
    perfect_possible: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def select_reference_collections(
    collections: list[Collection],
    *,
    exclude: set[str],
    required_iris: set[str],
    profile: SampleProfile,
) -> list[Collection]:
    """Greedy disjoint cover of target genus IRIs. Fail-closed if incomplete."""
    remaining = [
        c for c in collections
        if c.slug not in exclude and c.column_count > 0
    ]
    selected: list[Collection] = []
    covered: set[str] = set()
    columns_used = 0
    required = set(required_iris)

    def gain(c: Collection) -> tuple:
        iris = genus_iris(c)
        new = len(iris & (required - covered))
        return (new, len(iris), -c.column_count)

    while not required <= covered and remaining and len(selected) < profile.max_collections:
        remaining.sort(key=gain, reverse=True)
        c = remaining[0]
        if gain(c)[0] == 0:
            break
        if columns_used + c.column_count > profile.max_columns:
            remaining.pop(0)
            continue
        selected.append(c)
        covered |= genus_iris(c)
        columns_used += c.column_count
        remaining.remove(c)

    missing = required - covered
    if missing:
        raise PairError(
            f"{GURU_IRI} disjoint reference cannot cover target genus IRIs "
            f"{sorted(missing)}. Ægir must emit collections from "
            f"finepdfs+schemapile that realize those IRIs without overlapping "
            f"the target sample."
        )
    return selected


def assert_pair(target_id: str, reference_id: str) -> PairReport:
    """Fail-closed: disjoint collections; report IRI and SKOS cover."""
    if not is_sample_source_id(target_id) or not is_sample_source_id(reference_id):
        raise PairError(
            f"{GURU_SHAPE} pair requires two sample ids "
            f"(sdg-corpora/<pin>_<profile>), got {target_id!r} / {reference_id!r}"
        )
    if target_id == reference_id:
        raise PairError(f"{GURU_OVERLAP} target and reference are the same sample")
    tdir = sample_dir_from_source_id(target_id)
    rdir = sample_dir_from_source_id(reference_id)
    if tdir is None or rdir is None:
        raise PairError(
            f"{GURU_SHAPE} sample dir missing for {target_id!r} or {reference_id!r}"
        )
    tslugs, rslugs = _slugs(tdir), _slugs(rdir)
    overlap = sorted(tslugs & rslugs)
    if overlap:
        raise PairError(
            f"{GURU_OVERLAP} target and reference share collections {overlap}"
        )
    # Genera: prefer corpus-backed lists stored on the sample when present.
    t_iris = _genera_from_manifest(tdir)
    r_iris = _genera_from_manifest(rdir)
    missing_iris = sorted(t_iris - r_iris) if t_iris else []
    t_skos, r_skos = _skos_codes(tdir), _skos_codes(rdir)
    skos_missing = tuple(sorted(t_skos - r_skos))
    notes = []
    if missing_iris:
        notes.append(f"{GURU_IRI} genus IRIs not in reference: {missing_iris}")
    if skos_missing:
        notes.append(
            f"{GURU_SKOS} {len(skos_missing)} target SKOS codes absent from "
            f"reference — NHSVM cannot perfectly label those columns until "
            f"Ægir emits covering collections (finepdfs+schemapile)"
        )
    if missing_iris:
        raise PairError(notes[0])
    return PairReport(
        target_id=target_id,
        reference_id=reference_id,
        target_slugs=tuple(sorted(tslugs)),
        reference_slugs=tuple(sorted(rslugs)),
        required_iris=tuple(sorted(t_iris)),
        covered_iris=tuple(sorted(r_iris & t_iris if t_iris else r_iris)),
        missing_iris=tuple(missing_iris),
        skos_missing=skos_missing,
        overlap=(),
        perfect_possible=not skos_missing,
        notes=tuple(notes),
    )
