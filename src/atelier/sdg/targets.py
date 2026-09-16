"""Sub-targets on an sdg-corpora *sample* for phased MaxSim/NHSVM work.

A sample is not the corpus and will not DST-converge. Operators name a
slice so successive runs can refine Qdrant and NHSVM without using
full-entity coverage as CONVERGED.

Target grammar (comma-separated):

- ``collection:<slug>``  tables in that sample collection
- ``table:<name>``       one table
- ``phase:precondition`` enrich + collection + head, then stop
- ``phase:maxsim``       semantic collection only
- ``phase:nhsvm``        NHSVM head only

Empty target on a sample: report coverage, do not fail the flow.
``source-id=sdg-corpora`` (full corpus) keeps the hard coverage gate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from atelier.sdg.sample import (
    is_corpus_source_id,
    is_sample_source_id,
    sample_dir_from_source_id,
)

PHASES = ("precondition", "maxsim", "nhsvm")


@dataclass(frozen=True)
class SampleTarget:
    source_id: str
    raw: str
    collection: str = ""
    table: str = ""
    phase: str = ""
    tables: tuple[str, ...] = ()
    entity_keys: tuple[str, ...] = ()
    hard_gate: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def stop_after_precondition(self) -> bool:
        return self.phase in PHASES

    @property
    def precondition_stages(self) -> tuple[str, ...] | None:
        if self.phase == "maxsim":
            return ("semantic_collection",)
        if self.phase == "nhsvm":
            return ("nhsvm_head",)
        if self.phase == "precondition":
            return ("semantic_collection", "nhsvm_head")
        return None


def parse_target_raw(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(
                f"target fragment {part!r} must be collection:<slug>, "
                f"table:<name>, or phase:precondition|maxsim|nhsvm"
            )
        kind, _, rest = part.partition(":")
        kind, rest = kind.strip().lower(), rest.strip()
        if kind not in ("collection", "table", "phase"):
            raise ValueError(f"unknown target kind {kind!r}")
        if kind == "phase" and rest not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {rest!r}")
        out[kind] = rest
    return out


def _manifest(sample_dir: Path) -> dict:
    path = sample_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tables_for_collection(manifest: dict, slug: str) -> list[str]:
    tables: set[str] = set()
    ri = manifest.get("referential_integrity") or {}
    for e in ri.get("edges") or []:
        if e.get("collection") == slug:
            tables.add(str(e.get("table") or ""))
            tables.add(str(e.get("references") or ""))
    prefix = slug + ":"
    for d in ri.get("designative_references") or []:
        if str(d).startswith(prefix):
            rest = str(d).split(":", 1)[1]
            tables.add(rest.split(".", 1)[0])
    slugs = {c.get("slug") for c in (manifest.get("collections") or [])}
    if slug not in slugs:
        raise ValueError(
            f"collection {slug!r} is not in this sample "
            f"(have {sorted(slugs)})"
        )
    return sorted(t for t in tables if t)


def entity_keys_for_tables(sample_dir: Path, tables: Sequence[str]) -> list[str]:
    keys: list[str] = []
    tdir = sample_dir / "tables"
    for name in tables:
        csv_path = tdir / f"{name}.csv"
        if not csv_path.is_file():
            continue
        with csv_path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), [])
        for col in header:
            if col:
                keys.append(f"{name}.{col}")
    return keys


def resolve_sample_target(source_id: str, raw: str = "") -> SampleTarget:
    """Resolve a sub-target. Full corpus keeps a hard coverage gate."""
    notes: list[str] = []
    parsed = parse_target_raw(raw) if raw else {}
    if is_corpus_source_id(source_id):
        return SampleTarget(
            source_id=source_id, raw=raw or "",
            phase=parsed.get("phase", ""),
            hard_gate=True,
            notes=("full corpus: coverage is CONVERGED",),
        )
    if not is_sample_source_id(source_id):
        return SampleTarget(source_id=source_id, raw=raw or "", hard_gate=True)

    sample_dir = sample_dir_from_source_id(source_id)
    if sample_dir is None:
        raise RuntimeError(f"sample {source_id!r} is not on disk")
    manifest = _manifest(sample_dir)
    collection = parsed.get("collection", "")
    table = parsed.get("table", "")
    phase = parsed.get("phase", "")
    tables: list[str] = []
    if collection:
        tables = tables_for_collection(manifest, collection)
    if table:
        tables = [table] if not tables else [t for t in tables if t == table]
        if table not in (manifest.get("tables") or {}) and table not in tables:
            # still allow a csv that exists
            if not (sample_dir / "tables" / f"{table}.csv").is_file():
                raise ValueError(f"table {table!r} not in sample {source_id}")
            tables = [table]
    keys = entity_keys_for_tables(sample_dir, tables) if tables else ()
    hard = bool(tables)  # a named slice is a real gate; bare sample is not
    if not tables:
        notes.append(
            "sample without collection/table sub-target: coverage is "
            "reported, not CONVERGED (77-gap macbook sample cannot DST-converge)"
        )
    return SampleTarget(
        source_id=source_id,
        raw=raw or "",
        collection=collection,
        table=table,
        phase=phase,
        tables=tuple(tables),
        entity_keys=tuple(keys),
        hard_gate=hard,
        notes=tuple(notes),
    )
