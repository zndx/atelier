"""Train, test, and validate splits for published collection tables.

The answer key is ``corpus/tagging/answer_key.csv``. Every column of a
published relational table is one row of that key, and every key row is
one published column. ``class_iri`` is the column's OWL class. The SKOS
label is ``reference_code`` when the key publishes one, gathered to its
genus parent in ``vocabulary/annotations.csv``. A leaf becomes that
genus. A code that is already a genus stays itself. A child of
``SDG.GENERIC`` stays itself: that parent is an anchorless bucket, not
a structural kind.

A column whose ``reference_code`` is empty keeps its IRI and is not a
labeled target. No other term is written in its place. A code that is
not in the vocabulary is an error.

A collection stays on one side. Tables in that collection stay with it.
The labeled genera are covered: each genus is in train, a genus that
spans two collections is also in test, and a genus that spans three is
also in validate.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from atelier.classify.decisions_walk import genus_term
from atelier.classify.insitu_corpus import InSituError, _corpora_root, _sql_column_types
from atelier.classify.taxonomy import load_annotations_from_filesystem

_ANSWER_KEY = Path("corpus") / "tagging" / "answer_key.csv"
_VOCAB = Path("vocabulary") / "annotations.csv"
SPLITS = ("train", "test", "validate")
_SPLIT_QUOTA = {"train": 0.8, "test": 0.1, "validate": 0.1}
_SPLIT_PRIORITY = {"train": 0, "test": 1, "validate": 2}
_KEY_FIELDS = (
    "collection",
    "table",
    "column",
    "ordinal",
    "class_iri",
    "property_iri",
    "range_iri",
    "genus",
    "reference_code",
    "ref_basis",
)


@dataclass(frozen=True)
class ColumnAssociation:
    """One published column, its OWL record, and its genus label."""

    collection: str
    table: str
    column: str
    ordinal: int
    class_iri: str
    property_iri: str
    range_iri: str
    genus_iri: str
    reference_code: str
    ref_basis: str
    genus_code: str
    split: str
    column_type: str
    table_columns: tuple[str, ...]

    @property
    def labeled(self) -> bool:
        return bool(self.genus_code)


@dataclass(frozen=True)
class GenusSplit:
    """Collection-sided split. Labeled columns carry a vocabulary genus."""

    columns: tuple[ColumnAssociation, ...]

    def collections(self) -> dict[str, str]:
        sides: dict[str, str] = {}
        for column in self.columns:
            previous = sides.get(column.collection)
            if previous is not None and previous != column.split:
                raise InSituError(
                    f"collection {column.collection} is on more than one side"
                )
            sides[column.collection] = column.split
        return sides

    def labeled_columns(self) -> tuple[ColumnAssociation, ...]:
        return tuple(column for column in self.columns if column.labeled)

    def genus_codes(self) -> tuple[str, ...]:
        return tuple(sorted({column.genus_code for column in self.columns if column.labeled}))


def align_genus_split(
    corpora: str | Path | None = None,
    *,
    category_set=None,
) -> GenusSplit:
    """Join published tables to the answer key and assign the three sides.

    ``category_set`` defaults to ``vocabulary/annotations.csv`` in the
    same checkout. The genus of a published code is ``genus_term``.
    """
    root = _corpora_root(corpora)
    if root is None:
        raise InSituError("sdg-corpora checkout is not on disk")
    key_path = root / _ANSWER_KEY
    if not key_path.is_file():
        raise InSituError(
            f"collection answer key is not at {key_path}. "
            "The published file is corpus/tagging/answer_key.csv."
        )
    cats = category_set
    if cats is None:
        vocab_path = root / _VOCAB
        if not vocab_path.is_file():
            raise InSituError(f"vocabulary is not at {vocab_path}")
        cats = load_annotations_from_filesystem(vocab_path, hierarchical=True)
    published = _published_columns(root)
    keyed = _read_answer_key(key_path)
    _require_same_columns(published, keyed)
    records = [
        _associate(keyed[key], published[key], cats) for key in sorted(keyed)
    ]
    records.sort(key=lambda row: (row.collection, row.table, row.ordinal, row.column))
    sides = _assign_collections(records)
    columns = tuple(
        ColumnAssociation(
            collection=row.collection,
            table=row.table,
            column=row.column,
            ordinal=row.ordinal,
            class_iri=row.class_iri,
            property_iri=row.property_iri,
            range_iri=row.range_iri,
            genus_iri=row.genus_iri,
            reference_code=row.reference_code,
            ref_basis=row.ref_basis,
            genus_code=row.genus_code,
            split=sides[row.collection],
            column_type=row.column_type,
            table_columns=row.table_columns,
        )
        for row in records
    )
    split = GenusSplit(columns)
    gaps = _coverage_gaps(split)
    if gaps:
        raise InSituError(
            "genus split left a learnable genus off a required side: "
            + "; ".join(gaps[:8])
        )
    return split


def _published_columns(root: Path) -> dict[tuple[str, str, str], tuple[str, tuple[str, ...]]]:
    """CSV header columns. The value is ``(sql type, table columns)``."""
    base = root / "corpus" / "collections"
    found: dict[tuple[str, str, str], tuple[str, tuple[str, ...]]] = {}
    if not base.is_dir():
        return found
    for slug_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        tables = slug_dir / "tables"
        if not tables.is_dir():
            continue
        for csv_path in sorted(tables.glob("*.csv")):
            with csv_path.open(newline="", encoding="utf-8") as handle:
                header = next(csv.reader(handle), [])
            columns = tuple(name for name in header if name)
            types = _sql_column_types(tables / f"{csv_path.stem}.sql")
            for name in columns:
                found[(slug_dir.name, csv_path.stem, name)] = (
                    types.get(name, ""),
                    columns,
                )
    return found


def _read_answer_key(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in _KEY_FIELDS if name not in set(reader.fieldnames or [])]
        if missing:
            raise InSituError(
                "collection answer key is missing columns: " + ", ".join(missing)
            )
        for row in reader:
            slug = str(row.get("collection") or "").strip()
            table = str(row.get("table") or "").strip()
            column = str(row.get("column") or "").strip()
            if not slug or not table or not column:
                raise InSituError("collection answer key has a blank column identity")
            identity = (slug, table, column)
            if identity in keyed:
                raise InSituError(
                    f"collection answer key repeats {slug}/{table}.{column}"
                )
            class_iri = str(row.get("class_iri") or "").strip()
            if not class_iri:
                raise InSituError(
                    f"{slug}/{table}.{column} has no class_iri in the answer key"
                )
            keyed[identity] = {name: str(row.get(name) or "").strip() for name in _KEY_FIELDS}
    return keyed


def _require_same_columns(
    published: dict[tuple[str, str, str], tuple[str, tuple[str, ...]]],
    keyed: dict[tuple[str, str, str], dict[str, str]],
) -> None:
    if set(published) == set(keyed):
        return
    missing = sorted(set(published) - set(keyed))
    extra = sorted(set(keyed) - set(published))
    detail = []
    if missing:
        slug, table, column = missing[0]
        detail.append(f"{slug}/{table}.{column} is not in the answer key")
    if extra:
        slug, table, column = extra[0]
        detail.append(f"{slug}/{table}.{column} is not a published table column")
    raise InSituError(
        "published tables and the answer key differ "
        f"({len(missing)} unpublished, {len(extra)} missing tables): "
        + "; ".join(detail)
    )


def _associate(
    row: dict[str, str],
    published: tuple[str, tuple[str, ...]],
    category_set,
) -> ColumnAssociation:
    code = row["reference_code"]
    genus_code = ""
    if code:
        known = getattr(category_set, "all_by_code", None) or getattr(category_set, "parent", {})
        if code not in known:
            raise InSituError(
                f"{row['collection']}/{row['table']}.{row['column']} "
                f"reference_code {code} is not in the sdg-corpora vocabulary"
            )
        genus_code = genus_term(category_set, code) or ""
        if not genus_code or genus_code not in known:
            raise InSituError(
                f"{row['collection']}/{row['table']}.{row['column']} "
                f"code {code} has no genus parent in the vocabulary"
            )
    column_type, table_columns = published
    try:
        ordinal = int(row["ordinal"])
    except ValueError as exc:
        raise InSituError(
            f"{row['collection']}/{row['table']}.{row['column']} "
            f"has ordinal {row['ordinal']!r}"
        ) from exc
    return ColumnAssociation(
        collection=row["collection"],
        table=row["table"],
        column=row["column"],
        ordinal=ordinal,
        class_iri=row["class_iri"],
        property_iri=row["property_iri"],
        range_iri=row["range_iri"],
        genus_iri=row["genus"],
        reference_code=code,
        ref_basis=row["ref_basis"],
        genus_code=genus_code,
        split="",
        column_type=column_type,
        table_columns=table_columns,
    )


def _assign_collections(records: list[ColumnAssociation]) -> dict[str, str]:
    """Put each collection on one side and keep genera learnable."""
    genera_of: dict[str, set[str]] = defaultdict(set)
    for row in records:
        genera_of.setdefault(row.collection, set())
        if row.genus_code:
            genera_of[row.collection].add(row.genus_code)
    slugs = sorted(genera_of)
    members: dict[str, list[str]] = defaultdict(list)
    for slug in slugs:
        for genus in sorted(genera_of[slug]):
            members[genus].append(slug)
    assigned: dict[str, str] = {}
    order = sorted(members, key=lambda genus: (len(members[genus]), genus))
    for genus in order:
        have = Counter(assigned[slug] for slug in members[genus] if slug in assigned)
        wants: list[str] = []
        if have["train"] == 0:
            wants.append("train")
        if len(members[genus]) >= 2 and have["test"] == 0:
            wants.append("test")
        if len(members[genus]) >= 3 and have["validate"] == 0:
            wants.append("validate")
        for split in wants:
            choices = [slug for slug in members[genus] if slug not in assigned]
            if not choices:
                break
            choices.sort(key=lambda slug, genus=genus: _reservation_cost(
                slug, genus, genera_of, members,
            ))
            assigned[choices[0]] = split
    total = len(slugs)
    for slug in slugs:
        if slug in assigned:
            continue
        counts = Counter(assigned.values())

        def deficit(split: str, counts: Counter = counts) -> float:
            return _SPLIT_QUOTA[split] * total - counts[split]

        assigned[slug] = max(
            SPLITS, key=lambda split: (deficit(split), -_SPLIT_PRIORITY[split]),
        )
    return assigned


def _reservation_cost(
    slug: str,
    genus: str,
    genera_of: dict[str, set[str]],
    members: dict[str, list[str]],
) -> tuple[int, str]:
    others = genera_of[slug] - {genus}
    if not others:
        return (0, slug)
    return (min(len(members[other]) for other in others), slug)


def _coverage_gaps(split: GenusSplit) -> list[str]:
    by_genus: dict[str, list[ColumnAssociation]] = defaultdict(list)
    collections_of: dict[str, set[str]] = defaultdict(set)
    for column in split.labeled_columns():
        by_genus[column.genus_code].append(column)
        collections_of[column.genus_code].add(column.collection)
    gaps: list[str] = []
    for genus, columns in sorted(by_genus.items()):
        sides = Counter(column.split for column in columns)
        collection_count = len(collections_of[genus])
        missing = []
        if sides["train"] == 0:
            missing.append("train")
        if collection_count >= 2 and sides["test"] == 0:
            missing.append("test")
        if collection_count >= 3 and sides["validate"] == 0:
            missing.append("validate")
        if missing:
            gaps.append(f"{genus} missing {', '.join(missing)}")
    return gaps
