"""Published SDG terms the vocabulary tests name explicitly.

The rows come from the sdg-corpora checkout. A code that is not in that
vocabulary is not a term this suite will stand in for.
"""

from __future__ import annotations

import csv
from pathlib import Path

from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory

_ROOT = Path(__file__).resolve().parents[2]
ANNOTATIONS = _ROOT / "external" / "sdg-corpora" / "vocabulary" / "annotations.csv"

PROCESS = "SDG.PROCESS"
ROLE = "SDG.ROLE"
PDF_EXPORT = "SDG.PROCESS.PDF_EXPORT_PROCESS"
PARENTING = "SDG.ROLE.PARENTING_ROLE"
YOUNG_RESEARCHER = "SDG.ROLE.YOUNG_RESEARCHER_ROLE"
CLINICAL = "SDG.PROCESS.CLINICAL_DOCUMENTATION_ACTIVITY"
ORGANIZATION = "SDG.MATERIAL_ENTITY.ORGANIZATION"
ORGANIZATION_DOMAIN = "SDG.DOM.ORGANIZATION"

EXPLICIT = (
    PROCESS,
    ROLE,
    PDF_EXPORT,
    PARENTING,
    YOUNG_RESEARCHER,
    CLINICAL,
    ORGANIZATION,
    ORGANIZATION_DOMAIN,
)


def published_rows(codes: tuple[str, ...] | list[str] = EXPLICIT) -> list[dict[str, str]]:
    """Rows from the published annotations, in ``codes`` order."""
    if not ANNOTATIONS.is_file():
        raise AssertionError(f"published SDG vocabulary is missing at {ANNOTATIONS}")
    with ANNOTATIONS.open(newline="", encoding="utf-8") as handle:
        by_code = {row["code"]: row for row in csv.DictReader(handle)}
    missing = [code for code in codes if code not in by_code]
    if missing:
        raise AssertionError(
            "published SDG vocabulary has no row for " + ", ".join(missing)
        )
    return [by_code[code] for code in codes]


def taxonomy(codes: tuple[str, ...] | list[str]) -> HierarchicalCategorySet:
    categories = [
        ReferenceCategory(
            code=row["code"],
            label=row["label"],
            embedding_text=row["label"],
            abbrev=row["abbrev"],
            parent_code=row["parent_code"] or None,
            description=row["description"],
        )
        for row in published_rows(codes)
    ]
    return HierarchicalCategorySet(name="sdg", categories=categories)


def write_annotations(
    path: Path,
    codes: tuple[str, ...] | list[str],
    *,
    sentinel: bool = False,
) -> None:
    """Write an explicit slice. Sentinels sit in columns the pack must drop."""
    fields = [
        "code", "label", "abbrev", "parent_code", "description",
        "example_values", "common_names", "family",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in published_rows(codes):
            item = {field: row.get(field, "") for field in fields}
            if sentinel:
                item["example_values"] = "INV-SECRET"
                item["common_names"] = "naming_map"
            writer.writerow(item)


def relational_procedure(iri: str, column: str, token: str):
    """One synthetic table whose labeled column carries ``iri``."""

    def generate(rng, _iri: str = iri, _column: str = column, _token: str = token):
        return {
            "table": f"syn_{rng.randint(1000, 9999)}",
            "columns": [
                {
                    "name": _column,
                    "values": [f"{_token}-{rng.randint(1000, 9999)}" for _ in range(3)],
                    "iri": _iri,
                },
                {
                    "name": "recorded_on",
                    "values": [f"2020-0{rng.randint(1, 9)}-15"],
                },
            ],
        }

    return generate
