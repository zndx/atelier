"""NHSVM rows and MaxSim prototypes from the local sdg-corpora checkout.

The column label is the SKOS code published with the DDL footprint.
When a naming-map row carries ``code`` or ``skos_code``, that value is
the label. Otherwise the code is the vocabulary row whose ``abbrev``
matches ``slot_ref`` (or ``template_id`` for a surrogate ``__pk__``).
A duplicated abbrev resolves only when exactly one hit has
``family=catalog``. Columns that do not resolve are omitted.

Deployments have the submodule at ``external/sdg-corpora``. This module
does not read a release directory and does not import another project.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atelier.enrichment.llm_generator import EnrichmentGenerator, GenerationResult

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class OntologyCorpusError(RuntimeError):
    """The local sdg-corpora checkout did not yield a usable label table."""


@dataclass(frozen=True)
class LabeledColumn:
    table: str
    column: str
    code: str
    values: list[str]
    siblings: list[str]


def corpora_root(corpora: str | Path | None = None) -> Path:
    """Local sdg-corpora checkout. An explicit path is for tests."""
    if corpora is not None and str(corpora).strip():
        path = Path(corpora)
    else:
        path = _REPO_ROOT / "external" / "sdg-corpora"
    if not path.is_dir():
        raise OntologyCorpusError(
            f"sdg-corpora checkout is not at {path}. "
            "Atelier reads that repository and does not look elsewhere."
        )
    return path


def labeled_rows_from_corpora(category_set, corpora: str | Path | None = None) -> list:
    """One NHSVM ``Row`` per footprint column whose code is in the taxonomy."""
    columns = load_labeled_columns(corpora)
    return rows_from_labeled_columns(columns, category_set)


def rows_from_labeled_columns(columns, category_set=None) -> list:
    """NHSVM rows. ``columns`` are :class:`LabeledColumn` records."""
    from atelier.classify.ml_train import _infer_column_type
    from atelier.optimize.svm.reflect import Row

    valid = _valid_codes(category_set)
    rows = []
    for col in columns:
        code = str(getattr(col, "code", "") or "")
        if not code or (valid is not None and code not in valid):
            continue
        values = [str(v).strip() for v in (col.values or []) if str(v).strip()]
        rows.append(Row(
            table=col.table,
            column=col.column,
            column_type=_infer_column_type(values[:10]) or "object",
            sample_values=values[:8],
            siblings_full=[n for n in col.siblings if n != col.column],
            mnemonic="",
            code=code,
        ))
    return rows


def exemplars_by_code(columns, category_set=None, *, limit: int = 10) -> dict[str, list[str]]:
    """Column values grouped by SKOS code, for MaxSim prototypes."""
    valid = _valid_codes(category_set)
    out: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    for col in columns:
        code = str(getattr(col, "code", "") or "")
        if not code or (valid is not None and code not in valid):
            continue
        bucket = out.setdefault(code, [])
        have = seen.setdefault(code, set())
        if len(bucket) >= limit:
            continue
        for text in col.values or []:
            text = str(text).strip()
            if not text or text in have:
                continue
            have.add(text)
            bucket.append(text)
            if len(bucket) >= limit:
                break
    return out


def ontology_generator(cfg: Any, category_set, *, corpora: str | Path | None = None):
    """MaxSim generator fed by sdg-corpora, or None when no code overlaps.

    ``cfg`` is unused. The checkout is the submodule, not a config key.
    A missing or unreadable checkout fails closed. A checkout that shares
    no codes with the taxonomy leaves the LLM enrichment path in place.
    """
    del cfg
    columns = load_labeled_columns(corpora)
    exemplars = exemplars_by_code(columns, category_set)
    if not exemplars:
        return None
    return OntologyLabelGenerator(exemplars)


def load_labeled_columns(corpora: str | Path | None = None) -> list[LabeledColumn]:
    """Footprint columns with a resolved SKOS code and their cell values."""
    root = corpora_root(corpora)
    index = _vocab_index(_read_annotations(root))
    pin = _footprint_pin(root)
    naming = _read_parquet(pin / "naming_map.parquet")
    values = _values_by_column(_read_parquet(pin / "base_rows.parquet"))
    siblings: dict[str, list[str]] = {}
    for row in naming:
        table = str(row.get("table_name") or "")
        column = str(row.get("col_name") or "")
        if table and column and column not in siblings.setdefault(table, []):
            siblings[table].append(column)
    labeled: list[LabeledColumn] = []
    for row in naming:
        code = _published_code(row, index)
        if not code:
            continue
        table = str(row.get("table_name") or "")
        column = str(row.get("col_name") or "")
        if not table or not column:
            continue
        labeled.append(LabeledColumn(
            table=table,
            column=column,
            code=code,
            values=list(values.get((table, column), [])),
            siblings=list(siblings.get(table, [])),
        ))
    return labeled


class OntologyLabelGenerator(EnrichmentGenerator):
    """Prototype text is the footprint column's values, padded with the label."""

    def __init__(self, exemplars: dict[str, list[str]]) -> None:
        self._exemplars = {str(k): list(v) for k, v in exemplars.items()}

    @property
    def name(self) -> str:
        return "ontology:sdg-corpora"

    def generate(
        self,
        source_row: dict,
        *,
        taxonomy_context: dict,
        prior_attempt: GenerationResult | None = None,
        verifier_feedback: dict | None = None,
    ) -> GenerationResult:
        code = str(source_row.get("code") or "")
        label = str(source_row.get("label") or code)
        values = list(self._exemplars.get(code) or [])
        while len(values) < 3:
            values.append(label)
        expected = taxonomy_context.get("expected_parent_path")
        parent_path = list(expected) if isinstance(expected, list) else [label]
        hints: list[str] = []
        for raw in (source_row.get("mnemonic"), label):
            hint = str(raw or "").strip().lower().replace(" ", "_")
            if hint and hint not in hints:
                hints.append(hint)
        enrichment = {
            "prototype_values": values[:10],
            "value_patterns": [{"kind": "regex", "expr": ".+"}],
            "name_hints": hints or [label.lower() or code.lower() or "ontology"],
            "anti_examples": [],
            "parent_path": parent_path,
        }
        return GenerationResult(
            enrichment=enrichment,
            attempts=1,
            notes="sdg-corpora column label",
        )


def _published_code(row: dict, index: dict[str, list[dict]]) -> str | None:
    explicit = str(row.get("code") or row.get("skos_code") or "").strip()
    if explicit:
        return explicit
    slot = str(row.get("slot_ref") or "").strip()
    if slot == "__pk__":
        return _resolve_name(str(row.get("template_id") or ""), index)
    return _resolve_name(slot, index)


def _resolve_name(token: str, index: dict[str, list[dict]]) -> str | None:
    token = (token or "").strip()
    if not token or token.startswith("__") or ":" in token:
        return None
    hits = index.get(_scream(token)) or []
    if len(hits) == 1:
        return str(hits[0]["code"])
    catalog = [h for h in hits if (h.get("family") or "") == "catalog"]
    if len(catalog) == 1:
        return str(catalog[0]["code"])
    return None


def _vocab_index(rows: list[dict]) -> dict[str, list[dict]]:
    index: dict[str, list[dict]] = {}
    for row in rows:
        abbrev = _scream(str(row.get("abbrev") or ""))
        code = str(row.get("code") or "").strip()
        if not abbrev or not code:
            continue
        index.setdefault(abbrev, []).append({
            "code": code,
            "family": str(row.get("family") or "").strip(),
        })
    return index


def _read_annotations(root: Path) -> list[dict]:
    path = root / "vocabulary" / "annotations.csv"
    if not path.is_file():
        raise OntologyCorpusError(
            f"sdg-corpora vocabulary is missing at {path}."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _footprint_pin(root: Path) -> Path:
    base = root / "ddl"
    pins = []
    if base.is_dir():
        pins = sorted(
            p for p in base.iterdir()
            if p.is_dir()
            and (p / "naming_map.parquet").is_file()
            and (p / "base_rows.parquet").is_file()
        )
    if len(pins) != 1:
        raise OntologyCorpusError(
            f"sdg-corpora ddl footprint at {base} must contain exactly one "
            f"pin with naming_map.parquet and base_rows.parquet "
            f"(found {len(pins)})."
        )
    return pins[0]


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    if not path.is_file():
        raise OntologyCorpusError(f"sdg-corpora file is missing: {path}")
    return pq.read_table(path).to_pylist()


def _values_by_column(rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    seen: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        table = str(row.get("table_name") or "")
        column = str(row.get("col_name") or "")
        text = str(row.get("value") or "").strip()
        if not table or not column or not text:
            continue
        key = (table, column)
        have = seen.setdefault(key, set())
        if text in have:
            continue
        have.add(text)
        out.setdefault(key, []).append(text)
    return out


def _valid_codes(category_set) -> set[str] | None:
    if category_set is None:
        return None
    cats = getattr(category_set, "all_categories", None)
    if cats is None:
        cats = getattr(category_set, "categories", []) or []
    return {c.code for c in cats}


def _scream(name: str) -> str:
    return _CAMEL.sub("_", name).replace("-", "_").replace(" ", "_").upper()
