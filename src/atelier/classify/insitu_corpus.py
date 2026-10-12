"""In-situ NHSVM rows and MaxSim prototypes for an unseen relational corpus.

Projected columns are split by their assigned SKOS code. A caller that
already knows each column's published SKOS IRI passes that assignment
in. The assignment is the label. Column names, types, cell values, and
sibling names are the features. Every code the holdout classifies keeps
at least one real support column, so the head has seen that IRI on a
real row. A code with a single column stays in the support census; its
generator is what teaches the IRI to entities outside this corpus.
Holdout columns stay out of the training rows.

Each assigned IRI needs a relational generator. The generator returns
a new table whose columns and values connote that IRI, and exactly one
of those columns carries the IRI. Support columns below the floor get
more of those tables. A projected column with no assignment is not a
target. An empty footprint has no targets, so nothing is synthesized.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atelier.enrichment.llm_generator import EnrichmentGenerator, GenerationResult

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent

PROJECTED_HOLDOUT_SOURCE = "projected-holdout"
SUPPORT_FLOOR = 2
BASE_VARIANTS = 4
AMPLIFIED_VARIANTS = 12
ROWS_PER_COLUMN = 8
PROTOTYPES_PER_CODE = 6


class InSituError(RuntimeError):
    """The projected footprint or the in-situ generator could not be built."""


@dataclass
class InSituPlan:
    """Synthetic training plan for one taxonomy and one relational split."""

    support: list[tuple[str, str]]
    holdout: list[tuple[str, str]]
    counts: dict[str, int]
    variants: dict[str, int]
    exemplars: dict[str, list[str]]
    rows: list[Any] = field(default_factory=list)
    artifact_root: Path | None = None
    extended: list[str] = field(default_factory=list)

    @property
    def under_represented(self) -> list[str]:
        return sorted(
            code for code, count in self.counts.items()
            if count < SUPPORT_FLOOR
        )


def relational_split_root(artifact_root: str | Path | None = None) -> Path:
    root = Path(artifact_root) if artifact_root else Path("build")
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root / "relational_split"


def holdout_tables_dir(artifact_root: str | Path | None = None) -> Path:
    return relational_split_root(artifact_root) / "holdout" / "tables"


def prepare_insitu(
    category_set,
    artifact_root: str | Path | None = None,
    *,
    corpora: str | Path | None = None,
    seed: int = 42,
    floor: int = SUPPORT_FLOOR,
    base_variants: int = BASE_VARIANTS,
    amplified_variants: int = AMPLIFIED_VARIANTS,
    rows_per_column: int = ROWS_PER_COLUMN,
    extra_generators: dict | None = None,
    assignments: dict[tuple[str, str], str] | None = None,
) -> InSituPlan:
    """Split the projected tables and build the augmented training plan.

    ``assignments`` maps ``(table, column)`` to a published SDG code.
    Only columns that exist in the footprint, and whose code is in the
    taxonomy, are targets. ``corpora`` defaults to the local sdg-corpora
    checkout. A checkout that is absent yields an empty split and no
    synthetic rows.
    """
    cells = load_projected_cells(corpora)
    by_table = _columns_by_table(cells)
    present = {(cell["table"], cell["column"]) for cell in cells}
    known = _known_assignments(assignments, present, category_set)
    support, holdout = split_by_code(by_table, known, floor=floor)
    counts = {code: 0 for code in _target_codes(known)}
    for key in support:
        code = known.get(key)
        if code in counts:
            counts[code] += 1
    variants = {
        code: (amplified_variants if counts[code] < floor else base_variants)
        for code in counts
    }
    generators, extended = _generators(list(counts), extra_generators)
    exemplars = _exemplars(generators, seed=seed, rows_per_column=rows_per_column)
    plan = InSituPlan(
        support=support,
        holdout=holdout,
        counts=counts,
        variants=variants,
        exemplars=exemplars,
        artifact_root=relational_split_root(artifact_root),
        extended=sorted(extended),
    )
    plan.rows = _support_rows(
        cells, support, known, rows_per_column=rows_per_column,
    ) + synthetic_rows(
        variants, generators, rows_per_column=rows_per_column, seed=seed,
    )
    _write_split(plan, cells)
    return plan


def split_projected(
    table_columns: dict[str, list[str]],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratify columns by table. Even indexes are the holdout.

    A table with two or more columns contributes to both sides. Tables
    with one column alternate, in table-name order, starting with the
    holdout. The two sides are disjoint and cover every column.
    """
    support: list[tuple[str, str]] = []
    holdout: list[tuple[str, str]] = []
    singletons: list[tuple[str, str]] = []
    for table in sorted(table_columns):
        columns = sorted({c for c in table_columns[table] if c})
        if len(columns) >= 2:
            for index, column in enumerate(columns):
                bucket = holdout if index % 2 == 0 else support
                bucket.append((table, column))
        elif len(columns) == 1:
            singletons.append((table, columns[0]))
    for index, item in enumerate(singletons):
        bucket = holdout if index % 2 == 0 else support
        bucket.append(item)
    return support, holdout


def split_by_code(
    table_columns: dict[str, list[str]],
    assignments: dict[tuple[str, str], str],
    *,
    floor: int = SUPPORT_FLOOR,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratify projected columns by their assigned SKOS code.

    A code with two or more columns keeps a support column and a
    holdout column. Support grows up to ``floor`` before further
    columns are held out, so every classified column has a real
    training row of the same code. A code with one column stays in
    the support census. Columns with no assignment stay on the
    support side as sibling context and are not classification targets.
    """
    if floor < 1:
        raise InSituError("support floor must be positive")
    by_code: dict[str, list[tuple[str, str]]] = {}
    unassigned: list[tuple[str, str]] = []
    for table in sorted(table_columns):
        for column in sorted({c for c in table_columns[table] if c}):
            key = (table, column)
            code = str(assignments.get(key) or "").strip()
            if code:
                by_code.setdefault(code, []).append(key)
            else:
                unassigned.append(key)
    support: list[tuple[str, str]] = []
    holdout: list[tuple[str, str]] = []
    for code in sorted(by_code):
        columns = by_code[code]
        if len(columns) == 1:
            support.extend(columns)
            continue
        support_n = min(floor, len(columns) - 1)
        support.extend(columns[:support_n])
        holdout.extend(columns[support_n:])
    support.extend(unassigned)
    return support, holdout


def conflicting_known_entities(
    assigned: dict[tuple[str, str], str],
    predicted: dict[tuple[str, str], str],
) -> list[dict[str, str]]:
    """Known entities whose predicted code differs from the assigned code.

    The pipeline classifies these entities. A miss is a singleton the
    NHSVM would commit beside the other evidence sources.
    """
    misses: list[dict[str, str]] = []
    for table, column in sorted(assigned):
        code = assigned[(table, column)]
        got = str(predicted.get((table, column)) or "")
        if got != code:
            misses.append({
                "table": table,
                "column": column,
                "assigned": code,
                "predicted": got,
            })
    return misses


def support_counts(
    columns: list[tuple[str, str]],
    assignments: dict[tuple[str, str], str],
) -> dict[str, int]:
    """How many of ``columns`` are assigned each code.

    The count is the explicit assignment. A column name does not select
    a code. Columns outside ``columns`` are not counted, so a holdout
    assignment does not steer amplification when ``columns`` is the
    support half.
    """
    counts: dict[str, int] = {}
    for key in columns:
        code = assignments.get(key)
        if not code:
            continue
        counts[code] = counts.get(code, 0) + 1
    return counts


def synthetic_rows(
    variants: dict[str, int],
    generators: dict[str, Any],
    *,
    rows_per_column: int = ROWS_PER_COLUMN,
    seed: int = 42,
) -> list[Any]:
    """NHSVM rows from relational generators, one table per variant."""
    from atelier.classify.extension_patterns import code_iri

    rng = random.Random(seed)
    seen_tables: set[str] = set()
    rows: list[Any] = []
    for code in sorted(variants):
        width = int(variants.get(code, 0))
        if width <= 0:
            continue
        generate = generators.get(code)
        iri = code_iri(code)
        if generate is None or iri is None:
            raise InSituError(f"{code} has no relational generator")
        for index in range(width):
            table = _call_table(generate, rng, iri, code)
            name = str(table["table"])
            if name in seen_tables:
                name = f"{name}_{index + 1}"
            seen_tables.add(name)
            rows.append(_labeled_row(name, table, iri, code, rows_per_column))
    return rows


class InSituPrototypeGenerator(EnrichmentGenerator):
    """MaxSim prototypes are synthetic values for the ontology code."""

    def __init__(self, plan: InSituPlan) -> None:
        self._plan = plan

    @property
    def name(self) -> str:
        return "ontology:insitu"

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
        values = list(self._plan.exemplars.get(code) or [])
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
            notes="in-situ synthetic prototype",
        )


def load_projected_cells(corpora: str | Path | None = None) -> list[dict]:
    """Cell values from the projected footprint. No naming map is opened."""
    pin = _footprint_pin(corpora)
    if pin is None:
        return []
    path = pin / "base_rows.parquet"
    if not path.is_file():
        return []
    import pyarrow.parquet as pq

    cells = []
    for row in pq.read_table(path).to_pylist():
        table = str(row.get("table_name") or "")
        column = str(row.get("col_name") or "")
        if not table or not column:
            continue
        cells.append({
            "table": table,
            "column": column,
            "value": str(row.get("value") or ""),
            "row_ix": int(row.get("row_ix") or 0),
        })
    cells.sort(key=lambda c: (c["table"], c["column"], c["row_ix"]))
    return cells


def _write_split(plan: InSituPlan, cells: list[dict]) -> None:
    root = plan.artifact_root
    if root is None:
        return
    holdout_dir = root / "holdout" / "tables"
    if holdout_dir.exists():
        for stale in holdout_dir.glob("*.csv"):
            stale.unlink()
    holdout_dir.mkdir(parents=True, exist_ok=True)
    by_table: dict[str, list[str]] = {}
    for table, column in plan.holdout:
        columns = by_table.setdefault(table, [])
        if column not in columns:
            columns.append(column)
    values: dict[tuple[str, str], list[str]] = {}
    for cell in cells:
        if not cell["value"]:
            continue
        values.setdefault((cell["table"], cell["column"]), []).append(cell["value"])
    for table, columns in by_table.items():
        width = max((len(values.get((table, column), [])) for column in columns), default=0)
        path = holdout_dir / f"{table}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for index in range(width):
                writer.writerow([
                    _at(values.get((table, column), []), index) for column in columns
                ])
    payload = {
        "support_columns": len(plan.support),
        "holdout_columns": len(plan.holdout),
        "support_tables": sorted({table for table, _column in plan.support}),
        "holdout_tables": sorted(by_table),
        "floor": SUPPORT_FLOOR,
        "under_represented": plan.under_represented,
        "variants": plan.variants,
        "extension_codes": plan.extended,
    }
    (root / "split.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8",
    )


def _at(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def _columns_by_table(cells: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for cell in cells:
        columns = out.setdefault(cell["table"], [])
        if cell["column"] not in columns:
            columns.append(cell["column"])
    return out


def _known_assignments(
    assignments: dict[tuple[str, str], str] | None,
    present: set[tuple[str, str]],
    category_set,
) -> dict[tuple[str, str], str]:
    allowed = set(_codes(category_set))
    known: dict[tuple[str, str], str] = {}
    for key, code in (assignments or {}).items():
        code = str(code or "").strip()
        if key in present and code in allowed:
            known[key] = code
    return known


def _target_codes(assignments: dict[tuple[str, str], str]) -> list[str]:
    return sorted(set(assignments.values()))


def _generators(
    codes: list[str], extra_generators: dict | None = None,
) -> tuple[dict[str, Any], set[str]]:
    from atelier.classify.extension_patterns import code_iri

    if not codes:
        return {}, set()
    if extra_generators is None:
        extra_generators = _installed_generators()
    out = {}
    extended: set[str] = set()
    missing: list[str] = []
    for code in codes:
        iri = code_iri(code)
        generate = extra_generators.get(iri) if iri else None
        if generate is None:
            missing.append(code)
            continue
        out[code] = generate
        extended.add(code)
    if missing:
        sample = ", ".join(missing[:8])
        raise InSituError(
            f"{len(missing)} target entities have no relational generator ({sample})."
        )
    return out, extended


def _installed_generators() -> dict:
    from atelier.classify.extension_patterns import ExtensionError, extensions_path, load_generators

    path = extensions_path()
    if not path.is_file():
        return {}
    try:
        return load_generators(path.read_text(encoding="utf-8"))
    except ExtensionError as exc:
        raise InSituError(f"Installed generator extension is unusable: {exc}") from exc


def _exemplars(
    generators: dict[str, Any], *, seed: int, rows_per_column: int,
) -> dict[str, list[str]]:
    from atelier.classify.extension_patterns import code_iri

    rng = random.Random(seed + 1)
    exemplars: dict[str, list[str]] = {}
    for code, generate in generators.items():
        iri = code_iri(code)
        if iri is None:
            continue
        seen: set[str] = set()
        values: list[str] = []
        for _ in range(PROTOTYPES_PER_CODE * 3):
            table = _call_table(generate, rng, iri, code)
            for text in _iri_values(table, iri):
                if text in seen:
                    continue
                seen.add(text)
                values.append(text)
                if len(values) >= PROTOTYPES_PER_CODE:
                    break
            if len(values) >= PROTOTYPES_PER_CODE:
                break
        exemplars[code] = values[:rows_per_column]
    return exemplars


def _support_rows(
    cells: list[dict],
    support: list[tuple[str, str]],
    assignments: dict[tuple[str, str], str],
    *,
    rows_per_column: int,
) -> list[Any]:
    from atelier.classify.ml_train import _infer_column_type
    from atelier.optimize.svm.reflect import Row

    grouped: dict[tuple[str, str], list[str]] = {}
    for cell in cells:
        if not cell["value"]:
            continue
        grouped.setdefault((cell["table"], cell["column"]), []).append(cell["value"])
    by_table: dict[str, list[str]] = {}
    for table, column in support:
        columns = by_table.setdefault(table, [])
        if column not in columns:
            columns.append(column)
    rows: list[Any] = []
    for table, column in support:
        code = assignments.get((table, column))
        if not code:
            continue
        values = [value for value in grouped.get((table, column), []) if value][:rows_per_column]
        siblings = [name for name in by_table.get(table, []) if name != column]
        rows.append(Row(
            table=table,
            column=column,
            column_type=_infer_column_type(values[:10]) or "object",
            sample_values=values[:8],
            siblings_full=siblings,
            mnemonic="",
            code=code,
        ))
    return rows


def _labeled_row(table: str, generated: dict, iri: str, code: str, limit: int):
    from atelier.classify.ml_train import _infer_column_type
    from atelier.optimize.svm.reflect import Row

    labeled = next(
        column for column in generated["columns"]
        if str(column.get("iri") or "") == iri
    )
    values = [str(value) for value in labeled["values"] if str(value).strip()][:limit]
    siblings = [
        str(column["name"]) for column in generated["columns"]
        if column is not labeled
    ]
    return Row(
        table=table,
        column=str(labeled["name"]),
        column_type=_infer_column_type(values[:10]) or "object",
        sample_values=values[:8],
        siblings_full=siblings,
        mnemonic="",
        code=code,
    )


def _call_table(generate, rng, iri: str, code: str) -> dict:
    from atelier.classify.extension_patterns import relational_table_problem

    try:
        value = generate(rng)
    except Exception as exc:
        raise InSituError(
            f"{code} generator raised {type(exc).__name__}: {exc}"
        ) from exc
    problem = relational_table_problem(value, iri)
    if problem:
        raise InSituError(f"{code} {problem}")
    return value


def _iri_values(table: dict, iri: str) -> list[str]:
    values: list[str] = []
    for column in table["columns"]:
        if str(column.get("iri") or "") == iri:
            values.extend(str(value).strip() for value in column["values"] if str(value).strip())
    return values


def _footprint_pin(corpora: str | Path | None) -> Path | None:
    if corpora is not None and str(corpora).strip():
        root = Path(corpora)
    else:
        root = _REPO_ROOT / "external" / "sdg-corpora"
    if not root.is_dir():
        return None
    base = root / "ddl"
    if not base.is_dir():
        return None
    pins = sorted(
        path for path in base.iterdir()
        if path.is_dir() and (path / "base_rows.parquet").is_file()
    )
    if not pins:
        return None
    if len(pins) != 1:
        raise InSituError(
            f"sdg-corpora ddl footprint at {base} must contain exactly one "
            f"pin with base_rows.parquet (found {len(pins)})."
        )
    return pins[0]


def _categories(category_set) -> list[Any]:
    cats = getattr(category_set, "all_categories", None)
    if cats is None:
        cats = getattr(category_set, "categories", []) or []
    return list(cats)


def _codes(category_set) -> list[str]:
    return sorted({c.code for c in _categories(category_set) if c.code})
