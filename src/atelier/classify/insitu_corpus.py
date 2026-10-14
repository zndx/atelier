"""In-situ NHSVM rows and MaxSim prototypes for an unseen relational corpus.

Column type annotation holds out whole collections of related tables.
A collection is one referentially closed bundle: every table in it is
support, or every table in it is holdout. The label is the published
SKOS code on each target column. The features are the classification
pipeline's discrete features, including sibling columns and sampled
values, so SHAP and SAGE can ablate them. Training rows are the real
support columns. Stub synthetic tables are not added.
A column split that keeps one column of a table in training and its
siblings in the holdout is not this exam.

A caller that already knows each column's published SKOS code passes
that assignment in. Collection CTA then labels the column with its
term in the neighborhood of the structural genera. The code is not
a feature.
Collection directories do not carry that assignment; the answer key is
a separate CSV of exact codes. Column names are not labels. Holding a
collection out skews the tag distribution. That deficit is recorded
on the plan and is not filled here. An empty footprint has no targets.
"""

from __future__ import annotations

import csv
import json
import re
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
_ANSWER_KEY = Path("corpus") / "cta_answer_key.csv"
_SQL_COLUMN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s+([A-Za-z]+)(?:\s*\([^)]*\))?",
)
_SQL_SKIP = frozenset({
    "create", "primary", "foreign", "constraint", "unique",
    "check", "key", "references", "alter",
})


class InSituError(RuntimeError):
    """The projected footprint or the in-situ generator could not be built."""


@dataclass
class InSituPlan:
    """Training plan for one taxonomy and one relational split.

    ``rows`` are the real support columns. ``variants`` records the
    distribution deficit and is not filled with stub tables.
    """

    support: list[tuple[str, str]]
    holdout: list[tuple[str, str]]
    counts: dict[str, int]
    variants: dict[str, int]
    exemplars: dict[str, list[str]]
    rows: list[Any] = field(default_factory=list)
    artifact_root: Path | None = None
    extended: list[str] = field(default_factory=list)
    holdout_collections: list[str] = field(default_factory=list)
    holdout_counts: dict[str, int] = field(default_factory=dict)
    holdout_variants: dict[str, int] = field(default_factory=dict)
    holdout_rows: list[Any] = field(default_factory=list)

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
    collections: dict[str, str] | None = None,
    holdout_collections: set[str] | None = None,
    cells: list[dict] | None = None,
) -> InSituPlan:
    """Split the projected tables and build the augmented training plan.

    ``assignments`` maps ``(table, column)`` to a published SDG code.
    ``collections`` maps each table to its collection slug. When it is
    set, ``holdout_collections`` names the bundles held out whole.
    Only columns that exist in the footprint, and whose code is in the
    taxonomy, are targets. ``corpora`` defaults to the local sdg-corpora
    checkout. A checkout that is absent yields an empty split.
    ``cells`` replaces the projected footprint when the caller already
    loaded collection tables. ``extra_generators`` is accepted and
    ignored: stub tables are not emitted.
    """
    if cells is None:
        cells = load_projected_cells(corpora)
    by_table = _columns_by_table(cells)
    present = {(cell["table"], cell["column"]) for cell in cells}
    known = _known_assignments(assignments, present, category_set)
    held: list[str] = []
    if collections is not None:
        support, holdout = split_by_collection(
            by_table, collections, holdout_collections or set(),
        )
        held = sorted(set(holdout_collections or set()))
    else:
        support, holdout = split_by_code(by_table, known, floor=floor)
    counts = {code: 0 for code in _target_codes(known)}
    for key in support:
        code = known.get(key)
        if code in counts:
            counts[code] += 1
    holdout_counts: dict[str, int] = {}
    holdout_variants: dict[str, int] = {}
    del extra_generators, seed
    if collections is not None:
        # Real collection membership fixes the tag histogram. The
        # deficit is the count of columns that would bring each side
        # up to the pooled share. It is not filled with stub tables.
        holdout_counts = support_counts(holdout, known)
        variants = distribution_deficit(counts, holdout_counts)
        holdout_variants = distribution_deficit(holdout_counts, counts)
    else:
        variants = {
            code: (amplified_variants if counts[code] < floor else base_variants)
            for code in counts
        }
    plan = InSituPlan(
        support=support,
        holdout=holdout,
        counts=counts,
        variants=variants,
        exemplars={},
        artifact_root=relational_split_root(artifact_root),
        extended=[],
        holdout_collections=held,
        holdout_counts=holdout_counts,
        holdout_variants=holdout_variants,
    )
    plan.rows = _support_rows(
        cells, support, known, rows_per_column=rows_per_column,
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


def split_by_collection(
    table_columns: dict[str, list[str]],
    collections: dict[str, str],
    holdout_collections: set[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Hold out every table that belongs to a named collection.

    Tables in one collection stay together, so a target column's
    siblings and the related tables those siblings point at are on the
    same side of the exam. Column order inside a table is preserved.
    """
    if not holdout_collections:
        raise InSituError("hold out at least one collection")
    missing_tables = sorted(table for table in table_columns if table not in collections)
    if missing_tables:
        raise InSituError(
            f"tables have no collection: {', '.join(missing_tables[:8])}"
        )
    known = set(collections.values())
    unknown = sorted(set(holdout_collections) - known)
    if unknown:
        raise InSituError(
            f"unknown holdout collections: {', '.join(unknown[:8])}"
        )
    support: list[tuple[str, str]] = []
    holdout: list[tuple[str, str]] = []
    for table in sorted(table_columns):
        bucket = holdout if collections[table] in holdout_collections else support
        seen: set[str] = set()
        for column in table_columns[table]:
            if not column or column in seen:
                continue
            seen.add(column)
            bucket.append((table, column))
    if not support:
        raise InSituError("held-out collections left no training tables")
    return support, holdout


def prepare_collection_cta(
    category_set,
    artifact_root: str | Path | None = None,
    *,
    corpora: str | Path | None = None,
    assignments: dict[tuple[str, str], str] | None = None,
    holdout_collections: set[str] | None = None,
    slugs: set[str] | None = None,
    seed: int = 42,
    floor: int = SUPPORT_FLOOR,
    base_variants: int = BASE_VARIANTS,
    amplified_variants: int = AMPLIFIED_VARIANTS,
    rows_per_column: int = ROWS_PER_COLUMN,
    extra_generators: dict | None = None,
    order: int = 1,
) -> InSituPlan:
    """Hold out whole collection directories and build the CTA plan.

    Table ids are ``slug/table``. ``assignments`` is the exact
    ``(table id, column) → SKOS code`` map. Each target column is
    labeled with its term in the genus neighborhood. Order 1 keeps a
    genus, its parent, and its children as distinct labels. Order 2
    also keeps the siblings of those genera. A taxonomy with no
    structural genera keeps the genus-level term. Columns missing
    from the map stay in the bundle as sibling context and are not
    targets.
    """
    from atelier.classify.decisions_walk import associate_genera

    cells, collections = load_collection_cells(corpora, slugs=slugs)
    if not cells:
        raise InSituError("no collection tables to hold out")
    assignments = associate_genera(assignments or {}, category_set, order=order)
    return prepare_insitu(
        category_set,
        artifact_root,
        corpora=corpora,
        seed=seed,
        floor=floor,
        base_variants=base_variants,
        amplified_variants=amplified_variants,
        rows_per_column=rows_per_column,
        extra_generators=extra_generators,
        assignments=assignments,
        collections=collections,
        holdout_collections=holdout_collections,
        cells=cells,
    )


def load_collection_cells(
    corpora: str | Path | None = None,
    *,
    slugs: set[str] | None = None,
) -> tuple[list[dict], dict[str, str]]:
    """Read ``corpus/collections/<slug>/tables`` into cell records.

    The second map sends each ``slug/table`` id to its collection slug.
    SQL files supply the column type when present. An absent collections
    directory yields an empty pair.
    """
    root = _corpora_root(corpora)
    base = root / "corpus" / "collections" if root is not None else None
    if base is None or not base.is_dir():
        return [], {}
    cells: list[dict] = []
    collections: dict[str, str] = {}
    for slug_dir in sorted(path for path in base.iterdir() if path.is_dir()):
        slug = slug_dir.name
        if slugs is not None and slug not in slugs:
            continue
        manifest_path = slug_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tables_dir = slug_dir / "tables"
        names = [
            str(table.get("name") or "")
            for table in (manifest.get("tables") or [])
            if isinstance(table, dict) and table.get("name")
        ]
        if not names and tables_dir.is_dir():
            names = sorted(path.stem for path in tables_dir.glob("*.csv"))
        for name in names:
            csv_path = tables_dir / f"{name}.csv"
            if not csv_path.is_file():
                continue
            table_id = f"{slug}/{name}"
            collections[table_id] = slug
            column_types = _sql_column_types(tables_dir / f"{name}.sql")
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                columns = list(reader.fieldnames or [])
                for row_ix, row in enumerate(reader):
                    for column in columns:
                        if not column:
                            continue
                        cells.append({
                            "table": table_id,
                            "column": column,
                            "value": str((row or {}).get(column) or ""),
                            "row_ix": row_ix,
                            "column_type": column_types.get(column, ""),
                        })
    return cells, collections


def load_collection_answer_key(
    corpora: str | Path | None = None,
) -> tuple[dict[tuple[str, str], str], set[str]]:
    """Exact collection column codes and the collections held out whole.

    The file is ``corpus/cta_answer_key.csv`` with columns
    ``collection,table,column,code,split``. ``split`` is ``support`` or
    ``holdout``. One collection is entirely one side. A missing file
    yields an empty assignment and an empty holdout. Column names in
    the collection CSVs are not consulted.
    """
    root = _corpora_root(corpora)
    path = root / _ANSWER_KEY if root is not None else None
    if path is None or not path.is_file():
        return {}, set()
    assignments: dict[tuple[str, str], str] = {}
    sides: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"collection", "table", "column", "code", "split"}
        missing = sorted(required - fields)
        if missing:
            raise InSituError(
                f"collection answer key is missing columns: {', '.join(missing)}"
            )
        for row in reader:
            slug = str(row.get("collection") or "").strip()
            table = str(row.get("table") or "").strip()
            column = str(row.get("column") or "").strip()
            code = str(row.get("code") or "").strip()
            side = str(row.get("split") or "").strip()
            if not slug or not table or not column or not code:
                continue
            if side not in {"support", "holdout"}:
                raise InSituError(
                    f"collection answer key split for {slug}/{table}.{column} "
                    f"must be support or holdout"
                )
            previous = sides.get(slug)
            if previous is not None and previous != side:
                raise InSituError(
                    f"collection {slug} is on both sides of the answer key"
                )
            sides[slug] = side
            assignments[(f"{slug}/{table}", column)] = code
    holdout = {slug for slug, side in sides.items() if side == "holdout"}
    return assignments, holdout


def _corpora_root(corpora: str | Path | None) -> Path | None:
    if corpora is not None and str(corpora).strip():
        root = Path(corpora)
    else:
        root = _REPO_ROOT / "external" / "sdg-corpora"
    return root if root.is_dir() else None


def _sql_column_types(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    types: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip().rstrip(",")
        if not stripped or stripped.startswith((")", "--")):
            continue
        match = _SQL_COLUMN.match(stripped)
        if match is None:
            continue
        name, raw_type = match.group(1), match.group(2)
        if name.lower() in _SQL_SKIP:
            continue
        types[name] = raw_type
    return types


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


def distribution_deficit(
    side: dict[str, int],
    other: dict[str, int],
) -> dict[str, int]:
    """Columns that would bring ``side`` up to the pooled tag share.

    The count is recorded on the plan. It is not filled with stub tables.

    ``side`` and ``other`` map a SKOS code to its real column count on
    that split. The pooled share of a code is its count on both sides
    divided by the number of assigned columns. The target on ``side``
    is that share times the side's real size, rounded up. A side with
    no real columns takes the other side's counts, so the empty split
    still carries every tag. Codes already at the target are omitted.
    """
    side_n = sum(side.values())
    other_n = sum(other.values())
    pooled = side_n + other_n
    if pooled == 0:
        return {}
    deficit: dict[str, int] = {}
    for code in set(side) | set(other):
        if side_n == 0:
            target = other.get(code, 0)
        else:
            numer = (side.get(code, 0) + other.get(code, 0)) * side_n
            target = (numer + pooled - 1) // pooled
        gap = target - side.get(code, 0)
        if gap > 0:
            deficit[code] = gap
    return dict(sorted(deficit.items()))


class InSituPrototypeGenerator(EnrichmentGenerator):
    """MaxSim prototypes from the plan. The plan no longer carries stub values."""

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
            notes="label-padded in-situ prototype",
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
        path = holdout_dir / f"{table.replace('/', '__')}.csv"
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
        "holdout_collections": plan.holdout_collections,
        "holdout_tag_counts": plan.holdout_counts,
        "holdout_augmentation": plan.holdout_variants,
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
    column_types: dict[tuple[str, str], str] = {}
    for cell in cells:
        raw_type = str(cell.get("column_type") or "").strip()
        if raw_type and (cell["table"], cell["column"]) not in column_types:
            column_types[(cell["table"], cell["column"])] = raw_type
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
        stored_type = column_types.get((table, column)) or _infer_column_type(values[:10])
        rows.append(Row(
            table=table,
            column=column,
            column_type=stored_type or "object",
            sample_values=values[:8],
            siblings_full=siblings,
            mnemonic="",
            code=code,
        ))
    return rows


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
