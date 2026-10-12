"""In-situ NHSVM rows and MaxSim prototypes for an unseen relational corpus.

The ontology (the loaded taxonomy) is the only label source. The
projected relational footprint contributes table and column names and
cell values, never a published column code. Those tables are split
into a support half and a holdout half. The support half is a census:
a column name that matches one vocabulary abbrev or label counts as
evidence that the code is already present. Codes below the floor get
more synthetic columns. The holdout half is the classification target.
Training rows and prototype text are drawn from Atelier's generator.
Codes the keyword matcher misses use insitu_extensions.py when that
map contains the term's exact vocabulary IRI. A term with neither a
keyword shape nor an exact IRI generator is left unshaped.
"""

from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from atelier.enrichment.llm_generator import EnrichmentGenerator, GenerationResult

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")

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
) -> InSituPlan:
    """Split the projected tables and synthesize the training plan.

    ``corpora`` defaults to the local sdg-corpora checkout. A checkout
    that is absent yields an empty split, and every code is amplified.
    """
    cells = load_projected_cells(corpora)
    by_table = _columns_by_table(cells)
    support, holdout = split_projected(by_table)
    counts = support_counts(support, category_set)
    variants = {
        code: (amplified_variants if counts.get(code, 0) < floor else base_variants)
        for code in _codes(category_set)
    }
    generators, extended = _generators(category_set, extra_generators)
    exemplars = _exemplars(generators, seed=seed)
    plan = InSituPlan(
        support=support,
        holdout=holdout,
        counts=counts,
        variants=variants,
        exemplars=exemplars,
        artifact_root=relational_split_root(artifact_root),
        extended=sorted(extended),
    )
    plan.rows = synthetic_rows(
        category_set, variants, generators,
        rows_per_column=rows_per_column, seed=seed,
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


def support_counts(support: list[tuple[str, str]], category_set) -> dict[str, int]:
    """How many support columns name each ontology code.

    A column counts only when its name matches exactly one abbrev or
    label. The holdout is not an argument: those names do not steer
    amplification.
    """
    index = _name_index(category_set)
    counts = {code: 0 for code in _codes(category_set)}
    for _table, column in support:
        code = index.get(_scream(column))
        if code is None or code not in counts:
            continue
        counts[code] += 1
    return counts


def synthetic_rows(
    category_set,
    variants: dict[str, int],
    generators: dict[str, Any],
    *,
    rows_per_column: int = ROWS_PER_COLUMN,
    seed: int = 42,
    table_width: int = 8,
) -> list[Any]:
    """NHSVM rows whose codes are assigned by the generator."""
    from atelier.classify.ml_train import _infer_column_type
    from atelier.classify.synth import _generate_opaque_names, _generate_semantic_names
    from atelier.optimize.svm.reflect import Row

    rng = random.Random(seed)
    seen: set[str] = set()
    drafted: list[tuple[str, str, list[str]]] = []
    cats = {c.code: c for c in _categories(category_set)}
    for code in _codes(category_set):
        width = int(variants.get(code, 0))
        if width <= 0:
            continue
        cat = cats[code]
        gen = generators.get(code)
        if gen is None:
            continue
        names = _column_names(
            cat, width, rng, seen,
            semantic=_generate_semantic_names,
            opaque=_generate_opaque_names,
        )
        for name in names:
            values = [str(gen(rng)) for _ in range(rows_per_column)]
            drafted.append((code, name, values))

    rows: list[Any] = []
    for start in range(0, len(drafted), table_width):
        chunk = drafted[start:start + table_width]
        table = f"insitu_{start // table_width + 1:03d}"
        siblings = [name for _code, name, _values in chunk]
        for code, name, values in chunk:
            rows.append(Row(
                table=table,
                column=name,
                column_type=_infer_column_type(values[:10]) or "object",
                sample_values=[v for v in values if v][:8],
                siblings_full=[s for s in siblings if s != name],
                mnemonic="",
                code=code,
            ))
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


def _generators(category_set, extra_generators: dict | None = None) -> tuple[dict[str, Any], set[str]]:
    from atelier.classify.extension_patterns import code_iri
    from atelier.classify.synth_registry import _infer_generator

    if extra_generators is None:
        extra_generators = _installed_generators()
    out = {}
    extended: set[str] = set()
    for cat in _categories(category_set):
        inferred = _infer_generator(cat)
        if inferred is not None:
            out[cat.code] = inferred
            continue
        iri = code_iri(cat.code)
        generate = extra_generators.get(iri) if iri else None
        if generate is None:
            continue
        out[cat.code] = generate
        extended.add(cat.code)
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


def _exemplars(generators: dict[str, Any], *, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed + 1)
    exemplars: dict[str, list[str]] = {}
    for code, gen in generators.items():
        seen: set[str] = set()
        values: list[str] = []
        for _ in range(PROTOTYPES_PER_CODE * 3):
            text = str(gen(rng)).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            values.append(text)
            if len(values) >= PROTOTYPES_PER_CODE:
                break
        exemplars[code] = values
    return exemplars


def _column_names(cat, count: int, rng, seen: set[str], *, semantic, opaque) -> list[str]:
    proposed = semantic(
        getattr(cat, "label", "") or cat.code,
        getattr(cat, "abbrev", "") or "",
        getattr(cat, "common_names", "") or "",
        rng,
        count,
    )
    proposed = list(proposed) + list(opaque(rng, count))
    names: list[str] = []
    for candidate in proposed:
        name = candidate
        if name in seen:
            name = f"{candidate}_{cat.code.replace('.', '_')}"
        if name in seen or not name:
            continue
        seen.add(name)
        names.append(name)
        if len(names) >= count:
            return names
    while len(names) < count:
        extra = f"syn_{cat.code.replace('.', '_')}_{len(names)}"
        if extra not in seen:
            seen.add(extra)
            names.append(extra)
    return names


def _name_index(category_set) -> dict[str, str | None]:
    """Map a screamed name to one code, or None when the name is shared."""
    buckets: dict[str, set[str]] = {}
    for cat in _categories(category_set):
        keys = {_scream(getattr(cat, "abbrev", "") or "")}
        label = _scream(getattr(cat, "label", "") or "")
        if label:
            keys.add(label)
        for key in keys:
            if not key:
                continue
            buckets.setdefault(key, set()).add(cat.code)
    return {
        key: (next(iter(codes)) if len(codes) == 1 else None)
        for key, codes in buckets.items()
    }


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


def _scream(name: str) -> str:
    return _CAMEL.sub("_", name).replace("-", "_").replace(" ", "_").upper()
