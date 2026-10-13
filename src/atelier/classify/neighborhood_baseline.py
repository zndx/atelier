"""In-situ NHSVM baseline on the genus neighborhood.

The scored corpus is the published DDL footprint: each column already
has a SKOS code, and each template is one table. Cross-family foreign
keys were not emitted, so the whole table is the bundle that stays on
one side of the exam. Collection chapter directories still have no
column answer key, so they are not this baseline.

Order 1 of ``associate_genera`` is the label. A child of a structural
genus keeps that child. A leaf on a root takes the root, and those
roots stay in the baseline. Synthetic tables are emitted only for a
label whose published IRI has a caller-supplied relational generator.
The seven roots have no such generator.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping

from atelier.classify.decisions_walk import associate_genera, genus_term
from atelier.classify.extension_patterns import code_iri
from atelier.classify.insitu_corpus import (
    ROWS_PER_COLUMN,
    _support_rows,
    load_projected_cells,
    synthetic_rows,
)
from atelier.classify.ontology_corpus import load_labeled_columns
from atelier.classify.taxonomy import HierarchicalCategorySet


@dataclass
class NeighborhoodBaseline:
    """Real support rows, real holdout rows, and the synthetic supplement."""

    support_rows: list[Any]
    holdout_rows: list[Any]
    synthetic_rows: list[Any]
    support_tables: list[str]
    holdout_tables: list[str]
    support_counts: dict[str, int]
    holdout_counts: dict[str, int]
    generatable: list[str]
    unfilled: list[str]
    variants_per_code: int = 0
    labels: list[str] = field(default_factory=list)


def ancestor_frame(
    category_set: Any,
    labels: list[str] | set[str],
) -> HierarchicalCategorySet:
    """Taxonomy of the training labels closed under ancestors.

    A parent that is not itself a column label stays in the frame so
    the factorized head can score the path. Codes outside those paths
    are omitted.
    """
    needed: set[str] = set()
    for code in labels:
        needed.update(category_set.path_from_root(str(code)))
    chosen = [
        cat for cat in category_set.all_categories if cat.code in needed
    ]
    return HierarchicalCategorySet(name="neighborhood-baseline", categories=chosen)


def split_whole_tables(
    table_labels: Mapping[str, set[str]],
) -> tuple[list[str], list[str]]:
    """Hold out one whole table per label that has another table.

    A candidate table is eligible when every label on it still has a
    different table outside the holdout. A label that lives on one
    table stays in support. Tables are never split.
    """
    members: dict[str, list[str]] = {}
    for table, labels in table_labels.items():
        for label in labels:
            members.setdefault(label, []).append(table)
    for label in members:
        members[label] = sorted(set(members[label]))
    holdout: set[str] = set()

    def eligible(table: str) -> bool:
        if table in holdout:
            return False
        for label in table_labels[table]:
            remaining = [
                other for other in members[label]
                if other != table and other not in holdout
            ]
            if not remaining:
                return False
        return True

    order = sorted(members, key=lambda code: (len(members[code]), code))
    for label in order:
        if len(members[label]) < 2:
            continue
        if any(table in holdout for table in members[label]):
            continue
        candidates = [table for table in members[label] if eligible(table)]
        if not candidates:
            continue
        candidates.sort(key=lambda table: (len(table_labels[table]), table))
        holdout.add(candidates[0])
    support = sorted(table for table in table_labels if table not in holdout)
    return support, sorted(holdout)


def placement(category_set: Any, gold: str, predicted: str) -> str:
    """Where a prediction sits relative to the gold code.

    ``exact`` is the gold code. ``ancestor`` and ``descendant`` lie on
    its path. ``same_genus`` shares ``genus_term`` and leaves the path.
    Anything else is ``miss``.
    """
    if predicted == gold:
        return "exact"
    gold_path = set(category_set.path_from_root(gold))
    pred_path = set(category_set.path_from_root(predicted))
    if predicted in gold_path:
        return "ancestor"
    if gold in pred_path:
        return "descendant"
    if genus_term(category_set, predicted) == genus_term(category_set, gold):
        return "same_genus"
    return "miss"


def label_kind(category_set: Any, code: str) -> str:
    """``root`` has no parent. ``child`` is the child of a structural genus."""
    parent = category_set.parent.get(code)
    if not parent:
        return "root"
    grandparent = category_set.parent.get(parent)
    if grandparent and category_set.children.get(parent):
        return "child"
    return "other"


def prepare_neighborhood_baseline(
    category_set: Any,
    *,
    assignments: dict[tuple[str, str], str] | None = None,
    cells: list[dict] | None = None,
    corpora: str | None = None,
    generators: dict | None = None,
    variants_per_code: int = 0,
    rows_per_column: int = ROWS_PER_COLUMN,
    seed: int = 42,
) -> NeighborhoodBaseline:
    """Lift footprint columns onto the genus neighborhood and split tables.

    ``assignments`` and ``cells`` override the published footprint.
    ``generators`` maps a published SKOS IRI to a relational table
    callable. ``variants_per_code`` synthetic tables are added for each
    label that has one. Labels without one are listed on ``unfilled``
    and stay real rows only.
    """
    if assignments is None:
        loaded = load_labeled_columns(corpora)
        assignments = {(item.table, item.column): item.code for item in loaded}
    lifted = associate_genera(assignments, category_set, order=1)
    if cells is None:
        cells = load_projected_cells(corpora)
    table_columns: dict[str, list[str]] = {}
    for cell in cells:
        columns = table_columns.setdefault(str(cell["table"]), [])
        column = str(cell["column"])
        if column and column not in columns:
            columns.append(column)
    for table, column in lifted:
        columns = table_columns.setdefault(table, [])
        if column not in columns:
            columns.append(column)
    table_labels: dict[str, set[str]] = {}
    for (table, _column), label in lifted.items():
        table_labels.setdefault(table, set()).add(label)
    support_tables, holdout_tables = split_whole_tables(table_labels)
    support_keys = [
        (table, column)
        for table in support_tables
        for column in table_columns.get(table, [])
    ]
    holdout_keys = [
        (table, column)
        for table in holdout_tables
        for column in table_columns.get(table, [])
    ]
    support_rows = _support_rows(
        cells, support_keys, lifted, rows_per_column=rows_per_column,
    )
    holdout_rows = _support_rows(
        cells, holdout_keys, lifted, rows_per_column=rows_per_column,
    )
    labels = sorted({row.code for row in support_rows + holdout_rows})
    supplied = generators or {}
    generatable = [
        code for code in labels if (code_iri(code) or "") in supplied
    ]
    unfilled = [code for code in labels if code not in generatable]
    synthetic: list[Any] = []
    if variants_per_code > 0 and generatable:
        variants = {code: variants_per_code for code in generatable}
        gen_map = {code: supplied[code_iri(code)] for code in generatable}
        synthetic = synthetic_rows(
            variants, gen_map, rows_per_column=rows_per_column, seed=seed,
        )
    return NeighborhoodBaseline(
        support_rows=support_rows,
        holdout_rows=holdout_rows,
        synthetic_rows=synthetic,
        support_tables=support_tables,
        holdout_tables=holdout_tables,
        support_counts=dict(Counter(row.code for row in support_rows)),
        holdout_counts=dict(Counter(row.code for row in holdout_rows)),
        generatable=generatable,
        unfilled=unfilled,
        variants_per_code=variants_per_code,
        labels=labels,
    )
