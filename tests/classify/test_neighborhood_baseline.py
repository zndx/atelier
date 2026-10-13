"""Genus-neighborhood baseline: whole tables, root leaves included."""

from __future__ import annotations

from atelier.classify.extension_patterns import code_iri
from atelier.classify.neighborhood_baseline import (
    ancestor_frame,
    label_kind,
    placement,
    prepare_neighborhood_baseline,
    split_whole_tables,
)
from atelier.classify.taxonomy import load_annotations_from_filesystem
from tests.classify.explicit_sdg import (
    PDF_EXPORT,
    PROCESS,
    relational_procedure,
    taxonomy,
)

_DESCRIPTIVE = "SDG.ICE.DESCRIPTIVE"
_BENEFIT = "SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION"
_CHAIN = (
    "SDG.GDC", "SDG.ICE", _DESCRIPTIVE, _BENEFIT, PROCESS, PDF_EXPORT,
)


def _cells(table: str, columns: dict[str, list[str]]) -> list[dict]:
    cells = []
    for column, values in columns.items():
        for index, value in enumerate(values):
            cells.append({
                "table": table,
                "column": column,
                "value": value,
                "row_ix": index,
            })
    return cells


def test_one_table_per_label_is_held_out_whole() -> None:
    labels = {
        "alpha": {_BENEFIT},
        "beta": {_BENEFIT},
        "gamma": {PDF_EXPORT},
        "delta": {PDF_EXPORT},
    }
    support, holdout = split_whole_tables(labels)
    assert support == ["beta", "gamma"]
    assert holdout == ["alpha", "delta"]
    singleton = {"only": {PDF_EXPORT}, "alpha": {_BENEFIT}, "beta": {_BENEFIT}}
    support, holdout = split_whole_tables(singleton)
    assert "only" in support
    assert "only" not in holdout


def test_a_label_with_two_rich_tables_holds_out_a_rich_one() -> None:
    benefit = "SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION"
    labels = {
        "stub_table": {benefit},
        "rich_a": {benefit},
        "rich_b": {benefit},
    }
    support, holdout = split_whole_tables(
        labels, rich={("rich_a", benefit), ("rich_b", benefit)},
    )
    assert holdout == ["rich_a"]
    assert support == ["rich_b", "stub_table"]


def test_child_stays_itself_and_a_root_leaf_is_included() -> None:
    cats = taxonomy(_CHAIN)
    benefit_iri = code_iri(_BENEFIT)
    generators = {
        benefit_iri: relational_procedure(benefit_iri, "option", "plan"),
    }
    cells = []
    cells += _cells("alpha", {"option": ["vest-1", "vest-2"]})
    cells += _cells("beta", {"option": ["vest-3"]})
    cells += _cells("gamma", {"badge": ["A-14"], "unit": ["north"]})
    cells += _cells("delta", {"badge": ["B-2"]})
    assignments = {
        ("alpha", "option"): _BENEFIT,
        ("beta", "option"): _BENEFIT,
        ("gamma", "badge"): PDF_EXPORT,
        ("delta", "badge"): PDF_EXPORT,
    }
    plan = prepare_neighborhood_baseline(
        cats, assignments=assignments, cells=cells, generators=generators,
        variants_per_code=2, rows_per_column=2,
    )
    assert plan.holdout_tables == ["alpha", "delta"]
    assert {row.table for row in plan.holdout_rows}.isdisjoint(plan.support_tables)
    assert {row.table for row in plan.support_rows}.isdisjoint(plan.holdout_tables)
    codes = {row.code for row in plan.support_rows + plan.holdout_rows}
    assert _BENEFIT in codes
    assert PROCESS in codes
    assert PDF_EXPORT not in codes
    assert plan.generatable == [_BENEFIT]
    assert PROCESS in plan.unfilled
    assert {row.code for row in plan.synthetic_rows} == {_BENEFIT}
    assert len(plan.synthetic_rows) == 2
    benefit = next(row for row in plan.support_rows if row.code == _BENEFIT)
    assert _BENEFIT not in " ".join(benefit.sample_values)
    frame = ancestor_frame(cats, plan.labels)
    assert {cat.code for cat in frame.all_categories} >= {
        "SDG.GDC", "SDG.ICE", _DESCRIPTIVE, _BENEFIT, PROCESS,
    }
    assert PDF_EXPORT not in {cat.code for cat in frame.all_categories}
    assert label_kind(cats, _BENEFIT) == "child"
    assert label_kind(cats, PROCESS) == "root"
    assert placement(cats, _BENEFIT, _BENEFIT) == "exact"
    assert placement(cats, _BENEFIT, _DESCRIPTIVE) == "ancestor"
    assert placement(cats, _BENEFIT, PROCESS) == "miss"


def test_published_footprint_holds_out_whole_tables() -> None:
    cats = load_annotations_from_filesystem(
        "external/sdg-corpora/vocabulary/annotations.csv",
    )
    plan = prepare_neighborhood_baseline(cats, variants_per_code=0)
    assert plan.support_tables
    assert plan.holdout_tables
    assert set(plan.support_tables).isdisjoint(plan.holdout_tables)
    assert {row.table for row in plan.holdout_rows}.isdisjoint(plan.support_tables)
    assert {row.table for row in plan.support_rows}.isdisjoint(plan.holdout_tables)
    codes = {row.code for row in plan.support_rows + plan.holdout_rows}
    for root in (PROCESS, "SDG.ROLE", "SDG.GENERIC", "SDG.GDC"):
        assert root in codes
    assert plan.synthetic_rows == []
    assert "SDG.PROCESS" in plan.unfilled
    child_rows = [
        row for row in plan.holdout_rows if label_kind(cats, row.code) == "child"
    ]
    root_rows = [
        row for row in plan.holdout_rows if label_kind(cats, row.code) == "root"
    ]
    assert child_rows
    assert root_rows
    assert len(plan.labels) < len(cats.all_categories)
