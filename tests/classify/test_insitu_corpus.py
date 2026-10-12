"""In-situ split, amplification, and synthetic NHSVM rows."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atelier.classify.insitu_corpus import (
    BASE_VARIANTS,
    InSituPrototypeGenerator,
    prepare_insitu,
    split_projected,
    support_counts,
)
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
from atelier.enrichment.verifiers import run_verifier_suite
from atelier.flows.resident import load_sweep_samples


def _taxonomy() -> HierarchicalCategorySet:
    cats = [
        ReferenceCategory(
            code="SDG.A.1", label="invoice identifier", embedding_text="invoice",
            abbrev="INVOICE_ID", parent_code="SDG.A",
        ),
        ReferenceCategory(
            code="SDG.B", label="status", embedding_text="status",
            abbrev="STATUS", parent_code="SDG", description="A status flag for the row.",
        ),
        ReferenceCategory(
            code="SDG.ROLE.PARENTING_ROLE", label="Parenting Role",
            embedding_text="role", abbrev="PARENTING_ROLE", parent_code="SDG.ROLE",
        ),
    ]
    return HierarchicalCategorySet(name="sdg", categories=cats)


def _footprint(root: Path) -> None:
    pin = root / "ddl" / "pin1"
    pin.mkdir(parents=True)
    # A naming map sitting beside the cells must not be consulted.
    pq.write_table(pa.table({
        "table_name": ["billing"],
        "col_name": ["invoice_id"],
        "slot_ref": ["TotallyDifferent"],
        "code": ["SDG.B"],
    }), pin / "naming_map.parquet")
    pq.write_table(pa.table({
        "table_name": ["billing", "billing", "billing", "other", "other", "lone"],
        "col_name": ["amount", "invoice_id", "note", "b", "invoice_id", "id"],
        "value": ["10", "INV-1", "hello", "x", "INV-9", "1"],
        "row_ix": [0, 0, 0, 0, 0, 0],
    }), pin / "base_rows.parquet")


def test_split_puts_wide_tables_on_both_sides() -> None:
    support, holdout = split_projected({
        "billing": ["amount", "invoice_id", "note"],
        "other": ["b", "invoice_id"],
        "lone": ["id"],
    })
    assert set(support).isdisjoint(holdout)
    assert len(support) + len(holdout) == 6
    billing_sides = {side for side, cols in (("support", support), ("holdout", holdout))
                     for table, _col in cols if table == "billing"}
    assert billing_sides == {"support", "holdout"}
    assert ("billing", "invoice_id") in support
    assert ("other", "invoice_id") in support


def test_support_census_ignores_holdout_names() -> None:
    support, holdout = split_projected({
        "billing": ["amount", "invoice_id", "note"],
        "other": ["b", "invoice_id"],
        "lone": ["id"],
    })
    counts = support_counts(support, _taxonomy())
    assert counts["SDG.A.1"] == 2
    assert counts["SDG.B"] == 0
    assert ("billing", "note") in holdout
    recount = support_counts(holdout, _taxonomy())
    assert recount["SDG.A.1"] == 0


def test_prepare_amplifies_unseen_codes_and_writes_a_blind_holdout(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    artifacts = tmp_path / "build"
    _footprint(corpora)
    plan = prepare_insitu(
        _taxonomy(), artifacts, corpora=corpora,
        base_variants=2, amplified_variants=5, rows_per_column=3, floor=2,
    )
    assert plan.variants["SDG.A.1"] == 2
    assert plan.variants["SDG.B"] == 5
    assert plan.variants["SDG.ROLE.PARENTING_ROLE"] == 5
    by_code: dict[str, int] = {}
    for row in plan.rows:
        by_code[row.code] = by_code.get(row.code, 0) + 1
        assert row.code in plan.variants
    assert by_code["SDG.B"] == 5
    assert by_code["SDG.A.1"] == 2
    split = json.loads((artifacts / "relational_split" / "split.json").read_text())
    assert "SDG.B" in split["under_represented"]
    assert "SDG.A.1" not in split["under_represented"]
    assert "slot_ref" not in json.dumps(split)
    holdout_csv = (artifacts / "relational_split" / "holdout" / "tables" / "billing.csv")
    text = holdout_csv.read_text()
    assert "invoice_id" not in text.splitlines()[0]
    assert "INV-1" not in text
    source = Path(prepare_insitu.__code__.co_filename).read_text().lower()
    assert "naming_map" not in source
    assert "slot_ref" not in source


def test_prototypes_are_synthetic_and_pass_the_verifier(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    _footprint(corpora)
    plan = prepare_insitu(
        _taxonomy(), tmp_path / "build", corpora=corpora,
        base_variants=1, amplified_variants=2, rows_per_column=2,
    )
    generated = InSituPrototypeGenerator(plan).generate(
        {"code": "SDG.ROLE.PARENTING_ROLE", "label": "Parenting Role", "mnemonic": "parenting_role"},
        taxonomy_context={"expected_parent_path": ["Role"]},
    )
    prototypes = generated.enrichment["prototype_values"]
    assert prototypes
    assert "INV-1" not in prototypes
    assert "INV-9" not in prototypes
    report = run_verifier_suite(generated.enrichment, expected_parent_path=["Role"])
    assert report.passed
    assert InSituPrototypeGenerator(plan).name == "ontology:insitu"


def test_missing_corpora_still_synthesizes_every_code(tmp_path) -> None:
    plan = prepare_insitu(
        _taxonomy(), tmp_path / "build", corpora=tmp_path / "absent",
        base_variants=BASE_VARIANTS, amplified_variants=3, rows_per_column=2,
    )
    assert plan.support == []
    assert plan.holdout == []
    assert set(plan.variants) == {"SDG.A.1", "SDG.B", "SDG.ROLE.PARENTING_ROLE"}
    assert all(n == 3 for n in plan.variants.values())
    assert {row.code for row in plan.rows} == set(plan.variants)


def test_projected_holdout_source_loads_only_the_holdout(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    artifacts = tmp_path / "build"
    _footprint(corpora)
    prepare_insitu(_taxonomy(), artifacts, corpora=corpora, rows_per_column=2)
    from types import SimpleNamespace
    samples = load_sweep_samples(
        SimpleNamespace(artifact_root=str(artifacts), classify_sample_size=10),
        "projected-holdout",
    )
    names = {(table.name, col.name) for table in samples for col in table.columns}
    assert ("billing", "invoice_id") not in names
    assert names
    assert all(col.reference_code in (None, "") for table in samples for col in table.columns)
