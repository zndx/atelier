"""Ontology-labeled NHSVM rows and MaxSim prototypes from sdg-corpora."""

from __future__ import annotations

import csv
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atelier.classify.ontology_corpus import (
    LabeledColumn,
    OntologyCorpusError,
    OntologyLabelGenerator,
    corpora_root,
    exemplars_by_code,
    labeled_rows_from_corpora,
    ontology_generator,
    rows_from_labeled_columns,
)
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
from atelier.enrichment.verifiers import run_verifier_suite


def _taxonomy() -> HierarchicalCategorySet:
    cats = [
        ReferenceCategory(code="SDG.A.1", label="invoice identifier", embedding_text="invoice", parent_code="SDG.A"),
        ReferenceCategory(code="SDG.B", label="status", embedding_text="status", parent_code="SDG"),
        ReferenceCategory(code="SDG.ROLE.PARENTING_ROLE", label="Parenting Role", embedding_text="role", parent_code="SDG.ROLE"),
    ]
    return HierarchicalCategorySet(name="sdg", categories=cats)


def _write_vocab(root: Path, rows: list[dict]) -> None:
    path = root / "vocabulary" / "annotations.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["code", "label", "abbrev", "parent_code", "family"])
        writer.writeheader()
        writer.writerows(rows)


def _write_footprint(root: Path, naming: list[dict], cells: list[dict]) -> None:
    pin = root / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(naming), pin / "naming_map.parquet")
    pq.write_table(pa.Table.from_pylist(cells), pin / "base_rows.parquet")


def _fixture(root: Path) -> None:
    _write_vocab(root, [
        {"code": "SDG.A.1", "label": "invoice identifier", "abbrev": "INVOICE_ID", "parent_code": "SDG.A", "family": "catalog"},
        {"code": "SDG.B", "label": "status", "abbrev": "STATUS", "parent_code": "SDG", "family": "catalog"},
        {"code": "SDG.ROLE.PARENTING_ROLE", "label": "Parenting Role", "abbrev": "PARENTING_ROLE", "parent_code": "SDG.ROLE", "family": "catalog"},
        {"code": "SDG.DOM.PARENTING_ROLE", "label": "Parenting Role domain", "abbrev": "PARENTING_ROLE", "parent_code": "SDG.DOM", "family": ""},
        {"code": "SDG.GENERIC.ORGANIZATION", "label": "Organization", "abbrev": "ORGANIZATION", "parent_code": "SDG.GENERIC", "family": "catalog"},
        {"code": "SDG.DOM.ORGANIZATION", "label": "Organization domain", "abbrev": "ORGANIZATION", "parent_code": "SDG.DOM", "family": ""},
    ])
    _write_footprint(root, [
        {"table_name": "billing", "col_name": "invoice_id", "slot_ref": "InvoiceId", "template_id": "invoice_id", "code": ""},
        {"table_name": "billing", "col_name": "amount", "slot_ref": "DisabledAdult", "template_id": "invoice_id", "code": ""},
        {"table_name": "billing", "col_name": "id", "slot_ref": "__pk__", "template_id": "parenting_role", "code": ""},
        {"table_name": "billing", "col_name": "encoding", "slot_ref": "data:sdg:hasEncoding", "template_id": "invoice_id", "code": ""},
        {"table_name": "billing", "col_name": "org_id", "slot_ref": "Organization", "template_id": "invoice_id", "code": ""},
        {"table_name": "billing", "col_name": "stamped", "slot_ref": "NotAClass", "template_id": "invoice_id", "code": "SDG.B"},
    ], [
        {"table_name": "billing", "col_name": "invoice_id", "value": "INV-1"},
        {"table_name": "billing", "col_name": "invoice_id", "value": "INV-2"},
        {"table_name": "billing", "col_name": "invoice_id", "value": "INV-1"},
        {"table_name": "billing", "col_name": "amount", "value": "10"},
        {"table_name": "billing", "col_name": "id", "value": "ROLE-0001"},
        {"table_name": "billing", "col_name": "stamped", "value": "open"},
    ])


def test_labeled_rows_keep_codes_in_the_taxonomy() -> None:
    columns = [
        LabeledColumn("billing", "invoice_id", "SDG.A.1", ["INV-1", "INV-2"], ["invoice_id", "amount", "note"]),
        LabeledColumn("billing", "amount", "SDG.NOT_IN_TAX", ["10", "20"], ["invoice_id", "amount", "note"]),
        LabeledColumn("billing", "note", "", [], ["invoice_id", "amount", "note"]),
    ]
    rows = rows_from_labeled_columns(columns, _taxonomy())
    assert len(rows) == 1
    assert rows[0].code == "SDG.A.1"
    assert rows[0].column == "invoice_id"
    assert rows[0].sample_values[:2] == ["INV-1", "INV-2"]
    assert rows[0].siblings_full == ["amount", "note"]


def test_generator_prototypes_are_column_values() -> None:
    columns = [
        LabeledColumn("billing", "invoice_id", "SDG.A.1", ["INV-1", "INV-2"], ["invoice_id"]),
    ]
    exemplars = exemplars_by_code(columns, _taxonomy())
    assert exemplars["SDG.A.1"][:2] == ["INV-1", "INV-2"]
    generated = OntologyLabelGenerator(exemplars).generate(
        {"code": "SDG.A.1", "label": "invoice identifier", "mnemonic": "inv"},
        taxonomy_context={"expected_parent_path": ["genus A"]},
    )
    prototypes = generated.enrichment["prototype_values"]
    assert prototypes[0] == "INV-1"
    assert prototypes[1] == "INV-2"
    assert "stub-prototype" not in " ".join(prototypes)
    report = run_verifier_suite(
        generated.enrichment, expected_parent_path=["genus A"],
    )
    assert report.passed
    assert OntologyLabelGenerator(exemplars).name == "ontology:sdg-corpora"


def test_short_exemplar_list_pads_with_the_ontology_label() -> None:
    generated = OntologyLabelGenerator({"SDG.B": ["open"]}).generate(
        {"code": "SDG.B", "label": "status"},
        taxonomy_context={"expected_parent_path": []},
    )
    assert generated.enrichment["prototype_values"] == ["open", "status", "status"]
    assert generated.enrichment["parent_path"] == []


def test_no_taxonomy_overlap_keeps_the_llm_generator_path(tmp_path) -> None:
    _fixture(tmp_path)
    other = HierarchicalCategorySet(name="other", categories=[
        ReferenceCategory(code="OTHER.X", label="other", embedding_text="other", parent_code=""),
    ])
    assert ontology_generator(object(), other, corpora=tmp_path) is None
    assert labeled_rows_from_corpora(other, corpora=tmp_path) == []


def test_missing_checkout_fails_closed(tmp_path) -> None:
    missing = tmp_path / "absent"
    with pytest.raises(OntologyCorpusError):
        ontology_generator(object(), _taxonomy(), corpora=missing)


def test_footprint_join_uses_abbrev_catalog_family_and_explicit_code(tmp_path) -> None:
    _fixture(tmp_path)
    rows = labeled_rows_from_corpora(None, corpora=tmp_path)
    by_column = {row.column: row for row in rows}
    assert by_column["invoice_id"].code == "SDG.A.1"
    assert by_column["invoice_id"].sample_values == ["INV-1", "INV-2"]
    assert "amount" not in by_column
    assert "encoding" not in by_column
    assert by_column["id"].code == "SDG.ROLE.PARENTING_ROLE"
    assert by_column["org_id"].code == "SDG.GENERIC.ORGANIZATION"
    assert by_column["stamped"].code == "SDG.B"
    assert by_column["stamped"].sample_values == ["open"]
    kept = {row.column for row in labeled_rows_from_corpora(_taxonomy(), corpora=tmp_path)}
    assert "org_id" not in kept
    assert "invoice_id" in kept
    generator = ontology_generator(object(), _taxonomy(), corpora=tmp_path)
    assert generator is not None
    assert generator.name == "ontology:sdg-corpora"
    from atelier.classify import ontology_corpus
    text = Path(ontology_corpus.__file__).read_text().lower()
    assert "aegir" not in text
    assert "release_dir" not in text


def test_published_checkout_is_the_submodule() -> None:
    root = corpora_root()
    assert root.name == "sdg-corpora"
    assert (root / "vocabulary" / "annotations.csv").is_file()
    rows = labeled_rows_from_corpora(None)
    assert len(rows) > 100
    assert all(row.code.startswith("SDG.") for row in rows)


def test_two_footprint_pins_fail_closed(tmp_path) -> None:
    _fixture(tmp_path)
    extra = tmp_path / "ddl" / "pin2"
    extra.mkdir()
    for name in ("naming_map.parquet", "base_rows.parquet"):
        (extra / name).write_bytes((tmp_path / "ddl" / "pin1" / name).read_bytes())
    with pytest.raises(OntologyCorpusError):
        labeled_rows_from_corpora(_taxonomy(), corpora=tmp_path)
