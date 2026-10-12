"""Ontology-labeled NHSVM rows and MaxSim prototypes from sdg-corpora."""

from __future__ import annotations

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
from tests.classify.explicit_sdg import (
    ORGANIZATION,
    PARENTING,
    PDF_EXPORT,
    YOUNG_RESEARCHER,
    published_rows,
    taxonomy,
    write_annotations,
)

_FRAME = (PDF_EXPORT, PARENTING, YOUNG_RESEARCHER)
_VOCAB = (*_FRAME, ORGANIZATION, "SDG.DOM.ORGANIZATION")


def _write_footprint(root: Path, naming: list[dict], cells: list[dict]) -> None:
    pin = root / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(naming), pin / "naming_map.parquet")
    pq.write_table(pa.Table.from_pylist(cells), pin / "base_rows.parquet")


def _fixture(root: Path) -> None:
    write_annotations(root / "vocabulary" / "annotations.csv", _VOCAB)
    _write_footprint(root, [
        {"table_name": "billing", "col_name": "doc_ref", "slot_ref": "PdfExportProcess", "template_id": "doc_ref", "code": ""},
        {"table_name": "billing", "col_name": "amount", "slot_ref": "DisabledAdult", "template_id": "doc_ref", "code": ""},
        {"table_name": "billing", "col_name": "surrogate", "slot_ref": "__pk__", "template_id": "parenting_role", "code": ""},
        {"table_name": "billing", "col_name": "encoding", "slot_ref": "data:sdg:hasEncoding", "template_id": "doc_ref", "code": ""},
        {"table_name": "billing", "col_name": "org_id", "slot_ref": "Organization", "template_id": "doc_ref", "code": ""},
        {"table_name": "billing", "col_name": "stamped", "slot_ref": "NotAClass", "template_id": "doc_ref", "code": YOUNG_RESEARCHER},
    ], [
        {"table_name": "billing", "col_name": "doc_ref", "value": "INV-1"},
        {"table_name": "billing", "col_name": "doc_ref", "value": "INV-2"},
        {"table_name": "billing", "col_name": "doc_ref", "value": "INV-1"},
        {"table_name": "billing", "col_name": "amount", "value": "10"},
        {"table_name": "billing", "col_name": "surrogate", "value": "ROLE-0001"},
        {"table_name": "billing", "col_name": "stamped", "value": "open"},
    ])


def test_labeled_rows_keep_codes_in_the_taxonomy() -> None:
    columns = [
        LabeledColumn("billing", "doc_ref", PDF_EXPORT, ["INV-1", "INV-2"], ["doc_ref", "amount", "note"]),
        LabeledColumn("billing", "org_id", ORGANIZATION, ["10", "20"], ["doc_ref", "org_id", "note"]),
        LabeledColumn("billing", "note", "", [], ["doc_ref", "amount", "note"]),
    ]
    rows = rows_from_labeled_columns(columns, taxonomy(_FRAME))
    assert len(rows) == 1
    assert rows[0].code == PDF_EXPORT
    assert rows[0].column == "doc_ref"
    assert rows[0].sample_values[:2] == ["INV-1", "INV-2"]
    assert rows[0].siblings_full == ["amount", "note"]


def test_generator_prototypes_are_column_values() -> None:
    columns = [
        LabeledColumn("billing", "doc_ref", PDF_EXPORT, ["INV-1", "INV-2"], ["doc_ref"]),
    ]
    frame = taxonomy(_FRAME)
    exemplars = exemplars_by_code(columns, frame)
    assert exemplars[PDF_EXPORT][:2] == ["INV-1", "INV-2"]
    label = published_rows([PDF_EXPORT])[0]["label"]
    generated = OntologyLabelGenerator(exemplars).generate(
        {"code": PDF_EXPORT, "label": label, "mnemonic": "pdf_export"},
        taxonomy_context={"expected_parent_path": ["Process"]},
    )
    prototypes = generated.enrichment["prototype_values"]
    assert prototypes[0] == "INV-1"
    assert prototypes[1] == "INV-2"
    assert "stub-prototype" not in " ".join(prototypes)
    report = run_verifier_suite(
        generated.enrichment, expected_parent_path=["Process"],
    )
    assert report.passed
    assert OntologyLabelGenerator(exemplars).name == "ontology:sdg-corpora"


def test_short_exemplar_list_pads_with_the_ontology_label() -> None:
    row = published_rows([YOUNG_RESEARCHER])[0]
    generated = OntologyLabelGenerator({YOUNG_RESEARCHER: ["open"]}).generate(
        {"code": YOUNG_RESEARCHER, "label": row["label"]},
        taxonomy_context={"expected_parent_path": []},
    )
    assert generated.enrichment["prototype_values"] == ["open", row["label"], row["label"]]
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
        ontology_generator(object(), taxonomy(_FRAME), corpora=missing)


def test_footprint_join_uses_abbrev_catalog_family_and_explicit_code(tmp_path) -> None:
    _fixture(tmp_path)
    rows = labeled_rows_from_corpora(None, corpora=tmp_path)
    by_column = {row.column: row for row in rows}
    assert by_column["doc_ref"].code == PDF_EXPORT
    assert by_column["doc_ref"].sample_values == ["INV-1", "INV-2"]
    assert "amount" not in by_column
    assert "encoding" not in by_column
    assert by_column["surrogate"].code == PARENTING
    assert by_column["org_id"].code == ORGANIZATION
    assert by_column["stamped"].code == YOUNG_RESEARCHER
    assert by_column["stamped"].sample_values == ["open"]
    kept = {row.column for row in labeled_rows_from_corpora(taxonomy(_FRAME), corpora=tmp_path)}
    assert "org_id" not in kept
    assert "doc_ref" in kept
    generator = ontology_generator(object(), taxonomy(_FRAME), corpora=tmp_path)
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
        labeled_rows_from_corpora(taxonomy(_FRAME), corpora=tmp_path)
