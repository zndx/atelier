"""Ontology-labeled NHSVM rows and MaxSim prototypes from an Aegir release."""

from __future__ import annotations

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atelier.classify.ontology_corpus import (
    OntologyCorpusError,
    OntologyLabelGenerator,
    exemplars_by_code,
    labeled_rows_from_release,
    ontology_generator,
    rows_from_labeled_columns,
)
from atelier.classify.sampler import ColumnSample, TableSample
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory
from atelier.enrichment.verifiers import run_verifier_suite


def _taxonomy() -> HierarchicalCategorySet:
    cats = [
        ReferenceCategory(code="SDG.A.1", label="invoice identifier", embedding_text="invoice", parent_code="SDG.A"),
        ReferenceCategory(code="SDG.B", label="status", embedding_text="status", parent_code="SDG"),
    ]
    return HierarchicalCategorySet(name="sdg", categories=cats)


def _samples() -> list[TableSample]:
    return [
        TableSample(name="billing", columns=[
            ColumnSample(
                name="invoice_id", table_name="billing",
                values=["INV-1"], all_values=["INV-1", "INV-2"],
                reference_code="SDG.A.1",
            ),
            ColumnSample(
                name="amount", table_name="billing",
                values=["10"], all_values=["10", "20"],
                reference_code="SDG.NOT_IN_TAX",
            ),
            ColumnSample(
                name="note", table_name="billing",
                values=[], reference_code=None,
            ),
        ]),
    ]


def test_labeled_rows_keep_reference_codes_in_the_taxonomy() -> None:
    rows = rows_from_labeled_columns(_samples(), _taxonomy())
    assert len(rows) == 1
    assert rows[0].code == "SDG.A.1"
    assert rows[0].column == "invoice_id"
    assert rows[0].sample_values[:2] == ["INV-1", "INV-2"]
    assert rows[0].siblings_full == ["amount", "note"]


def test_generator_prototypes_are_column_values() -> None:
    exemplars = exemplars_by_code(_samples(), _taxonomy())
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


def test_short_exemplar_list_pads_with_the_ontology_label() -> None:
    generated = OntologyLabelGenerator({"SDG.B": ["open"]}).generate(
        {"code": "SDG.B", "label": "status"},
        taxonomy_context={"expected_parent_path": []},
    )
    assert generated.enrichment["prototype_values"] == ["open", "status", "status"]
    assert generated.enrichment["parent_path"] == []


def test_empty_release_dir_keeps_the_llm_generator_path() -> None:
    assert ontology_generator(SimpleNamespace(aegir_release_dir=""), _taxonomy()) is None


def test_release_parquet_round_trip_drops_unknown_codes(tmp_path) -> None:
    release = tmp_path / "rel"
    release.mkdir()
    pq.write_table(pa.table({
        "table_id": ["billing", "billing"],
        "column_id": ["c1", "c2"],
        "column_name": ["invoice_id", "amount"],
        "n_rows": [2, 2],
        "sample_values": pa.array(
            [["INV-1", "INV-2"], ["10", "20"]],
            type=pa.list_(pa.string()),
        ),
    }), release / "corpus_columns.parquet")
    pq.write_table(pa.table({
        "table_id": ["billing", "billing"],
        "column_id": ["c1", "c2"],
        "reference_code": ["SDG.A.1", "SDG.NOT_IN_TAX"],
        "template_id": ["t", "t"],
    }), release / "reference.parquet")
    rows = labeled_rows_from_release(str(release), _taxonomy())
    assert [r.code for r in rows] == ["SDG.A.1"]
    generator = ontology_generator(SimpleNamespace(aegir_release_dir=str(release)), _taxonomy())
    assert generator is not None
    assert generator.name == "ontology:aegir-release"


def test_release_with_no_taxonomy_overlap_fails_closed(tmp_path) -> None:
    release = tmp_path / "rel"
    release.mkdir()
    pq.write_table(pa.table({
        "table_id": ["billing"],
        "column_id": ["c1"],
        "column_name": ["amount"],
        "n_rows": [1],
        "sample_values": pa.array([["10"]], type=pa.list_(pa.string())),
    }), release / "corpus_columns.parquet")
    pq.write_table(pa.table({
        "table_id": ["billing"],
        "column_id": ["c1"],
        "reference_code": ["SDG.NOT_IN_TAX"],
        "template_id": ["t"],
    }), release / "reference.parquet")
    with pytest.raises(OntologyCorpusError):
        ontology_generator(SimpleNamespace(aegir_release_dir=str(release)), _taxonomy())
