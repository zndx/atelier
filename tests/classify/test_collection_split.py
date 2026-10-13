"""Published tables, the answer key, and the genus-level split."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from atelier.classify.collection_split import align_genus_split
from atelier.classify.decisions_walk import genus_term
from atelier.classify.features import extract_features
from atelier.classify.insitu_corpus import InSituError
from atelier.classify.taxonomy import load_annotations_from_filesystem

_VOCAB = """code,label,abbrev,parent_code,taxonomy,description
SDG.PROCESS,Process,PROCESS,,sdg,process
SDG.PROCESS.EXPORT,Export,EXPORT,SDG.PROCESS,sdg,export
SDG.PROCESS.EXPORT.PDF,Pdf,PDF,SDG.PROCESS.EXPORT,sdg,pdf
SDG.GENERIC,Generic,GENERIC,,sdg,generic
"""

_KEY_HEADER = (
    "collection,table,column,ordinal,class_iri,property_iri,range_iri,"
    "genus,reference_code,ref_basis\n"
)


def _checkout(root: Path) -> None:
    vocab = root / "vocabulary"
    vocab.mkdir(parents=True)
    (vocab / "annotations.csv").write_text(_VOCAB, encoding="utf-8")
    key_dir = root / "corpus" / "tagging"
    key_dir.mkdir(parents=True)
    (key_dir / "answer_key.csv").write_text(
        _KEY_HEADER
        + "alpha,jobs,pdf_id,0,sdg:PdfExport,sdg:pdfId,xsd:string,bfo:0000015,"
        "SDG.PROCESS.EXPORT.PDF,leaf\n"
        + "alpha,jobs,created_at,1,sdg:PdfExport,,,bfo:0000015,,audit-unclaimed\n"
        + "alpha,batches,batch_id,0,sdg:Export,sdg:batchId,xsd:string,bfo:0000015,"
        "SDG.PROCESS.EXPORT,entity-leaf\n"
        + "beta,runs,run_id,0,sdg:Process,sdg:runId,xsd:string,bfo:0000015,"
        "SDG.PROCESS,genus-anchor\n"
        + "gamma,docs,doc_id,0,sdg:PdfExport,sdg:docId,xsd:string,bfo:0000015,"
        "SDG.PROCESS.EXPORT.PDF,leaf\n",
        encoding="utf-8",
    )
    tables = {
        ("alpha", "jobs"): "pdf_id,created_at\n1,2020-01-01\n",
        ("alpha", "batches"): "batch_id\n9\n",
        ("beta", "runs"): "run_id\n3\n",
        ("gamma", "docs"): "doc_id\n4\n",
    }
    for (slug, table), body in tables.items():
        directory = root / "corpus" / "collections" / slug / "tables"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{table}.csv").write_text(body, encoding="utf-8")
        (directory / f"{table}.sql").write_text(
            f"CREATE TABLE {table} (\n  {table}_unused INTEGER\n);\n",
            encoding="utf-8",
        )
    jobs_sql = root / "corpus" / "collections" / "alpha" / "tables" / "jobs.sql"
    jobs_sql.write_text(
        "CREATE TABLE jobs (\n  pdf_id INTEGER,\n  created_at TIMESTAMP\n);\n",
        encoding="utf-8",
    )


def test_leaf_codes_gather_to_their_genus_and_splits_stay_with_the_collection(tmp_path) -> None:
    _checkout(tmp_path)
    split = align_genus_split(tmp_path)
    by_identity = {
        (column.collection, column.table, column.column): column
        for column in split.columns
    }
    leaf = by_identity[("alpha", "jobs", "pdf_id")]
    assert leaf.genus_code == "SDG.PROCESS.EXPORT"
    assert leaf.reference_code == "SDG.PROCESS.EXPORT.PDF"
    assert leaf.class_iri == "sdg:PdfExport"
    assert leaf.column_type == "INTEGER"
    assert leaf.table_columns == ("pdf_id", "created_at")
    audit = by_identity[("alpha", "jobs", "created_at")]
    assert audit.reference_code == ""
    assert audit.genus_code == ""
    assert audit.class_iri == "sdg:PdfExport"
    assert audit.split == leaf.split
    assert by_identity[("alpha", "batches", "batch_id")].split == leaf.split
    assert by_identity[("alpha", "batches", "batch_id")].genus_code == "SDG.PROCESS.EXPORT"
    sides = split.collections()
    assert sides["alpha"] == "train"
    assert sides["gamma"] == "test"
    assert sides["beta"] == "train"
    assert "SDG.GENERIC" not in split.genus_codes()
    features = extract_features(
        column_name=leaf.column,
        column_type=leaf.column_type,
        values=["1"],
        siblings=list(leaf.table_columns),
        source_table=leaf.table,
        total_count=1,
    )
    text = features.to_embedding_text()
    assert "pdf id" in text
    assert "created at" in text
    assert "integer" in text
    assert leaf.genus_code not in text


def test_a_column_missing_from_the_key_is_refused(tmp_path) -> None:
    _checkout(tmp_path)
    extra = tmp_path / "corpus" / "collections" / "alpha" / "tables" / "jobs.csv"
    extra.write_text("pdf_id,created_at,note\n1,2020-01-01,x\n", encoding="utf-8")
    with pytest.raises(InSituError, match="not in the answer key"):
        align_genus_split(tmp_path)


def test_a_code_outside_the_vocabulary_is_refused(tmp_path) -> None:
    _checkout(tmp_path)
    path = tmp_path / "corpus" / "tagging" / "answer_key.csv"
    text = path.read_text(encoding="utf-8").replace(
        "SDG.PROCESS,genus-anchor", "SDG.MADE_UP,genus-anchor",
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(InSituError, match="not in the sdg-corpora vocabulary"):
        align_genus_split(tmp_path)


def test_published_collections_match_the_answer_key_at_genus_level() -> None:
    root = Path("external/sdg-corpora")
    key_path = root / "corpus" / "tagging" / "answer_key.csv"
    if not key_path.is_file():
        pytest.skip("sdg-corpora tagging answer key is not in this checkout")
    cats = load_annotations_from_filesystem(
        root / "vocabulary" / "annotations.csv", hierarchical=True,
    )
    split = align_genus_split(root, category_set=cats)
    with key_path.open(newline="", encoding="utf-8") as handle:
        key_rows = list(csv.DictReader(handle))
    assert len(split.columns) == len(key_rows) == 26446
    identities = {
        (column.collection, column.table, column.column) for column in split.columns
    }
    assert identities == {
        (row["collection"], row["table"], row["column"]) for row in key_rows
    }
    labeled = split.labeled_columns()
    assert len(labeled) == 26446
    assert len(split.columns) - len(labeled) == 0
    assert "SDG.GENERIC" not in split.genus_codes()
    assert all(column.reference_code != "SDG.GENERIC" for column in split.columns)
    for column in split.columns:
        assert column.class_iri
        if column.reference_code:
            assert column.genus_code == genus_term(cats, column.reference_code)
            assert column.genus_code in cats.all_by_code
        else:
            assert column.genus_code == ""
    sides = split.collections()
    assert len(sides) == 453
    counts = {side: 0 for side in ("train", "test", "validate")}
    for side in sides.values():
        counts[side] += 1
    assert counts == {"train": 363, "test": 45, "validate": 45}
    status = next(
        column for column in split.columns
        if column.collection == "computational-material-record-044e3d9a"
        and column.table == "computational_material_records"
        and column.column == "status"
    )
    assert status.class_iri == "sdg:ComputationalMaterialRecord"
    assert status.property_iri == "sdg:status"
    assert status.range_iri == "xsd:string"
    assert status.reference_code == "SDG.ARTIFACT"
    assert status.genus_code == genus_term(cats, "SDG.ARTIFACT")
    material = next(
        column for column in split.columns
        if column.collection == "computational-material-record-044e3d9a"
        and column.table == "computational_material_records"
        and column.column == "material_system_id"
    )
    assert material.genus_iri == "sdg:System"
    assert material.reference_code == "SDG.ARTIFACT"
    assert material.ref_basis == "genus-walk"
    assert material.genus_code == genus_term(cats, "SDG.ARTIFACT")
    topic = next(
        column for column in split.columns
        if column.collection == "decision-assessment-36edfccb"
        and column.table == "best_interest_determinations"
        and column.column == "topic_id"
    )
    assert topic.genus_iri == "bfo:0000003"
    assert topic.reference_code == "SDG.ICE.DESCRIPTIVE.DECISION_SUBJECT"
    assert topic.ref_basis == "kind-authored"
    assert topic.genus_code == "SDG.ICE.DESCRIPTIVE.DECISION_SUBJECT"
    operator = next(
        column for column in split.columns
        if column.collection == "bus-route-e738d90a"
        and column.table == "bus_routes"
        and column.column == "bus_operator_license_number"
    )
    assert operator.genus_iri == "bfo:00000995"
    assert operator.reference_code == "SDG.MATERIAL_ENTITY.AGENT_ORGANIZATION"
    assert operator.ref_basis == "kind-authored"
    assert operator.genus_code == "SDG.MATERIAL_ENTITY.AGENT_ORGANIZATION"
    rate = next(
        column for column in split.columns
        if column.collection == "drug-application-0614fd04"
        and column.table == "geographic_regions"
        and column.column == "prevalence_rate"
    )
    assert rate.reference_code == "SDG.GENERIC.PREVALENCE_RATE"
    assert rate.genus_code == "SDG.GENERIC.PREVALENCE_RATE"
    engagement = next(
        column for column in split.columns
        if column.table == "clie_client_engagements" and column.column == "engagement_id"
    )
    assert engagement.class_iri == "sdg:ClientEngagement"
    assert engagement.genus_code == genus_term(cats, engagement.reference_code)
    assert engagement.genus_code != engagement.reference_code
