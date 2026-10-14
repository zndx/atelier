"""In-situ split, amplification, and synthetic NHSVM rows."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atelier.classify.decisions_walk import (
    associate_genera,
    genus_neighborhood,
    genus_term,
    neighborhood_term,
    structural_genera,
)
from atelier.classify.extension_patterns import code_iri
from atelier.classify.features import FEATURE_NAMES, extract_features
from atelier.classify.factorized_nhsvm import NHSVMHeadAdapter
from atelier.classify.insitu_corpus import (
    InSituError,
    InSituPrototypeGenerator,
    conflicting_known_entities,
    distribution_deficit,
    load_collection_answer_key,
    load_collection_cells,
    prepare_collection_cta,
    prepare_insitu,
    split_by_code,
    split_by_collection,
    split_projected,
    support_counts,
)
from atelier.optimize.precondition import PreconditionError, build_nhsvm_head
from atelier.optimize.svm.reflect import (
    NHSVM_FEATURE_TEXT,
    Row,
    build_texts_and_labels,
)
from atelier.classify.taxonomy import HierarchicalCategorySet
from atelier.enrichment.verifiers import run_verifier_suite
from atelier.flows.resident import load_sweep_samples
from tests.classify.explicit_sdg import (
    ANNOTATIONS,
    EXPLICIT,
    PARENTING,
    PDF_EXPORT,
    PROCESS,
    ROLE,
    YOUNG_RESEARCHER,
    published_rows,
    relational_procedure,
    taxonomy,
)

_LEAVES = (PDF_EXPORT, PARENTING, YOUNG_RESEARCHER)
_FRAME = (PROCESS, ROLE, *_LEAVES)


def _assignments() -> dict[tuple[str, str], str]:
    return {
        ("billing", "doc_ref"): PDF_EXPORT,
        ("other", "doc_ref"): PDF_EXPORT,
        ("billing", "note"): PARENTING,
        ("lone", "id"): YOUNG_RESEARCHER,
    }


def _generators() -> dict:
    return {
        code_iri(PDF_EXPORT): relational_procedure(
            code_iri(PDF_EXPORT), "export_batch", "batch",
        ),
        code_iri(PARENTING): relational_procedure(
            code_iri(PARENTING), "caregiver_role", "kin",
        ),
        code_iri(YOUNG_RESEARCHER): relational_procedure(
            code_iri(YOUNG_RESEARCHER), "study_role", "cohort",
        ),
    }


def _genus_generators() -> dict:
    gens = _generators()
    gens[code_iri(PROCESS)] = relational_procedure(
        code_iri(PROCESS), "activity", "proc",
    )
    gens[code_iri(ROLE)] = relational_procedure(
        code_iri(ROLE), "bearer", "role",
    )
    return gens


def _footprint(root: Path) -> None:
    pin = root / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.table({
        "table_name": ["billing", "billing", "billing", "other", "other", "lone"],
        "col_name": ["amount", "doc_ref", "note", "b", "doc_ref", "id"],
        "value": ["10", "INV-1", "hello", "x", "INV-9", "1"],
        "row_ix": [0, 0, 0, 0, 0, 0],
    }), pin / "base_rows.parquet")


def test_explicit_slice_is_the_published_vocabulary() -> None:
    rows = published_rows(EXPLICIT)
    assert [row["code"] for row in rows] == list(EXPLICIT)
    for row in rows:
        fragment = row["code"][4:].replace(".", "_")
        assert code_iri(row["code"]) == f"https://signals.zndx.org/sdg#{fragment}"
    assert code_iri("SDG.A.1") is None


def test_split_puts_wide_tables_on_both_sides() -> None:
    support, holdout = split_projected({
        "billing": ["amount", "doc_ref", "note"],
        "other": ["b", "doc_ref"],
        "lone": ["id"],
    })
    assert set(support).isdisjoint(holdout)
    assert len(support) + len(holdout) == 6
    billing_sides = {side for side, cols in (("support", support), ("holdout", holdout))
                     for table, _col in cols if table == "billing"}
    assert billing_sides == {"support", "holdout"}
    assert ("billing", "doc_ref") in support
    assert ("other", "doc_ref") in support


def test_support_census_uses_the_assignment() -> None:
    support, holdout = split_projected({
        "billing": ["amount", "doc_ref", "note"],
        "other": ["b", "doc_ref"],
        "lone": ["id"],
    })
    assignments = _assignments()
    counts = support_counts(support, assignments)
    assert counts[PDF_EXPORT] == 2
    assert counts.get(PARENTING, 0) == 0
    assert ("billing", "note") in holdout
    recount = support_counts(holdout, assignments)
    assert recount.get(PDF_EXPORT, 0) == 0
    assert recount[PARENTING] == 1


def test_a_miss_on_a_known_entity_is_a_conflict() -> None:
    assigned = {("billing", "doc_ref"): PDF_EXPORT, ("billing", "note"): PARENTING}
    assert conflicting_known_entities(
        assigned,
        {("billing", "doc_ref"): PDF_EXPORT, ("billing", "note"): PARENTING},
    ) == []
    misses = conflicting_known_entities(
        assigned, {("billing", "doc_ref"): PARENTING},
    )
    assert [row["column"] for row in misses] == ["doc_ref", "note"]
    assert misses[0]["predicted"] == PARENTING
    assert misses[1]["predicted"] == ""


def test_collection_holdout_keeps_related_tables_together(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    pin = corpora / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.table({
        "table_name": [
            "officers", "officers", "incidents", "incidents",
            "visits", "visits", "patients", "patients",
        ],
        "col_name": [
            "badge", "unit", "severity", "officer_ref",
            "reason", "clinic_ref", "species", "visit_ref",
        ],
        "value": [
            "A-14", "north", "bite", "A-14",
            "checkup", "c1", "canine", "v1",
        ],
        "row_ix": [0, 0, 0, 0, 0, 0, 0, 0],
    }), pin / "base_rows.parquet")
    assignments = {
        ("officers", "badge"): PDF_EXPORT,
        ("incidents", "severity"): PARENTING,
        ("visits", "reason"): YOUNG_RESEARCHER,
        ("patients", "species"): PDF_EXPORT,
    }
    collections = {
        "officers": "patrol",
        "incidents": "patrol",
        "visits": "clinic",
        "patients": "clinic",
    }
    plan = prepare_insitu(
        taxonomy(_FRAME), tmp_path / "build", corpora=corpora,
        assignments=assignments, extra_generators=_generators(),
        collections=collections, holdout_collections={"clinic"},
        base_variants=1, amplified_variants=2, rows_per_column=2, floor=2,
    )
    support_tables = {table for table, _column in plan.support}
    holdout_tables = {table for table, _column in plan.holdout}
    assert support_tables == {"officers", "incidents"}
    assert holdout_tables == {"visits", "patients"}
    assert ("visits", "clinic_ref") in plan.holdout
    assert ("officers", "unit") in plan.support
    for row in plan.rows:
        if row.table.startswith("syn_"):
            continue
        assert row.table in support_tables
        assert set(row.siblings_full).isdisjoint({"reason", "clinic_ref", "species", "visit_ref"})
    split = json.loads((tmp_path / "build" / "relational_split" / "split.json").read_text())
    assert split["holdout_collections"] == ["clinic"]
    visits = (tmp_path / "build" / "relational_split" / "holdout" / "tables" / "visits.csv").read_text()
    assert visits.splitlines()[0] == "clinic_ref,reason"
    assert "checkup" in visits
    texts, labels = build_texts_and_labels([
        row for row in plan.rows if row.table == "officers"
    ])
    assert labels == [PDF_EXPORT]
    assert "A-14" in texts[0]
    assert "unit" in texts[0]
    assert PDF_EXPORT not in texts[0]
    try:
        split_by_collection(
            {"visits": ["reason"], "patients": ["species"]},
            collections, set(),
        )
    except InSituError as exc:
        assert "at least one collection" in str(exc)
    else:
        raise AssertionError("empty holdout must fail")


def test_nhsvm_text_ablates_pipeline_features() -> None:
    row = Row(
        table="officers", column="badge", column_type="object",
        sample_values=["A-14", "B-2"], siblings_full=["unit", "shift"],
        mnemonic="", code=PDF_EXPORT,
    )
    text, labels = build_texts_and_labels([row])
    assert labels == [PDF_EXPORT]
    assert "A-14" in text[0]
    assert "unit" in text[0]
    features = extract_features(
        column_name=row.column, values=row.sample_values,
        siblings=[row.column, *row.siblings_full], source_table=row.table,
        total_count=2,
    )
    drop_values = {name: name != "sample_values" for name in FEATURE_NAMES}
    drop_siblings = {name: name != "sibling_context" for name in FEATURE_NAMES}
    without_values = features.to_embedding_text(drop_values)
    without_siblings = features.to_embedding_text(drop_siblings)
    assert "A-14" not in without_values
    assert "unit" in without_values
    assert "unit" not in without_siblings
    assert "A-14" in without_siblings


def _write_collection(root: Path, slug: str, tables: dict[str, tuple[str, str, str]]) -> None:
    """tables: name → (csv text, sql text, manifest column list is the csv header)."""
    base = root / "corpus" / "collections" / slug
    table_dir = base / "tables"
    table_dir.mkdir(parents=True)
    manifest_tables = []
    for name, (csv_text, sql_text, _columns) in tables.items():
        (table_dir / f"{name}.csv").write_text(csv_text, encoding="utf-8")
        (table_dir / f"{name}.sql").write_text(sql_text, encoding="utf-8")
        header = csv_text.splitlines()[0].split(",")
        manifest_tables.append({"name": name, "columns": header})
    (base / "manifest.json").write_text(
        json.dumps({"slug": slug, "tables": manifest_tables}),
        encoding="utf-8",
    )


def test_collection_directories_hold_out_whole_bundles(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    _write_collection(corpora, "patrol", {
        "officers": (
            "badge,unit,years\nA-14,north,3\nB-2,south,14\n",
            "CREATE TABLE officers (\n  badge VARCHAR(8),\n  unit VARCHAR(16),\n"
            "  years INTEGER,\n  PRIMARY KEY (badge)\n);\n",
            "",
        ),
        "incidents": (
            "severity,officer_ref\nbite,A-14\n",
            "CREATE TABLE incidents (\n  severity VARCHAR(16),\n"
            "  officer_ref VARCHAR(8)\n);\n",
            "",
        ),
    })
    _write_collection(corpora, "clinic", {
        "visits": (
            "reason,clinic_ref\ncheckup,c1\n",
            "CREATE TABLE visits (\n  reason VARCHAR(16),\n  clinic_ref VARCHAR(8)\n);\n",
            "",
        ),
        "patients": (
            "species,visit_ref\ncanine,v1\n",
            "CREATE TABLE patients (\n  species VARCHAR(16),\n  visit_ref VARCHAR(8)\n);\n",
            "",
        ),
    })
    assignments = {
        ("patrol/officers", "badge"): PDF_EXPORT,
        ("patrol/officers", "years"): PARENTING,
        ("patrol/incidents", "severity"): YOUNG_RESEARCHER,
        ("clinic/visits", "reason"): YOUNG_RESEARCHER,
        ("clinic/patients", "species"): PDF_EXPORT,
    }
    genera = _genus_generators()
    plan = prepare_collection_cta(
        taxonomy(_FRAME), tmp_path / "build", corpora=corpora,
        assignments=assignments, holdout_collections={"clinic"},
        extra_generators=genera,
        base_variants=1, amplified_variants=2, rows_per_column=2, floor=2,
    )
    assert {table for table, _column in plan.support} == {
        "patrol/officers", "patrol/incidents",
    }
    assert {table for table, _column in plan.holdout} == {
        "clinic/visits", "clinic/patients",
    }
    real = [row for row in plan.rows if not row.table.startswith("syn_")]
    assert {(row.table, row.column) for row in real} == {
        ("patrol/officers", "badge"),
        ("patrol/officers", "years"),
        ("patrol/incidents", "severity"),
    }
    assert {row.code for row in real} == {PROCESS, ROLE}
    assert plan.rows == real
    assert plan.holdout_rows == []
    assert plan.exemplars == {}
    augmented = list(
        (tmp_path / "build" / "relational_split" / "holdout" / "tables").glob("holdout_*.csv")
    )
    assert augmented == []
    years = next(row for row in real if row.column == "years")
    assert years.column_type == "INTEGER"
    assert "unit" in years.siblings_full
    assert set(years.siblings_full).isdisjoint({"reason", "clinic_ref", "species", "visit_ref"})
    texts, labels = build_texts_and_labels([years])
    assert labels == [ROLE]
    assert years.code == ROLE
    assert "integer" in texts[0]
    assert "3" in texts[0]
    assert "unit" in texts[0]
    assert PARENTING not in texts[0]
    visits = (
        tmp_path / "build" / "relational_split" / "holdout" / "tables"
        / "clinic__visits.csv"
    ).read_text()
    assert visits.splitlines()[0] == "reason,clinic_ref"
    assert YOUNG_RESEARCHER not in visits
    bare = prepare_collection_cta(
        taxonomy(_FRAME), tmp_path / "bare", corpora=corpora,
        assignments={}, holdout_collections={"clinic"},
    )
    assert bare.rows == []
    assert bare.holdout_collections == ["clinic"]
    partial = _genus_generators()
    partial.pop(code_iri(ROLE))
    withheld = prepare_collection_cta(
        taxonomy(_FRAME), tmp_path / "partial", corpora=corpora,
        assignments=assignments, holdout_collections={"clinic"},
        extra_generators=partial, rows_per_column=1,
    )
    assert withheld.rows
    assert withheld.holdout_rows == []
    assert withheld.exemplars == {}


def test_each_column_is_associated_with_its_genus() -> None:
    cats = taxonomy(_FRAME)
    assigned = {
        ("officers", "badge"): PDF_EXPORT,
        ("officers", "years"): PARENTING,
        ("officers", "duty"): YOUNG_RESEARCHER,
    }
    assert associate_genera(assigned, cats) == {
        ("officers", "badge"): PROCESS,
        ("officers", "years"): ROLE,
        ("officers", "duty"): ROLE,
    }
    assert genus_term(cats, PROCESS) == PROCESS
    assert genus_term(cats, ROLE) == ROLE
    descriptive = "SDG.ICE.DESCRIPTIVE"
    benefit = "SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION"
    deeper = taxonomy((
        "SDG.GDC", "SDG.ICE", descriptive, benefit,
    ))
    assert genus_term(deeper, benefit) == descriptive
    assert genus_term(deeper, descriptive) == descriptive
    assert structural_genera(deeper) == ["SDG.ICE", descriptive]
    assert set(genus_neighborhood(deeper, 1)) == {
        "SDG.GDC", "SDG.ICE", descriptive, benefit,
    }
    assert neighborhood_term(deeper, benefit) == benefit
    assert neighborhood_term(deeper, "SDG.GDC") == "SDG.GDC"
    assert associate_genera({("plans", "option"): benefit}, deeper) == {
        ("plans", "option"): benefit,
    }


def test_published_neighborhood_keeps_the_child_and_admits_siblings() -> None:
    import csv

    codes = [row["code"] for row in csv.DictReader(ANNOTATIONS.open(encoding="utf-8"))]
    cats = taxonomy(codes)
    genera = structural_genera(cats)
    first = genus_neighborhood(cats, 1)
    second = genus_neighborhood(cats, 2)
    assert len(genera) == 109
    assert len(first) == 748
    assert len(second) == 1005
    assert set(genera) < set(first) < set(second) < set(codes)
    frame = set(first)
    for genus in genera:
        assert cats.parent[genus] in frame
        for kid in cats.children[genus]:
            assert kid in frame
    for code in second:
        parent = cats.parent.get(code)
        if parent:
            assert parent in set(second)

    def depth(code: str) -> int:
        seen = 0
        current = cats.parent.get(code)
        while current:
            seen += 1
            current = cats.parent.get(current)
        return seen

    assert max(depth(code) for code in first) == 4
    assert max(depth(code) for code in second) == 4
    benefit = "SDG.ICE.DESCRIPTIVE.RETIREMENT_BENEFIT_OPTION"
    descriptive = "SDG.ICE.DESCRIPTIVE"
    condiment = "SDG.INDEPENDENT_CONTINUANT.CONDIMENT_STATION"
    continuant = "SDG.INDEPENDENT_CONTINUANT"
    assert neighborhood_term(cats, benefit) == benefit
    assert genus_term(cats, benefit) == descriptive
    rate = "SDG.GENERIC.PREVALENCE_RATE"
    assert genus_term(cats, rate) == rate
    assert genus_term(cats, "SDG.GENERIC") is None
    assert neighborhood_term(cats, descriptive) == descriptive
    assert neighborhood_term(cats, "SDG.ICE") == "SDG.ICE"
    assert neighborhood_term(cats, "SDG.GDC") == "SDG.GDC"
    assert condiment not in frame
    assert neighborhood_term(cats, condiment) == continuant
    assert neighborhood_term(cats, condiment, order=2) == condiment
    assert neighborhood_term(cats, PDF_EXPORT) == PROCESS
    assert neighborhood_term(cats, PDF_EXPORT, order=2) == PDF_EXPORT
    try:
        genus_neighborhood(cats, 0)
    except ValueError as exc:
        assert "order" in str(exc)
    else:
        raise AssertionError("order 0 must fail")


def test_collection_cta_labels_the_genus_neighborhood(tmp_path) -> None:
    continuant = "SDG.INDEPENDENT_CONTINUANT"
    artifact = "SDG.ARTIFACT"
    dataset = "SDG.DOM.DATASET_ENTITY"
    condiment = "SDG.INDEPENDENT_CONTINUANT.CONDIMENT_STATION"
    cats = taxonomy((continuant, artifact, dataset, condiment))
    corpora = tmp_path / "corpora"
    _write_collection(corpora, "patrol", {
        "officers": (
            "badge\nA-14\n",
            "CREATE TABLE officers (\n  badge VARCHAR(8)\n);\n",
            "",
        ),
    })
    _write_collection(corpora, "clinic", {
        "visits": (
            "reason\ncheckup\n",
            "CREATE TABLE visits (\n  reason VARCHAR(16)\n);\n",
            "",
        ),
    })
    child = {
        ("patrol/officers", "badge"): dataset,
        ("clinic/visits", "reason"): dataset,
    }
    plan = prepare_collection_cta(
        cats, tmp_path / "child", corpora=corpora,
        assignments=child, holdout_collections={"clinic"},
    )
    real = [row for row in plan.rows if not row.table.startswith("syn_")]
    assert [row.code for row in real] == [dataset]
    assert plan.counts == {dataset: 1}
    assert plan.variants == {}
    assert plan.holdout_rows == []
    texts, labels = build_texts_and_labels(real)
    assert labels == [dataset]
    assert dataset not in texts[0]
    assert "A-14" in texts[0]
    sibling = {
        ("patrol/officers", "badge"): condiment,
        ("clinic/visits", "reason"): condiment,
    }
    lifted = prepare_collection_cta(
        cats, tmp_path / "lifted", corpora=corpora,
        assignments=sibling, holdout_collections={"clinic"},
    )
    assert [row.code for row in lifted.rows if not row.table.startswith("syn_")] == [
        continuant,
    ]
    kept = prepare_collection_cta(
        cats, tmp_path / "kept", corpora=corpora,
        assignments=sibling, holdout_collections={"clinic"},
        order=2,
    )
    assert [row.code for row in kept.rows if not row.table.startswith("syn_")] == [
        condiment,
    ]


def test_distribution_deficit_raises_each_side_to_the_pooled_share() -> None:
    support = {PDF_EXPORT: 8, PARENTING: 2}
    holdout = {PDF_EXPORT: 1, PARENTING: 9}
    assert distribution_deficit(support, holdout) == {PARENTING: 4}
    assert distribution_deficit(holdout, support) == {PDF_EXPORT: 4}
    assert distribution_deficit({}, {PDF_EXPORT: 3, PARENTING: 1}) == {
        PARENTING: 1,
        PDF_EXPORT: 3,
    }
    assert distribution_deficit(
        {PDF_EXPORT: 2, PARENTING: 2},
        {PDF_EXPORT: 2, PARENTING: 2},
    ) == {}


def test_published_collections_are_unlabeled_until_the_answer_key(tmp_path) -> None:
    slugs = {
        "animal-control-officer-1da7ef2d",
        "academic-appointment-0bf5440d",
    }
    cells, collections = load_collection_cells(slugs=slugs)
    assert {collections[cell["table"]] for cell in cells} == slugs
    typed = [
        cell for cell in cells
        if cell["column"] == "animal_control_officer_id"
        and str(cell["column_type"]).upper() == "INTEGER"
    ]
    assert typed
    plan = prepare_collection_cta(
        taxonomy(_FRAME), tmp_path / "build",
        assignments={}, holdout_collections={"animal-control-officer-1da7ef2d"},
        slugs=slugs,
    )
    assert plan.rows == []
    assert {collections[table] for table, _column in plan.holdout} == {
        "animal-control-officer-1da7ef2d",
    }
    assert {collections[table] for table, _column in plan.support} == {
        "academic-appointment-0bf5440d",
    }
    officer_tables = {
        table for table, _column in plan.holdout if table.startswith("animal-control-officer-1da7ef2d/")
    }
    assert "animal-control-officer-1da7ef2d/animal_control_officers" in officer_tables
    assert "animal-control-officer-1da7ef2d/incident_reports" in officer_tables


def test_answer_key_is_exact_and_collection_sided(tmp_path) -> None:
    assert load_collection_answer_key(tmp_path) == ({}, set())
    corpora = tmp_path / "corpora"
    key = corpora / "corpus" / "cta_answer_key.csv"
    key.parent.mkdir(parents=True)
    key.write_text(
        "collection,table,column,code,split\n"
        f"patrol,officers,badge,{PDF_EXPORT},support\n"
        f"clinic,visits,reason,{YOUNG_RESEARCHER},holdout\n"
        f"clinic,visits,clinic_ref,{PARENTING},support\n",
        encoding="utf-8",
    )
    try:
        load_collection_answer_key(corpora)
    except InSituError as exc:
        assert "both sides" in str(exc)
    else:
        raise AssertionError("a collection on both sides must fail")
    key.write_text(
        "collection,table,column,code,split\n"
        f"patrol,officers,badge,{PDF_EXPORT},support\n"
        f"clinic,visits,reason,{YOUNG_RESEARCHER},holdout\n",
        encoding="utf-8",
    )
    assignments, held = load_collection_answer_key(corpora)
    assert assignments == {
        ("patrol/officers", "badge"): PDF_EXPORT,
        ("clinic/visits", "reason"): YOUNG_RESEARCHER,
    }
    assert held == {"clinic"}
    try:
        build_nhsvm_head(
            None, taxonomy(_FRAME), taxonomy_id="projected-holdout", vocab_sig="abc",
        )
    except PreconditionError as exc:
        message = str(exc)
        assert "answer key" in message
        assert "column split" in message
    else:
        raise AssertionError("a missing answer key must refuse the fit")


def test_registered_feature_text_matches_the_pipeline_mask() -> None:
    features = extract_features(
        column_name="badge",
        column_type="INTEGER",
        values=["3", "14"],
        siblings=["badge", "unit", "years"],
        source_table="officers",
        total_count=2,
    )
    adapter = NHSVMHeadAdapter.__new__(NHSVMHeadAdapter)
    adapter.training_metadata = {"feature_text": NHSVM_FEATURE_TEXT}
    text = adapter._build_text(features)
    assert text == features.to_embedding_text()
    assert "integer" in text
    assert "unit" in text
    drop_values = {name: name != "sample_values" for name in FEATURE_NAMES}
    drop_siblings = {name: name != "sibling_context" for name in FEATURE_NAMES}
    assert "3" not in features.to_embedding_text(drop_values)
    assert "unit" not in features.to_embedding_text(drop_siblings)
    adapter.training_metadata = {}
    legacy = adapter._build_text(features)
    assert "cardinality=" not in legacy
    assert "cardinality=" in text


def test_split_by_code_keeps_a_support_row_for_every_holdout_code() -> None:
    support, holdout = split_by_code(
        {"t": ["a", "b", "c", "d"], "s": ["x"]},
        {
            ("t", "a"): PDF_EXPORT,
            ("t", "b"): PDF_EXPORT,
            ("t", "c"): PDF_EXPORT,
            ("t", "d"): PARENTING,
        },
        floor=2,
    )
    assert set(support).isdisjoint(holdout)
    assert ("t", "a") in support and ("t", "b") in support
    assert ("t", "c") in holdout
    assert ("t", "d") in support and ("t", "d") not in holdout
    assert ("s", "x") in support
    holdout_codes = {PDF_EXPORT}
    support_codes = {PDF_EXPORT, PARENTING}
    assert holdout_codes <= support_codes


def test_prepare_amplifies_thin_codes_and_writes_a_blind_holdout(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    artifacts = tmp_path / "build"
    _footprint(corpora)
    plan = prepare_insitu(
        taxonomy(_FRAME), artifacts, corpora=corpora,
        assignments=_assignments(), extra_generators=_generators(),
        base_variants=2, amplified_variants=5, rows_per_column=3, floor=2,
    )
    assert set(plan.variants) == set(_LEAVES)
    assert PROCESS not in plan.variants
    assert ROLE not in plan.variants
    assert plan.counts[PDF_EXPORT] == 1
    assert plan.counts[PARENTING] == 1
    assert plan.counts[YOUNG_RESEARCHER] == 1
    assert plan.variants == {PDF_EXPORT: 5, PARENTING: 5, YOUNG_RESEARCHER: 5}
    support_rows = list(plan.rows)
    assert plan.holdout_rows == []
    assert plan.exemplars == {}
    assert {row.code for row in support_rows} == set(_LEAVES)
    assert {(row.table, row.column) for row in support_rows} == {
        ("billing", "doc_ref"), ("billing", "note"), ("lone", "id"),
    }
    holdout_keys = set(plan.holdout)
    assert holdout_keys == {("other", "doc_ref")}
    for row in support_rows:
        assert (row.table, row.column) not in holdout_keys
        assert all((row.table, name) not in holdout_keys for name in row.siblings_full)
    split = json.loads((artifacts / "relational_split" / "split.json").read_text())
    assert set(split["under_represented"]) == set(_LEAVES)
    assert "slot_ref" not in json.dumps(split)
    tables = artifacts / "relational_split" / "holdout" / "tables"
    assert not (tables / "billing.csv").exists()
    text = (tables / "other.csv").read_text()
    assert text.splitlines()[0] == "doc_ref"
    assert "INV-9" in text
    assert "INV-1" not in text
    source = Path(prepare_insitu.__code__.co_filename).read_text().lower()
    assert "naming_map" not in source
    assert "slot_ref" not in source


def test_prototypes_are_synthetic_and_pass_the_verifier(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    _footprint(corpora)
    plan = prepare_insitu(
        taxonomy(_FRAME), tmp_path / "build", corpora=corpora,
        assignments=_assignments(), extra_generators=_generators(),
        base_variants=1, amplified_variants=2, rows_per_column=2,
    )
    label = published_rows([PARENTING])[0]["label"]
    generated = InSituPrototypeGenerator(plan).generate(
        {"code": PARENTING, "label": label, "mnemonic": "parenting_role"},
        taxonomy_context={"expected_parent_path": ["Role"]},
    )
    prototypes = generated.enrichment["prototype_values"]
    assert prototypes == [label, label, label]
    assert not any(str(value).startswith("kin-") for value in prototypes)
    assert "INV-1" not in prototypes
    assert "INV-9" not in prototypes
    report = run_verifier_suite(generated.enrichment, expected_parent_path=["Role"])
    assert report.passed
    assert InSituPrototypeGenerator(plan).name == "ontology:insitu"


def test_missing_corpora_has_no_target_entities(tmp_path) -> None:
    plan = prepare_insitu(
        taxonomy(_FRAME), tmp_path / "build", corpora=tmp_path / "absent",
        assignments=_assignments(), extra_generators=_generators(),
        amplified_variants=3, rows_per_column=2,
    )
    assert plan.support == []
    assert plan.holdout == []
    assert plan.variants == {}
    assert plan.rows == []


def test_projected_holdout_source_loads_only_the_holdout(tmp_path) -> None:
    corpora = tmp_path / "corpora"
    artifacts = tmp_path / "build"
    _footprint(corpora)
    prepare_insitu(
        taxonomy(_FRAME), artifacts, corpora=corpora,
        assignments=_assignments(), extra_generators=_generators(),
        rows_per_column=2,
    )
    from types import SimpleNamespace
    samples = load_sweep_samples(
        SimpleNamespace(artifact_root=str(artifacts), classify_sample_size=10),
        "projected-holdout",
    )
    names = {(table.name, col.name) for table in samples for col in table.columns}
    assert names == {("other", "doc_ref")}
    assert all(col.reference_code in (None, "") for table in samples for col in table.columns)


def test_projected_holdout_loads_the_published_vocabulary() -> None:
    from types import SimpleNamespace

    from atelier.flows.resident import load_category_set

    cats = load_category_set(SimpleNamespace(), "projected-holdout")
    assert isinstance(cats, HierarchicalCategorySet)
    codes = {cat.code for cat in cats.all_categories}
    assert PDF_EXPORT in codes
    assert PARENTING in codes
    assert "SDG.A.1" not in codes
    assert len(codes) > 1000
