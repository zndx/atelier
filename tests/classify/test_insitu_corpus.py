"""In-situ split, amplification, and synthetic NHSVM rows."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from atelier.classify.extension_patterns import code_iri
from atelier.classify.insitu_corpus import (
    InSituPrototypeGenerator,
    conflicting_known_entities,
    prepare_insitu,
    split_by_code,
    split_projected,
    support_counts,
)
from atelier.classify.taxonomy import HierarchicalCategorySet
from atelier.enrichment.verifiers import run_verifier_suite
from atelier.flows.resident import load_sweep_samples
from tests.classify.explicit_sdg import (
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
    synthetic = [row for row in plan.rows if row.table.startswith("syn_")]
    support_rows = [row for row in plan.rows if not row.table.startswith("syn_")]
    assert {row.code for row in support_rows} == set(_LEAVES)
    assert {(row.table, row.column) for row in support_rows} == {
        ("billing", "doc_ref"), ("billing", "note"), ("lone", "id"),
    }
    holdout_keys = set(plan.holdout)
    assert holdout_keys == {("other", "doc_ref")}
    for row in support_rows:
        assert (row.table, row.column) not in holdout_keys
        assert all((row.table, name) not in holdout_keys for name in row.siblings_full)
    by_code: dict[str, int] = {}
    for row in synthetic:
        by_code[row.code] = by_code.get(row.code, 0) + 1
        assert "https://" not in " ".join(row.sample_values)
        assert row.column not in {"amount", "note", "b", "id", "doc_ref"}
        assert set(row.siblings_full).isdisjoint({"amount", "note", "b", "id", "doc_ref"})
    assert by_code == {PDF_EXPORT: 5, PARENTING: 5, YOUNG_RESEARCHER: 5}
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
    assert prototypes
    assert any(value.startswith("kin-") for value in prototypes)
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
