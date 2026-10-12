"""Procedural-generator extension pack, exact-IRI gate, and NHSVM row source."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atelier.classify.extension_patterns import (
    ExtensionError,
    accept_generators,
    code_iri,
    install_generators,
    load_generators,
)
from atelier.classify.insitu_corpus import InSituError, prepare_insitu
from atelier.classify.qwen_generator_harness import (
    generator_argv,
    incorporate,
    write_generator_pack,
)
from atelier.classify.qwen_synth_harness import PROVIDERS, QwenSynthError
from tests.classify.explicit_sdg import (
    CLINICAL,
    PARENTING,
    PDF_EXPORT,
    ROLE,
    YOUNG_RESEARCHER,
    published_rows,
    relational_procedure,
    taxonomy,
    write_annotations,
)


_IRI = "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE"
_GOOD = f'''\
import random

GENERATORS = {{
    "{_IRI}": lambda rng: {{
        "table": f"care_placement_{{rng.randint(10, 99)}}",
        "columns": [
            {{
                "name": "caregiver_role",
                "values": [f"{{rng.choice(['primary', 'kinship', 'foster'])}}-{{rng.randint(10, 99)}}"],
                "iri": "{_IRI}",
            }},
            {{
                "name": "started_on",
                "values": [f"201{{rng.randint(0, 9)}}-04-01"],
            }},
        ],
    }},
}}
'''


def _batch() -> list[dict]:
    row = published_rows([PARENTING])[0]
    return [{
        "code": row["code"],
        "label": row["label"],
        "description": row["description"],
        "common_names": "",
    }]


def test_pack_lists_the_exact_iri(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    write_annotations(annotations, [PARENTING, PDF_EXPORT], sentinel=True)
    (tmp_path / "naming_map.parquet").write_bytes(b"slot_ref")
    dest, codes = write_generator_pack(
        tmp_path / "work", annotations, [PARENTING, PDF_EXPORT],
        limit=10, installed=tmp_path / "missing.py",
    )
    assert codes == (PARENTING, PDF_EXPORT)
    ontology = (dest / "ontology.csv").read_text(encoding="utf-8")
    assert ontology.splitlines()[0] == "code,label,abbrev,parent_code,description,iri"
    assert _IRI in ontology
    assert "https://signals.zndx.org/sdg#PROCESS_PDF_EXPORT_PROCESS" in ontology
    assert "INV-SECRET" not in ontology
    assert "slot_ref" not in ontology
    assert "naming_map" not in ontology
    assert "example_values" not in ontology
    assert list(dest.rglob("*.parquet")) == []
    argv = generator_argv(dest)
    assert argv[argv.index("-m") + 1] == PROVIDERS["local"]
    assert "extend-procedural-generator" in argv[argv.index("-p") + 1]
    assert "qwen-thinking-cerebras" not in argv


def test_gate_accepts_only_the_requested_iri() -> None:
    batch = _batch()
    accepted, problems = accept_generators(_GOOD, batch)
    assert not problems
    assert accepted[0].code == PARENTING
    assert accepted[0].iri == _IRI
    genus = _GOOD.replace(_IRI, "https://signals.zndx.org/sdg#ROLE")
    missed, reasons = accept_generators(genus, batch)
    assert not missed
    assert any("not a requested IRI" in reason for reason in reasons)
    labeled = _GOOD.replace(_IRI, "Parenting Role")
    assert accept_generators(labeled, batch)[0] == []
    scalar = f'''\
import random

GENERATORS = {{
    "{_IRI}": lambda rng: f"PR-{{rng.randint(100, 999)}}",
}}
'''
    rejected, scalar_reasons = accept_generators(scalar, batch)
    assert not rejected
    assert any("return a table" in reason for reason in scalar_reasons)
    with pytest.raises(ExtensionError):
        load_generators("import os\nimport random\nGENERATORS = {}\n")


def test_installed_iri_becomes_a_relational_table(tmp_path: Path) -> None:
    path = tmp_path / "insitu_extensions.py"
    install_generators([
        f'''"{_IRI}": lambda rng: {{
        "table": f"care_placement_{{rng.randint(10, 99)}}",
        "columns": [
            {{"name": "caregiver_role", "values": [f"primary-{{rng.randint(10, 99)}}"], "iri": "{_IRI}"}},
            {{"name": "started_on", "values": [f"201{{rng.randint(0, 9)}}-04-01"]}},
        ],
    }}''',
    ], path)
    loaded = load_generators(path.read_text(encoding="utf-8"))
    assert set(loaded) == {_IRI}
    young_iri = code_iri(YOUNG_RESEARCHER)
    loaded[young_iri] = relational_procedure(young_iri, "study_role", "cohort")
    pin = tmp_path / "corpora" / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.table({
        "table_name": ["household", "household"],
        "col_name": ["caregiver", "study"],
        "value": ["kin-real", "cohort-real"],
        "row_ix": [0, 0],
    }), pin / "base_rows.parquet")
    frame = taxonomy([ROLE, PARENTING, YOUNG_RESEARCHER])
    plan = prepare_insitu(
        frame, tmp_path / "build", corpora=tmp_path / "corpora",
        assignments={
            ("household", "study"): PARENTING,
            ("household", "caregiver"): YOUNG_RESEARCHER,
        },
        extra_generators=loaded,
        base_variants=1, amplified_variants=1, rows_per_column=4, floor=1,
    )
    parenting = [
        value for row in plan.rows if row.code == PARENTING and row.table.startswith("care_")
        for value in row.sample_values
    ]
    young = [
        value for row in plan.rows if row.code == YOUNG_RESEARCHER for value in row.sample_values
    ]
    assert parenting and all(value.startswith("primary-") for value in parenting)
    assert young and all(value.startswith("cohort-") for value in young)
    assert plan.extended == [PARENTING, YOUNG_RESEARCHER]
    assert ROLE not in plan.extended


def test_missing_iri_generator_is_an_error(tmp_path: Path) -> None:
    pin = tmp_path / "corpora" / "ddl" / "pin1"
    pin.mkdir(parents=True)
    pq.write_table(pa.table({
        "table_name": ["chart", "chart"],
        "col_name": ["activity", "note"],
        "value": ["visit", "text"],
        "row_ix": [0, 0],
    }), pin / "base_rows.parquet")
    other = {
        _IRI: relational_procedure(_IRI, "caregiver_role", "kin"),
    }
    with pytest.raises(InSituError, match="no relational generator"):
        prepare_insitu(
            taxonomy([CLINICAL]), tmp_path / "build", corpora=tmp_path / "corpora",
            assignments={("chart", "activity"): CLINICAL},
            extra_generators=other, base_variants=1, amplified_variants=1, rows_per_column=4,
        )


def test_incorporate_installs_only_the_requested_iri(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    write_annotations(annotations, [PARENTING, PDF_EXPORT], sentinel=True)
    work = tmp_path / "work"
    write_generator_pack(
        work, annotations, [PARENTING, PDF_EXPORT],
        limit=5, installed=tmp_path / "none.py",
    )
    (work / "out" / "extend_generators.py").write_text(_GOOD + "\n", encoding="utf-8")
    installed = tmp_path / "installed.py"
    installed.write_text('"""empty"""\nimport random\n\nGENERATORS = {\n}\n', encoding="utf-8")
    report = incorporate(work, annotations, installed=installed)
    assert report["accepted"] == 1
    assert report["codes"] == [PARENTING]
    text = installed.read_text(encoding="utf-8")
    assert _IRI in text
    assert "INV-SECRET" not in text
    assert json.loads((work / "request.json").read_text())["codes"] == [
        PARENTING,
        PDF_EXPORT,
    ]


def test_code_outside_the_explicit_slice_is_not_packed(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    write_annotations(annotations, [PARENTING])
    with pytest.raises(QwenSynthError, match=CLINICAL):
        write_generator_pack(
            tmp_path / "work", annotations, [CLINICAL],
            installed=tmp_path / "none.py",
        )
