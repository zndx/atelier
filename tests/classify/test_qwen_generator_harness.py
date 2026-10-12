"""Procedural-generator extension pack, exact-IRI gate, and NHSVM row source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atelier.classify.extension_patterns import (
    ExtensionError,
    accept_generators,
    code_iri,
    install_generators,
    load_generators,
)
from atelier.classify.insitu_corpus import prepare_insitu
from atelier.classify.qwen_generator_harness import (
    generator_argv,
    incorporate,
    write_generator_pack,
)
from atelier.classify.qwen_synth_harness import PROVIDERS
from atelier.classify.taxonomy import HierarchicalCategorySet, ReferenceCategory


_IRI = "https://signals.zndx.org/sdg#ROLE_PARENTING_ROLE"
_GOOD = f'''\
import random

GENERATORS = {{
    "{_IRI}": lambda rng: f"PR-{{rng.randint(100, 999)}}",
}}
'''


def _annotations(path: Path) -> None:
    path.write_text(
        "code,label,abbrev,parent_code,description,example_values,common_names\n"
        "SDG.ROLE.PARENTING_ROLE,Parenting Role,PARENTING_ROLE,SDG.ROLE,"
        "A role borne by one adult.,SECRET,slot_ref\n"
        "SDG.CONTACT.EMAIL,Email Address,EMAIL,SDG.CONTACT,An email address.,a@b.c,naming_map\n",
        encoding="utf-8",
    )


def _taxonomy() -> HierarchicalCategorySet:
    return HierarchicalCategorySet(name="sdg", categories=[
        ReferenceCategory(
            code="SDG.ROLE.PARENTING_ROLE", label="Parenting Role",
            embedding_text="role", abbrev="PARENTING_ROLE",
            description="A role borne by one adult.",
        ),
        ReferenceCategory(
            code="SDG.B", label="status", embedding_text="status",
            abbrev="STATUS", description="A status flag.",
        ),
    ])


def test_pack_lists_the_exact_iri(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    _annotations(annotations)
    (tmp_path / "naming_map.parquet").write_bytes(b"slot_ref")
    dest, codes = write_generator_pack(
        tmp_path / "work", annotations, limit=10, installed=tmp_path / "missing.py",
    )
    assert codes == ("SDG.ROLE.PARENTING_ROLE",)
    ontology = (dest / "ontology.csv").read_text(encoding="utf-8")
    assert ontology.splitlines()[0] == "code,label,abbrev,parent_code,description,iri"
    assert _IRI in ontology
    assert "SECRET" not in ontology
    assert "slot_ref" not in ontology
    assert "naming_map" not in ontology
    assert "example_values" not in ontology
    assert list(dest.rglob("*.parquet")) == []
    argv = generator_argv(dest)
    assert argv[argv.index("-m") + 1] == PROVIDERS["local"]
    assert "extend-procedural-generator" in argv[argv.index("-p") + 1]
    assert "qwen-thinking-cerebras" not in argv


def test_gate_accepts_only_the_requested_iri() -> None:
    batch = [{
        "code": "SDG.ROLE.PARENTING_ROLE",
        "label": "Parenting Role",
        "description": "A role borne by one adult.",
        "common_names": "",
    }]
    accepted, problems = accept_generators(_GOOD, batch)
    assert not problems
    assert accepted[0].code == "SDG.ROLE.PARENTING_ROLE"
    assert accepted[0].iri == _IRI
    other = _GOOD.replace(_IRI, "https://signals.zndx.org/sdg#ROLE")
    missed, reasons = accept_generators(other, batch)
    assert not missed
    assert any("not a requested IRI" in reason for reason in reasons)
    labeled = _GOOD.replace(_IRI, "Parenting Role")
    assert accept_generators(labeled, batch)[0] == []
    with pytest.raises(ExtensionError):
        load_generators("import os\nimport random\nGENERATORS = {}\n")


def test_installed_iri_becomes_the_nhsvm_value(tmp_path: Path) -> None:
    path = tmp_path / "insitu_extensions.py"
    install_generators([
        f'"{_IRI}": lambda rng: f"PR-{{rng.randint(100, 999)}}"',
    ], path)
    loaded = load_generators(path.read_text(encoding="utf-8"))
    assert set(loaded) == {_IRI}
    plan = prepare_insitu(
        _taxonomy(), tmp_path / "build", corpora=tmp_path / "absent",
        extra_generators=loaded, base_variants=1, amplified_variants=1, rows_per_column=4,
    )
    parenting = [v for row in plan.rows if row.code == "SDG.ROLE.PARENTING_ROLE" for v in row.sample_values]
    status = [v for row in plan.rows if row.code == "SDG.B" for v in row.sample_values]
    assert parenting and all(v.startswith("PR-") for v in parenting)
    assert status and all(not v.startswith("PR-") for v in status)
    assert plan.extended == ["SDG.ROLE.PARENTING_ROLE"]


def test_unshaped_term_is_omitted(tmp_path: Path) -> None:
    taxonomy = HierarchicalCategorySet(name="sdg", categories=[
        ReferenceCategory(
            code="SDG.DOM.DATASET_ENTITY", label="Dataset Entity",
            embedding_text="dataset", abbrev="DATASET_ENTITY",
            description="A catalog artifact.",
        ),
    ])
    other = {
        "https://signals.zndx.org/sdg#ARTIFACT": lambda rng: f"ART-{rng.randint(1000, 9999)}",
    }
    plan = prepare_insitu(
        taxonomy, tmp_path / "build", corpora=tmp_path / "absent",
        extra_generators=other, base_variants=1, amplified_variants=1, rows_per_column=4,
    )
    assert plan.rows == []
    assert plan.extended == []
    assert code_iri("SDG.DOM.DATASET_ENTITY") not in other


def test_incorporate_installs_only_the_requested_iri(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    _annotations(annotations)
    work = tmp_path / "work"
    write_generator_pack(work, annotations, limit=5, installed=tmp_path / "none.py")
    (work / "out" / "extend_generators.py").write_text(_GOOD + "\n", encoding="utf-8")
    installed = tmp_path / "installed.py"
    installed.write_text('"""empty"""\nimport random\n\nGENERATORS = {\n}\n', encoding="utf-8")
    report = incorporate(work, annotations, installed=installed)
    assert report["accepted"] == 1
    assert report["codes"] == ["SDG.ROLE.PARENTING_ROLE"]
    text = installed.read_text(encoding="utf-8")
    assert _IRI in text
    assert "SECRET" not in text
    assert json.loads((work / "request.json").read_text())["codes"] == ["SDG.ROLE.PARENTING_ROLE"]
