"""The SDG agent-mediated stack: working-set builder, curation loop, audit.

Hermetic — tmp release/corpora fixtures, stubbed referee. Guards the blind
contract mechanics: single ingress with hashes, legacy-layout refusal,
schema-driven curation with invalid-code retry, resume safety, and the
integrity audit's violation detection.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


builder = _load("build_agent_mediated_sdg")
curate = _load("run_curate_local")
integrity = _load("audit_blind_integrity")


# ── Fixtures ─────────────────────────────────────────────────────────


def _write_world(tmp: Path) -> tuple[Path, Path]:
    """A key-separated v2 release + a corpora checkout, pin-consistent."""
    release = tmp / "rel"
    release.mkdir()
    key = tmp / "rel.key"
    key.mkdir()
    corpus = pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["c1", "c2", "c3"],
        "column_name": ["caregiver_roles", "adult_id", "checksum"],
        "register": ["natural"] * 3,
        "name_provenance": ["engine-derived", "composed", "semantic-passthrough"],
        "n_rows": [3, 3, 2],
        "sample_values": [
            ["guardian", "foster parent", "kinship carer"],
            ["a-1", "a-2", "a-3"],
            ["9f2c", "77ab"],
        ],
        "fk_to_table_id": [None, "tbl_9", None],
    })
    pq.write_table(corpus, release / "corpus_columns.parquet")
    pq.write_table(pa.table({
        "table_id": ["tbl_0"], "column_id": ["c1"],
        "reference_code": ["SDG.DOM.PARENTING_ROLE"],
    }), key / "reference.parquet")

    corpora = tmp / "corpora"
    (corpora / "vocabulary").mkdir(parents=True)
    pq.write_table(pa.table({
        "code": ["SDG.ICE", "SDG.DOM.PARENTING_ROLE", "SDG.GENERIC.IDENTIFIER",
                 "SDG.GENERIC.CHECKSUM"],
        "label": ["Information Content Entity", "Parenting Role",
                  "Identifier", "Checksum"],
        "abbrev": ["ICE", "PRL", "ID", "CK"],
        "notation": ["1", "1.1", "2", "2.1"],
        "parent_code": [None, "SDG.ICE", "SDG.ICE", "SDG.GENERIC.IDENTIFIER"],
        "taxonomy": ["sdg"] * 4,
        "description": ["root", "a parenting role", "an identifier", "a checksum"],
        "common_names": ["", "", "", ""],
        "example_values": ["", "", "", ""],
    }), corpora / "vocabulary" / "annotations.parquet")

    (release / "release_stats.json").write_text(json.dumps({
        "generation_manifest": {
            "ontology_sha": "abc1234", "vocab_generation": 4,
            "ddl_run_id": "r0", "corpus_run_id": None,
            "naming": "natural", "holdout_partition": "preview-unsplit",
        },
    }))
    return release, corpora


def _build_ws(tmp: Path) -> dict:
    release, corpora = _write_world(tmp)
    return builder.build_working_set(release, corpora)


# ── Builder ──────────────────────────────────────────────────────────


def test_builder_emits_blind_working_set_with_ingress_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atelier.classify.aegir_release._resolve_corpora_sha", lambda _: "abc1234"
    )
    ws = _build_ws(tmp_path)

    meta = ws["metadata"]
    assert meta["blind"] is True
    assert meta["vocabulary_size"] == 4
    assert meta["column_count"] == 3
    assert len(meta["ingress_sha256"]) == 2  # exactly two files, hashed
    col = ws["columns"]["tbl_0.caregiver_roles"]
    assert col["column_id"] == "c1"
    assert col["name_provenance"] == "engine-derived"
    assert "reference" not in json.dumps(ws["columns"])  # nothing key-shaped


def test_builder_refuses_legacy_in_dir_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atelier.classify.aegir_release._resolve_corpora_sha", lambda _: "abc1234"
    )
    release, corpora = _write_world(tmp_path)
    # Simulate the legacy layout: key beside the blind surface.
    (tmp_path / "rel.key" / "reference.parquet").rename(release / "reference.parquet")
    with pytest.raises(SystemExit, match="key-separated"):
        builder.build_working_set(release, corpora)
    # Dev override still works.
    ws = builder.build_working_set(release, corpora, allow_legacy_layout=True)
    assert ws["metadata"]["blind"] is True


# ── Curation loop (stubbed referee) ──────────────────────────────────


class _StubReferee:
    """Schema-aware stub: shortlists everything to two codes; decides
    PARENTING_ROLE for caregiver columns, first bad code once for adult_id
    (exercising the retry), CHECKSUM otherwise."""

    def __init__(self):
        self.calls = 0
        self.adult_attempts = 0

    def __call__(self, prompt, *, capability, system_prompt="", max_tokens=0,
                 temperature=0.0, json_schema="", timeout=0.0):
        self.calls += 1
        assert capability == "referee"
        schema = json.loads(json_schema)
        if "columns" in schema.get("properties", {}):  # shortlist
            names = [ln.split()[1] for ln in prompt.splitlines()
                     if ln.startswith("- ")]
            payload = {"columns": [
                {"column_name": n,
                 "candidates": ["SDG.DOM.PARENTING_ROLE", "SDG.GENERIC.IDENTIFIER"]}
                for n in names
            ]}
        else:  # decision
            name = prompt.split("Column: ")[1].split()[0]
            if name == "caregiver_roles":
                code = "SDG.DOM.PARENTING_ROLE"
            elif name == "adult_id":
                self.adult_attempts += 1
                code = "SDG.NOT_A_CODE" if self.adult_attempts == 1 \
                    else "SDG.GENERIC.IDENTIFIER"
            else:
                code = "SDG.GENERIC.CHECKSUM"
            payload = {"column_name": name, "code": code,
                       "confidence": 0.9, "rationale": "stub"}
        return {"text": json.dumps(payload), "reasoning_content": "trace",
                "finish_reason": "stop", "model": "stub-referee",
                "prompt_tokens": 1, "completion_tokens": 1, "latency_ms": 1.0}


def test_curator_walks_tables_with_retry_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atelier.classify.aegir_release._resolve_corpora_sha", lambda _: "abc1234"
    )
    ws = _build_ws(tmp_path)
    out_dir = tmp_path / "am"
    stub = _StubReferee()

    curator = curate.Curator(ws, out_dir, workers=1, complete_fn=stub)
    summary = curator.run()

    assert summary["decisions"] == 3
    assert summary["resolved"] == 3
    decisions = json.loads((out_dir / "agent_mediated.json").read_text())
    assert decisions["tbl_0.caregiver_roles"]["code"] == "SDG.DOM.PARENTING_ROLE"
    assert decisions["tbl_0.adult_id"]["code"] == "SDG.GENERIC.IDENTIFIER"
    audit = json.loads((out_dir / "audit.json").read_text())
    assert audit["tbl_0.adult_id"]["retried"] is True
    assert audit["tbl_0.adult_id"]["name_provenance"] == "composed"
    assert audit["tbl_0.caregiver_roles"]["reasoning_head"] == "trace"

    # Resume: a fresh curator over the same out_dir has nothing to do.
    calls_before = stub.calls
    curate.Curator(ws, out_dir, workers=1, complete_fn=stub).run()
    assert stub.calls == calls_before


# ── Blind-integrity audit ────────────────────────────────────────────


def test_integrity_audit_passes_then_catches_planted_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atelier.classify.aegir_release._resolve_corpora_sha", lambda _: "abc1234"
    )
    ws = _build_ws(tmp_path)
    out_dir = tmp_path / "am"
    out_dir.mkdir()
    (out_dir / "working_set.json").write_text(json.dumps(ws))
    curate.Curator(ws, out_dir, workers=1, complete_fn=_StubReferee()).run()

    assert integrity.audit(out_dir) == []

    # Plant an answer-key marker in the audit trail → caught.
    audit = json.loads((out_dir / "audit.json").read_text())
    audit["tbl_0.caregiver_roles"]["rationale"] = \
        "matched semantic_col parenting_role"
    (out_dir / "audit.json").write_text(json.dumps(audit))
    violations = integrity.audit(out_dir)
    assert any("semantic_col" in v for v in violations)


def test_integrity_audit_detects_ingress_change(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "atelier.classify.aegir_release._resolve_corpora_sha", lambda _: "abc1234"
    )
    release, corpora = _write_world(tmp_path)
    ws = builder.build_working_set(release, corpora)
    out_dir = tmp_path / "am"
    out_dir.mkdir()
    (out_dir / "working_set.json").write_text(json.dumps(ws))

    # Mutate the blind surface after the build.
    pq.write_table(pa.table({"table_id": ["x"], "column_id": ["y"],
                             "column_name": ["z"], "n_rows": [1],
                             "sample_values": [["v"]]}),
                   release / "corpus_columns.parquet")
    violations = integrity.audit(out_dir)
    assert any("CHANGED" in v for v in violations)
