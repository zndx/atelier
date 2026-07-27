"""The predictions emitter writes the settled efficacy-gate handoff schema.

Hermetic: exercises golden-mode, the run→release id join, and the parquet
schema against a tmp v2 release — no Ægir tree, no pipeline run.  The
script is loaded via importlib (``scripts/`` is not a package), mirroring
``test_baseline_projection.py``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "emit_aegir_predictions.py"
_spec = importlib.util.spec_from_file_location("emit_aegir_predictions", _SCRIPT)
emit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emit)  # type: ignore[union-attr]


def _write_release(tmp: Path) -> Path:
    corpus = pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["c1", "c2", "c3"],
        "column_name": ["caregiver_roles", "adult_id", "checksum"],
        "register": ["natural"] * 3,
        "name_provenance": ["engine-derived", "composed", "semantic-passthrough"],
        "n_rows": [2, 2, 2],
        "sample_values": [["a", "b"], ["1", "2"], ["x", "y"]],
        "fk_to_table_id": [None, None, None],
    })
    reference = pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["c1", "c2", "c3"],
        "reference_code": [
            "SDG.DOM.PARENTING_ROLE",
            # Set-valued reference (P5+): golden must emit a scoreable member.
            "SDG.GENERIC.IDENTIFIER|SDG.GENERIC.KEY",
            "SDG.GENERIC.CHECKSUM",
        ],
        "template_id": ["t", "t", "u"],
        "source_table": ["st", "st", "su"],
    })
    pq.write_table(corpus, tmp / "corpus_columns.parquet")
    pq.write_table(reference, tmp / "reference.parquet")
    return tmp


def test_golden_predictions_mirror_reference(tmp_path):
    _write_release(tmp_path)
    preds = emit.golden_predictions(tmp_path)

    assert len(preds) == 3
    by_key = {(p["table_id"], p["column_id"]): p for p in preds}
    assert by_key[("tbl_0", "c1")]["predicted_code"] == "SDG.DOM.PARENTING_ROLE"
    # Set-valued reference collapses to its first member (scorer credits
    # any member at 1.0).
    assert by_key[("tbl_0", "c2")]["predicted_code"] == "SDG.GENERIC.IDENTIFIER"
    assert all(p["belief"] == 1.0 and p["plausibility"] == 1.0 for p in preds)


def test_run_predictions_join_names_back_to_ids(tmp_path):
    _write_release(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "classifications.json").write_text(json.dumps([
        {"table_name": "tbl_0", "column_name": "caregiver_roles",
         "predicted_code": "SDG.DOM.PARENTING_ROLE",
         "belief": 0.82, "plausibility": 0.95},
        {"table_name": "tbl_1", "column_name": "checksum",
         "predicted_code": "SDG.GENERIC.CHECKSUM",
         "belief": 0.4, "plausibility": 0.9},
        # Not in the release — must be counted, not emitted.
        {"table_name": "tbl_9", "column_name": "ghost",
         "predicted_code": "SDG.ICE"},
    ]))

    preds, unjoined = emit.run_predictions(tmp_path, run_dir)
    assert unjoined == 1
    assert {(p["table_id"], p["column_id"]) for p in preds} == {
        ("tbl_0", "c1"), ("tbl_1", "c3"),
    }
    assert preds[0]["belief"] == pytest.approx(0.82)


def test_write_predictions_schema(tmp_path):
    out = tmp_path / "preds.parquet"
    emit.write_predictions(
        [
            {"table_id": "t", "column_id": "c", "predicted_code": "SDG.ICE",
             "belief": 1.0, "plausibility": 1.0},
        ],
        out,
    )
    schema = pq.read_schema(out)
    assert [f.name for f in schema] == [
        "table_id", "column_id", "predicted_code", "belief", "plausibility",
    ]


def test_write_predictions_refuses_empty(tmp_path):
    with pytest.raises(SystemExit):
        emit.write_predictions([], tmp_path / "empty.parquet")
