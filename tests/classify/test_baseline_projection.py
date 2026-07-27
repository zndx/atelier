"""The structural-baseline builder produces a golden embedding-atlas artifact.

Hermetic: builds a tiny release in tmp, runs the builder with no Atlas, and
asserts the golden invariants (prediction == curated reference, full belief,
code-clustered layout) plus schema parity with the pipeline's own writer.
"""
from __future__ import annotations

import importlib.util
import math
from argparse import Namespace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_baseline_projection.py"
_spec = importlib.util.spec_from_file_location("build_baseline_projection", _SCRIPT)
bbp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bbp)  # type: ignore[union-attr]


def test_humanize_and_code_leaf():
    assert bbp._humanize("hipaa_safeguard_admin") == "Hipaa Safeguard Admin"
    assert bbp._code_leaf("SDG.ICE.DIRECTIVE.HIPAA_SAFEGUARD_ADMIN") == "HIPAA_SAFEGUARD_ADMIN"
    assert bbp._code_leaf("") == ""


def test_det_unit_is_deterministic_and_bounded():
    a = bbp._det_unit("tbl_0_c1", "x")
    b = bbp._det_unit("tbl_0_c1", "x")
    c = bbp._det_unit("tbl_0_c1", "y")
    assert a == b and a != c
    assert -1.0 <= a < 1.0


def test_cluster_layout_separates_codes():
    codes = ["A", "A", "B", "B", "C"]
    keys = [("t", f"c{i}") for i in range(5)]
    xs, ys = bbp._cluster_layout(codes, keys)
    # same-code points are far closer than different-code points
    a0, a1 = (xs[0], ys[0]), (xs[1], ys[1])
    b0 = (xs[2], ys[2])
    assert math.dist(a0, a1) < math.dist(a0, b0)


def _write_tiny_release(tmp: Path) -> Path:
    tmp.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["tbl_0_c1", "tbl_0_c2", "tbl_1_c1"],
        "column_name": ["x", "y", "x"],
        "n_rows": [3, 3, 2],
        "sample_values": [["SecurityPolicy"], ["PHIRecord"], ["2021-01-01"]],
    }), tmp / "corpus_columns.parquet")
    pq.write_table(pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["tbl_0_c1", "tbl_0_c2", "tbl_1_c1"],
        "reference_code": ["SDG.ICE.DIRECTIVE.HIPAA_ADMIN", "SDG.ICE.DIRECTIVE.HIPAA_ADMIN",
                           "SDG.ICE.TIME.EFFECTIVE_DATE"],
        "slot_type": ["Class", "Class", "Class"],
        "template_id": ["hipaa_admin", "hipaa_admin", "effective_date"],
        "source_table": ["t_hipaa_admin", "t_hipaa_admin", "t_effective_date"],
        "chapter_id": ["c0", "c0", "c1"],
    }), tmp / "reference.parquet")
    return tmp


def test_build_emits_golden_run(tmp_path):
    release = _write_tiny_release(tmp_path / "rel")
    out_root = tmp_path / "results"

    args = Namespace(
        release_dir=str(release), run_id="baseline_test", results_root=str(out_root),
        source_id="aegir-baseline/test", per_template=0,
        atlas_url=None, atlas_user="admin", atlas_password="admin", verify_atlas=False,
    )
    rc = bbp.build(args)
    assert rc == 0

    run_dir = out_root / "baseline_test"
    assert (run_dir / "classifications.json").is_file()
    assert (run_dir / "settings_snapshot.json").is_file()
    assert (run_dir / "atlas_correspondence.csv").is_file()

    rows = pq.read_table(run_dir / "atelier_embeddings.parquet").to_pylist()
    assert len(rows) == 3
    # golden: every prediction equals its curated reference, at full belief
    assert all(r["predicted_code"] == r["reference_code"] for r in rows)
    assert all(r["matches_reference"] is True for r in rows)
    assert all(r["belief"] == 1.0 and r["conflict"] == 0.0 for r in rows)
    # hover text is "<label> - <annotation>"
    r0 = next(r for r in rows if "HIPAA" in r["predicted_code"])
    assert r0["text"] == "Hipaa Admin - HIPAA_ADMIN"
    # schema carries the embedding-atlas essentials
    assert {"text", "x", "y", "predicted_label", "reference_code"} <= set(rows[0])
