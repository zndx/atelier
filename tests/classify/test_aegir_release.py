"""The Ægir DDL-release loader produces blind classification input.

Hermetic: builds a tiny two-table release (corpus + held-back key) in a
tmp dir, so the loader contract is exercised without the on-disk
``/raid`` artifact.  Guards the core invariant — a blind load must not
leak ``reference_code`` onto any column — and the held-back key joins.

Also covers the v2 (natural-register) surface: provenance-ladder fields
carried per column, the sdg-corpora vocabulary loader, and the
generation-manifest pin guard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from atelier.classify.aegir_release import (
    ReleasePinError,
    check_release_pin,
    load_aegir_reference,
    load_aegir_reference_by_name,
    load_aegir_release_samples,
    load_generation_manifest,
    load_sdg_vocabulary,
)


def _write_release(tmp: Path) -> Path:
    corpus = pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["tbl_0_c1", "tbl_0_c2", "tbl_1_c1"],
        "column_name": ["x", "y", "x"],
        "n_rows": [3, 3, 2],
        "sample_values": [
            ["SecurityPolicy", "RiskAnalysis", "Sanction"],
            ["PHIRecord", "Workforce", "PHIRecord"],
            ["2021-01-01", "2022-06-15"],
        ],
    })
    reference = pa.table({
        "table_id": ["tbl_0", "tbl_0", "tbl_1"],
        "column_id": ["tbl_0_c1", "tbl_0_c2", "tbl_1_c1"],
        "reference_code": [
            "SDG.ICE.DIRECTIVE.HIPAA_ADMIN",
            "SDG.ICE.DIRECTIVE.HIPAA_ADMIN",
            "SDG.ICE.TIME.EFFECTIVE_DATE",
        ],
        "slot_type": ["Class", "Class", "DataProperty"],
        "template_id": ["hipaa_admin", "hipaa_admin", "effective_date"],
        "source_table": ["t_hipaa_admin", "t_hipaa_admin", "t_effective_date"],
        "chapter_id": ["ch0", "ch0", "ch1"],
    })
    pq.write_table(corpus, tmp / "corpus_columns.parquet")
    pq.write_table(reference, tmp / "reference.parquet")
    return tmp


def test_blind_load_carries_values_not_reference(tmp_path):
    _write_release(tmp_path)
    samples = load_aegir_release_samples(tmp_path)

    assert [t.name for t in samples] == ["tbl_0", "tbl_1"]
    assert sum(len(t.columns) for t in samples) == 3

    c0 = samples[0].columns[0]
    assert c0.name == "x"
    assert c0.values[:1] == ["SecurityPolicy"]
    assert c0.database == "aegir_release"
    assert c0.qualified_name == "tbl_0.x"

    # The load-bearing invariant: a blind load leaks no answer key.
    assert all(c.reference_code is None for t in samples for c in t.columns)


def test_with_reference_populates_validation_channel(tmp_path):
    _write_release(tmp_path)
    samples = load_aegir_release_samples(tmp_path, with_reference=True)
    codes = {c.qualified_name: c.reference_code for t in samples for c in t.columns}
    assert codes == {
        "tbl_0.x": "SDG.ICE.DIRECTIVE.HIPAA_ADMIN",
        "tbl_0.y": "SDG.ICE.DIRECTIVE.HIPAA_ADMIN",
        "tbl_1.x": "SDG.ICE.TIME.EFFECTIVE_DATE",
    }


def test_reference_keys_join_by_id_and_by_name(tmp_path):
    _write_release(tmp_path)
    by_id = load_aegir_reference(tmp_path)
    by_name = load_aegir_reference_by_name(tmp_path)

    assert by_id[("tbl_1", "tbl_1_c1")] == "SDG.ICE.TIME.EFFECTIVE_DATE"
    # by-name is what scores blind predictions (keyed by table_id + name)
    assert by_name[("tbl_1", "x")] == "SDG.ICE.TIME.EFFECTIVE_DATE"
    assert by_name[("tbl_0", "x")] == "SDG.ICE.DIRECTIVE.HIPAA_ADMIN"


def test_max_tables_caps_load(tmp_path):
    _write_release(tmp_path)
    samples = load_aegir_release_samples(tmp_path, max_tables=1)
    assert [t.name for t in samples] == ["tbl_0"]


# ── Release v2 (natural register) ────────────────────────────────────


def _write_release_v2(tmp: Path, *, manifest: dict | None = None) -> Path:
    corpus = pa.table({
        "table_id": ["tbl_0", "tbl_0"],
        "column_id": ["tbl_0_c1", "tbl_0_c2"],
        "column_name": ["caregiver_roles", "adult_id"],
        "register": ["natural", "natural"],
        "name_provenance": ["engine-derived", "degraded-mechanical"],
        "n_rows": [3, 3],
        "sample_values": [
            ["guardian", "foster parent", "kinship carer"],
            ["a-001", "a-002", "a-003"],
        ],
        "fk_to_table_id": [None, "tbl_9"],
    })
    reference = pa.table({
        "table_id": ["tbl_0", "tbl_0"],
        "column_id": ["tbl_0_c1", "tbl_0_c2"],
        "reference_code": ["SDG.DOM.PARENTING_ROLE", "SDG.GENERIC.IDENTIFIER"],
        "template_id": ["parenting_role", "parenting_role"],
        "source_table": ["t_parenting_role", "t_parenting_role"],
        "column_name": ["caregiver_roles", "adult_id"],
        "semantic_table": ["parenting_role", "parenting_role"],
        "semantic_col": ["parenting_role", "disabled_adult"],
        "slot_ref": ["", ""],
        "kind": ["subject", "fk"],
    })
    pq.write_table(corpus, tmp / "corpus_columns.parquet")
    pq.write_table(reference, tmp / "reference.parquet")
    stats: dict = {"release_version": "v2-natural-register", "n_tables": 1}
    if manifest is not None:
        stats["generation_manifest"] = manifest
    (tmp / "release_stats.json").write_text(json.dumps(stats))
    return tmp


def test_v2_load_carries_provenance_ladder_and_column_id(tmp_path):
    _write_release_v2(tmp_path)
    samples = load_aegir_release_samples(tmp_path)

    cols = {c.name: c for t in samples for c in t.columns}
    care = cols["caregiver_roles"]
    # column_id retained — the predictions-parquet join key.
    assert care.column_id == "tbl_0_c1"
    assert care.register == "natural"
    assert care.name_provenance == "engine-derived"
    assert care.fk_to_table_id is None
    assert cols["adult_id"].name_provenance == "degraded-mechanical"
    assert cols["adult_id"].fk_to_table_id == "tbl_9"
    # Blind stays blind on v2.
    assert all(c.reference_code is None for c in cols.values())
    # Provenance survives serialization (results artifacts can slice by rung).
    assert care.to_dict()["name_provenance"] == "engine-derived"


def test_v03_load_leaves_v2_fields_none(tmp_path):
    _write_release(tmp_path)
    samples = load_aegir_release_samples(tmp_path)
    c0 = samples[0].columns[0]
    assert c0.column_id == "tbl_0_c1"  # column_id existed in v0.3 too
    assert c0.register is None
    assert c0.name_provenance is None
    assert c0.fk_to_table_id is None
    # v0.3 artifacts keep their serialized shape (no rung keys).
    assert "register" not in c0.to_dict()


# ── SDG vocabulary (sdg-corpora annotations.parquet) ─────────────────


def _write_corpora(tmp: Path, *, n_extra: int = 0) -> Path:
    corpora = tmp / "corpora"
    (corpora / "vocabulary").mkdir(parents=True)
    codes = ["SDG.ICE", "SDG.ICE.DIRECTIVE", "SDG.DOM.EPISTEMIC_UNCERTAINTY_METRIC"]
    labels = ["Information Content Entity", "Directive ICE", "Epistemic Uncertainty Metric"]
    parents = [None, "SDG.ICE", "SDG.ICE"]
    examples = ["", "", "belief_interval | confidence | plausibility_upper_bound"]
    for i in range(n_extra):
        codes.append(f"SDG.GENERIC.X{i}")
        labels.append(f"X{i}")
        parents.append("SDG.ICE")
        examples.append("")
    table = pa.table({
        "code": codes,
        "label": labels,
        "abbrev": ["ICE", "DIR_ICE", "EUM"] + [f"X{i}" for i in range(n_extra)],
        "notation": ["1", "1.1", "1.9"] + ["9"] * n_extra,
        "parent_code": parents,
        "taxonomy": ["sdg"] * len(codes),
        "description": ["A CCO ICE."] * len(codes),
        "common_names": ["", "", "domain_hypernym"] + [""] * n_extra,
        "example_values": examples,
    })
    pq.write_table(table, corpora / "vocabulary" / "annotations.parquet")
    return corpora


def test_sdg_vocabulary_loads_hierarchically(tmp_path):
    corpora = _write_corpora(tmp_path)
    cats = load_sdg_vocabulary(corpora)

    assert len(cats.categories) == 3
    assert "SDG.ICE.DIRECTIVE" in cats.by_code
    # Explicit parent_code wins over dot-derivation: the DOM term parents
    # to SDG.ICE (cross-branch), exactly as annotations.parquet declares.
    assert cats.by_code["SDG.DOM.EPISTEMIC_UNCERTAINTY_METRIC"].parent_code == "SDG.ICE"
    assert cats.by_code["SDG.ICE"].parent_code is None
    # example_values feed embedding_text via the specifics channel.
    eum = cats.by_code["SDG.DOM.EPISTEMIC_UNCERTAINTY_METRIC"]
    assert "belief_interval" in eum.embedding_text
    assert eum.taxonomy == "sdg"


def test_sdg_vocabulary_missing_fails_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="submodule"):
        load_sdg_vocabulary(tmp_path / "nowhere")


# ── Generation-manifest pin guard ────────────────────────────────────


_MANIFEST = {
    "ontology_sha": "abc1234",
    "vocab_generation": 3,
    "ddl_run_id": "run0",
    "corpus_run_id": None,
    "naming": "natural",
    "holdout_partition": "preview-unsplit",
}


def test_pin_guard_accepts_matching_generation(tmp_path):
    release = _write_release_v2(tmp_path, manifest=_MANIFEST)
    corpora = _write_corpora(tmp_path)
    manifest = check_release_pin(
        release, corpora, expected_ontology_sha="abc1234"
    )
    assert manifest["ddl_run_id"] == "run0"
    # Prefix-tolerant sha comparison (--short length drift).
    check_release_pin(
        release, corpora, expected_ontology_sha="abc1234567890def"
    )


def test_pin_guard_refuses_pre_v2_release(tmp_path):
    release = _write_release_v2(tmp_path)  # stats without manifest
    corpora = _write_corpora(tmp_path)
    with pytest.raises(ReleasePinError, match="pre-v2"):
        check_release_pin(release, corpora, expected_ontology_sha="abc1234")


def test_pin_guard_refuses_sha_mismatch(tmp_path):
    release = _write_release_v2(tmp_path, manifest=_MANIFEST)
    corpora = _write_corpora(tmp_path)
    with pytest.raises(ReleasePinError, match="pin mismatch"):
        check_release_pin(release, corpora, expected_ontology_sha="fff0000")


def test_pin_guard_refuses_vocab_generation_drift(tmp_path):
    release = _write_release_v2(tmp_path, manifest=_MANIFEST)
    corpora = _write_corpora(tmp_path, n_extra=2)  # 5 codes vs declared 3
    with pytest.raises(ReleasePinError, match="vocabulary generation"):
        check_release_pin(release, corpora, expected_ontology_sha="abc1234")


def test_pin_guard_refuses_unresolvable_pin(tmp_path):
    release = _write_release_v2(tmp_path, manifest=_MANIFEST)
    corpora = _write_corpora(tmp_path)  # not a git checkout, no override
    with pytest.raises(ReleasePinError, match="unpinned"):
        check_release_pin(release, corpora)


def test_generation_manifest_absent_is_empty(tmp_path):
    _write_release(tmp_path)  # v0.3 shape — no stats file at all
    assert load_generation_manifest(tmp_path) == {}


def test_pin_guard_requires_holdout_partition(tmp_path):
    manifest = {k: v for k, v in _MANIFEST.items() if k != "holdout_partition"}
    release = _write_release_v2(tmp_path, manifest=manifest)
    corpora = _write_corpora(tmp_path)
    with pytest.raises(ReleasePinError, match="holdout_partition"):
        check_release_pin(release, corpora, expected_ontology_sha="abc1234")


def test_pin_guard_scored_refuses_preview_cut(tmp_path):
    release = _write_release_v2(tmp_path, manifest=_MANIFEST)
    corpora = _write_corpora(tmp_path)
    # Unscored (machinery/dev) runs accept preview cuts...
    check_release_pin(release, corpora, expected_ontology_sha="abc1234")
    # ...the efficacy gate does not.
    with pytest.raises(ReleasePinError, match="scored run refused"):
        check_release_pin(
            release, corpora, expected_ontology_sha="abc1234", scored=True
        )


# ── .key sibling layout (key separation, 2026-07-03) ─────────────────


def test_reference_resolves_from_key_sibling(tmp_path):
    from atelier.classify.aegir_release import resolve_reference_path

    release = tmp_path / "rel"
    release.mkdir()
    _write_release_v2(release)
    # Move the key to the sibling dir — the recut layout.
    key_dir = tmp_path / "rel.key"
    key_dir.mkdir()
    (release / "reference.parquet").rename(key_dir / "reference.parquet")

    assert resolve_reference_path(release) == key_dir / "reference.parquet"
    # Loading the reference and the blind samples both still work.
    ref = load_aegir_reference(release)
    assert ref[("tbl_0", "tbl_0_c1")] == "SDG.DOM.PARENTING_ROLE"
    samples = load_aegir_release_samples(release)
    assert all(c.reference_code is None for t in samples for c in t.columns)
