"""Sample sub-targets: collection/table/phase, not full-corpus CONVERGED."""

from __future__ import annotations

import pytest

from atelier.sdg.sample import current_sample_source_id, sample_dir_from_source_id
from atelier.sdg.targets import (
    parse_target_raw,
    resolve_sample_target,
    tables_for_collection,
)


def test_parse_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="unknown target kind"):
        parse_target_raw("foo:bar")
    with pytest.raises(ValueError, match="phase must be"):
        parse_target_raw("phase:sweep")


def test_collection_subtarget_on_live_sample() -> None:
    sid = current_sample_source_id()
    if not sid:
        pytest.skip("no sdg sample on disk")
    spec = resolve_sample_target(
        sid, "collection:computational-material-record-044e3d9a,phase:maxsim",
    )
    assert spec.collection == "computational-material-record-044e3d9a"
    assert spec.phase == "maxsim"
    assert spec.hard_gate
    assert spec.stop_after_precondition
    assert spec.precondition_stages == ("semantic_collection",)
    assert "computational_material_records" in spec.tables
    assert any(k.startswith("computational_material_records.") for k in spec.entity_keys)


def test_bare_sample_is_not_a_hard_coverage_gate() -> None:
    sid = current_sample_source_id()
    if not sid:
        pytest.skip("no sdg sample on disk")
    spec = resolve_sample_target(sid, "")
    assert not spec.hard_gate
    assert not spec.entity_keys
    assert spec.notes


def test_full_corpus_keeps_hard_gate() -> None:
    spec = resolve_sample_target("sdg-corpora", "")
    assert spec.hard_gate
    assert spec.precondition_stages is None


def test_tables_for_unknown_collection() -> None:
    sid = current_sample_source_id()
    d = sample_dir_from_source_id(sid) if sid else None
    if d is None:
        pytest.skip("no sdg sample on disk")
    import json
    manifest = json.loads((d / "manifest.json").read_text())
    with pytest.raises(ValueError, match="not in this sample"):
        tables_for_collection(manifest, "no-such-collection")
