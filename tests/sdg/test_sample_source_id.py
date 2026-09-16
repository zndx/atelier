"""Sample source-id is sdg-corpora/<pin>_<profile>, not the full corpus id."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atelier.sdg.sample import (
    CORPUS_SOURCE_ID,
    current_sample_source_id,
    is_corpus_source_id,
    is_sample_source_id,
    sample_dir_from_source_id,
    sample_source_id,
)


def test_sample_id_is_not_the_corpus_id() -> None:
    sid = sample_source_id("b24ef9f606605dd7bb8877bddd2d9672c78282ad", "macbook")
    assert sid == "sdg-corpora/b24ef9f60660_macbook"
    assert is_sample_source_id(sid)
    assert not is_corpus_source_id(sid)
    assert is_corpus_source_id(CORPUS_SOURCE_ID)
    assert not is_sample_source_id(CORPUS_SOURCE_ID)
    assert not is_sample_source_id("sdg-corpora/")


def test_current_sample_id_from_pointer(tmp_path: Path) -> None:
    name = "b24ef9f60660_macbook"
    sample = tmp_path / "sdg_sample" / name
    sample.mkdir(parents=True)
    (tmp_path / "sdg_sample" / "current.json").write_text(json.dumps({
        "path": str(sample),
        "source_id": "sdg-corpora",
        "corpus_commit": "b24ef9f606605dd7bb8877bddd2d9672c78282ad",
        "profile": "macbook",
    }))
    sid = current_sample_source_id(artifact_root=tmp_path)
    assert sid == f"sdg-corpora/{name}"
    assert sample_dir_from_source_id(sid, artifact_root=tmp_path) == sample
    assert sample_dir_from_source_id("sdg-corpora", artifact_root=tmp_path) is None


def test_sample_source_id_requires_pin_and_profile() -> None:
    with pytest.raises(Exception):
        sample_source_id("", "macbook")
