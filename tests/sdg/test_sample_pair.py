"""Target/reference pair: disjoint collections, genus IRI cover."""

from __future__ import annotations

from pathlib import Path

import pytest

from atelier.sdg.pair import (
    GURU_IRI,
    GURU_OVERLAP,
    PairError,
    genus_iris,
    select_reference_collections,
)
from atelier.sdg.sample import Collection, SampleProfile


def _coll(slug: str, genera: list[str], columns: int = 10) -> Collection:
    terms = [{"name": g, "genus": g} for g in genera]
    c = Collection(
        slug=slug,
        path=Path("/tmp") / slug,
        manifest={"terms": terms, "tables": [{"columns": [None] * columns}]},
        column_count=columns,
    )
    c.genera = set(genera)
    return c


def test_select_reference_covers_without_overlap() -> None:
    target = _coll("target-aaaa", ["bfo:0000015", "bfo:0000040"])
    a = _coll("ref-a", ["bfo:0000015", "cco:ont00000995"], 20)
    b = _coll("ref-b", ["bfo:0000040"], 15)
    noise = _coll("noise", ["bfo:0000023"], 8)
    profile = SampleProfile("reference", 8, 400, 400, 1)
    picked = select_reference_collections(
        [target, a, b, noise],
        exclude={target.slug},
        required_iris=genus_iris(target),
        profile=profile,
    )
    slugs = {c.slug for c in picked}
    assert target.slug not in slugs
    covered = set().union(*(genus_iris(c) for c in picked))
    assert genus_iris(target) <= covered


def test_select_reference_fails_when_corpus_cannot_cover() -> None:
    target = _coll("t", ["bfo:0000015", "cco:MISSING"])
    other = _coll("o", ["bfo:0000015"])
    profile = SampleProfile("reference", 8, 400, 400, 1)
    with pytest.raises(PairError, match=GURU_IRI):
        select_reference_collections(
            [target, other],
            exclude={target.slug},
            required_iris=genus_iris(target),
            profile=profile,
        )


def test_assert_pair_rejects_same_id() -> None:
    from atelier.sdg.pair import assert_pair

    with pytest.raises(PairError, match=GURU_OVERLAP):
        assert_pair(
            "sdg-corpora/b24ef9f60660_macbook",
            "sdg-corpora/b24ef9f60660_macbook",
        )
