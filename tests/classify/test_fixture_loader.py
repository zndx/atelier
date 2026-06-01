# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""The test-gittables fixture loader activates semantic absence.

Covers the pure-Python part of the hermetic ``--fixture`` mode (no DB, no
encoder): the CCO-rooted taxonomy loads into a HierarchicalCategorySet whose
leaves carry their referent ``cco_module``, so a classification on a Quality
leaf surfaces ``unit: UNRESOLVED``. The encode/fit/promote tail needs the
devenv stack and is exercised by ``just optimize svm --fixture``.
"""
from __future__ import annotations

from atelier.classify.belief import HierarchicalClassification
from atelier.optimize.svm.fixture import (
    _assert_hermetic,
    load_fixture_category_set,
    load_fixture_rows,
)


def test_fixture_is_present_and_hermetic():
    _assert_hermetic()  # raises if a required public asset is missing


def test_taxonomy_loads_with_cco_modules_on_leaves():
    cs = load_fixture_category_set()
    leaves = [c for c in cs.all_categories if getattr(c, "cco_module", None)]
    assert leaves, "fixture leaves must carry a referent cco_module"
    # cco_module is the canonical CODE (matches the manifest), not the label.
    assert {c.cco_module for c in leaves} <= {
        "INFO", "AGENT", "TIME", "QUAL", "GEO", "CUR", "EVENT", "ARTIFACT",
        "FACILITY", "UNIT", "REL",
    }


def test_quality_leaf_classification_surfaces_unit_absence():
    cs = load_fixture_category_set()
    qual = [c for c in cs.all_categories if getattr(c, "cco_module", None) == "QUAL"]
    assert qual, "expected Quality leaves (length/width/...) in the fixture"
    hc = HierarchicalClassification(category=qual[0], confidence=0.9, evidence="")
    assert hc.semantic_absences == {"unit": "UNRESOLVED"}


def test_fit_prerequisites_hold():
    # The category_set must support the NHSVM fit (alphas + root path).
    cs = load_fixture_category_set()
    alphas = cs.compute_nhsvm_alphas()
    assert len(alphas) == len(cs.all_categories)
    any_leaf = next(c for c in cs.all_categories if getattr(c, "cco_module", None))
    path = cs.path_from_root(any_leaf.code)
    assert path[0] == "GT" and path[-1] == any_leaf.code


def test_train_rows_match_row_schema():
    rows = load_fixture_rows()
    assert len(rows) >= 100
    r = rows[0]
    assert r.column and r.code and isinstance(r.sample_values, list)
