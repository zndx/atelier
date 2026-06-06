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
    assert path[0] == "SDG" and path[-1] == any_leaf.code


def test_train_rows_match_row_schema():
    rows = load_fixture_rows()
    assert len(rows) >= 100
    r = rows[0]
    assert r.column and r.code and isinstance(r.sample_values, list)


def test_taxonomy_terms_grounded_in_cco_annotations():
    # Carving into the void: our taxonomy terms (ICEs) satisfy CCO
    # ExtendedRelationOntology annotation properties, grounded from metadata
    # we already have.
    from atelier.classify.cco_annotations import CCO_ANNOTATION_PROPERTIES

    cs = load_fixture_category_set()
    grounded = [c for c in cs.all_categories if getattr(c, "cco_annotations", None)]
    assert grounded, "fixture leaves must carry CCO-grounded annotations"
    c = grounded[0]
    assert "acronym" in c.cco_annotations           # ont00001753 <- mnemonic
    assert "definition_source" in c.cco_annotations  # ont00001754 <- dbpedia IRI
    assert set(c.cco_annotations) <= set(CCO_ANNOTATION_PROPERTIES)
    # has_token_unit stays UNSET — the unit is the semantic absence, not a
    # filled annotation (the still-open part of the Extended Relation module).
    assert "has_token_unit" not in c.cco_annotations


def test_quality_leaf_grounded_annotations_coexist_with_unit_absence():
    cs = load_fixture_category_set()
    q = next(c for c in cs.all_categories if getattr(c, "cco_module", None) == "QUAL")
    hc = HierarchicalClassification(category=q, confidence=0.9, evidence="")
    # Filled annotation properties AND the explicit unit absence, side by side.
    assert "acronym" in (q.cco_annotations or {})
    assert hc.semantic_absences == {"unit": "UNRESOLVED"}
