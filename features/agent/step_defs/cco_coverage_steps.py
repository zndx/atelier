"""Step defs for cco_coverage.feature — confirm the test-gittables taxonomy
realizes CCO module coverage (tier-0; committed fixture + pure-Python
epistemic machinery, no stack)."""
from __future__ import annotations

import json
from pathlib import Path

from behave import given, then, when


def _manifest() -> dict:
    import atelier.classify
    p = Path(atelier.classify.__file__).parent / "ontology" / "cco_modules.json"
    return json.loads(p.read_text())


def _fixture_file(name: str) -> dict:
    import atelier.classify
    p = Path(atelier.classify.__file__).parent / "fixtures" / "test-gittables" / name
    return json.loads(p.read_text())


@given("the test-gittables taxonomy is loaded")
def step_load_taxonomy(context):
    from atelier.optimize.svm.fixture import load_fixture_category_set
    context.cs = load_fixture_category_set()
    context.leaves = [c for c in context.cs.all_categories
                      if getattr(c, "cco_module", None)]


@then("the taxonomy data-covers {n:d} of the 11 canonical CCO modules")
def step_data_covers(context, n):
    canonical = {m["code"] for m in _manifest()["modules"]}
    assert len(canonical) == 11, f"manifest must list 11 modules, has {len(canonical)}"
    covered = {c.cco_module for c in context.leaves}
    assert len(covered) == n, \
        f"expected {n} data-covered modules, got {len(covered)}: {sorted(covered)}"
    context.canonical = canonical
    context.covered = covered


@then('the only module without data leaves is "{code}"')
def step_only_uncovered(context, code):
    uncovered = context.canonical - context.covered
    assert uncovered == {code}, f"expected only {code} uncovered, got {sorted(uncovered)}"


@then('"{code}" is positively represented in the CCO manifest, not dropped')
def step_positively_represented(context, code):
    entry = next((m for m in _manifest()["modules"] if m["code"] == code), None)
    assert entry is not None, f"{code} missing from the manifest"
    assert entry.get("applied_status") and entry.get("note"), \
        f"{code} must carry an explicit status + note, not be silently absent"


@then('the "{code}" module has leaves grounded in DBpedia relation properties')
def step_rel_leaves(context, code):
    rel = [c for c in context.leaves if c.cco_module == code]
    assert rel, f"no leaves for module {code}"
    for c in rel:
        src = (c.cco_annotations or {}).get("definition_source", "")
        assert "dbpedia.org/ontology/" in src, f"{c.code} not grounded in a DBpedia IRI"


@then('the relation leaf "{rel_code}" is distinct from the type leaf "{type_code}"')
def step_distinct(context, rel_code, type_code):
    by_code = {c.code: c for c in context.cs.all_categories}
    assert rel_code in by_code and type_code in by_code, \
        f"both {rel_code} (relation) and {type_code} (type) must exist"
    assert by_code[rel_code].cco_module == "REL"
    assert by_code[type_code].cco_module != "REL", \
        "the type-twin must be a different CCO module (entity type, not relation)"


@when("I classify a Quality leaf of the taxonomy")
def step_classify_quality(context):
    from atelier.classify.belief import HierarchicalClassification
    q = next(c for c in context.leaves if c.cco_module == "QUAL")
    context.hc = HierarchicalClassification(category=q, confidence=0.9, evidence="")


@then('it surfaces the semantic absence "{axis}" as "{val}"')
def step_absence(context, axis, val):
    assert context.hc.semantic_absences.get(axis) == val, \
        f"expected {axis}={val}, got {context.hc.semantic_absences}"


@then('the unit absence is grounded in the CCO property "{prop}"')
def step_grounded(context, prop):
    from atelier.classify.cco_annotations import CCO_ANNOTATION_PROPERTIES
    from atelier.classify.semantic_absence import cco_property_for_axis
    assert cco_property_for_axis("unit") == CCO_ANNOTATION_PROPERTIES[prop]


@then('every leaf carries the CCO annotations "{a}" and "{b}"')
def step_annotations(context, a, b):
    for c in context.leaves:
        ann = c.cco_annotations or {}
        assert a in ann and b in ann, f"{c.code} missing {a}/{b}: {ann}"


@then("those annotations key only to canonical ExtendedRelationOntology IRIs")
def step_annotation_iris(context):
    from atelier.classify.cco_annotations import CCO_ANNOTATION_PROPERTIES
    for c in context.leaves:
        assert set(c.cco_annotations or {}) <= set(CCO_ANNOTATION_PROPERTIES), \
            f"{c.code} has non-canonical annotation keys: {c.cco_annotations}"


@given("the Extended Relation module is covered")
def step_rel_covered(context):
    rel = next(m for m in _manifest()["modules"] if m["code"] == "REL")
    assert rel["applied_status"] == "covered", \
        f"REL must be data-covered, got {rel['applied_status']}"


@when("a relational read leaves its relation unresolved")
def step_relational_read(context):
    from atelier.classify.semantic_absence import semantic_absence
    context.rel_absence = semantic_absence("INFO", relational_context=True)


@then('the relation axis is surfaced as "{val}"')
def step_relation_surfaced(context, val):
    assert context.rel_absence.get("relation") == val, \
        f"per-read relation absence must survive REL being covered: {context.rel_absence}"


_VERIFIED_ICE_IRI = {
    "DesignativeICE": "https://www.commoncoreontologies.org/ont00000686",
    "DescriptiveICE": "https://www.commoncoreontologies.org/ont00000853",
    "PrescriptiveICE": "https://www.commoncoreontologies.org/ont00000965",
}


@then('every leaf is an SDG term via "{p1}" or "{p2}"')
def step_sdg_property(context, p1, p2):
    for lf in _fixture_file("taxonomy.json")["leaves"]:
        assert lf.get("sdg_property") in (p1, p2), \
            f"{lf['code']} sdg_property={lf.get('sdg_property')}"
        assert str(lf.get("sdg_term", "")).startswith("sdg:"), \
            f"{lf['code']} not an sdg: term"


@then("every ICE class resolves to a verified CCO IRI")
def step_ice_iri(context):
    for lf in _fixture_file("taxonomy.json")["leaves"]:
        ic = lf.get("ice_class")
        if ic:  # REL (relation) leaves carry no ICE trichotomy class
            assert lf.get("ice_class_iri") == _VERIFIED_ICE_IRI[ic], \
                f"{lf['code']} ICE IRI not the verified CCO IRI"


@then("the fixture emits an SDG requirements artifact for Aegir")
def step_sdg_requirements(context):
    req = _fixture_file("sdg_requirements.json")
    assert req.get("terms"), "sdg_requirements must list proposed SDG terms"
    assert all(t["status"] == "proposed-extension" for t in req["terms"])
