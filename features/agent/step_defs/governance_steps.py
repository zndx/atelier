"""Step definitions for governance SDK integration scenarios."""

from behave import given, when, then


@given('an AtelierConfig with Atlas URL "{url}"')
def step_config_with_atlas(context, url):
    from atelier.config import load_config
    cfg = load_config()
    cfg.governance_atlas_url = url
    cfg.governance_atlas_username = "admin"
    cfg.governance_atlas_password = "secret"
    context.cfg = cfg


@given("an AtelierConfig with no Atlas URL")
def step_config_no_atlas(context):
    from atelier.config import load_config
    cfg = load_config()
    cfg.governance_atlas_url = ""
    context.cfg = cfg


@when("I create a GovernanceClient from config")
def step_create_gc(context):
    from atelier.governance.client import GovernanceClient
    context.gc = GovernanceClient.from_atelier_config(context.cfg)


@then('the Atlas client base URL contains "api/atlas/v2"')
def step_atlas_url_contains(context):
    url = context.gc.atlas.base_api_url
    assert "api/atlas/v2" in url, f"Expected api/atlas/v2 in {url}"


@then("accessing atlas raises RuntimeError")
def step_atlas_raises(context):
    import pytest
    with pytest.raises(RuntimeError, match="AtlasClient not configured"):
        _ = context.gc.atlas


@given("a ClassificationTag with confidence {conf:g}")
def step_tag_with_confidence(context, conf):
    from atelier.governance.atlas import ClassificationTag
    context.tag = ClassificationTag(type_name="PII", confidence=conf)


@then('the Atlas payload confidence attribute is "{expected}"')
def step_tag_payload_confidence(context, expected):
    payload = context.tag.to_atlas_payload()
    actual = payload["attributes"]["confidence"]
    assert actual == expected, f"Expected {expected}, got {actual}"


@given('a SyncResult with entity_name "{name}" and entity_guid "{guid}"')
def step_sync_result(context, name, guid):
    from atelier.governance.atlas import SyncResult
    context.sr = SyncResult(entity_name=name, entity_guid=guid, tags=["PII"], status="success")


@then('entity_guid is "{expected}"')
def step_check_guid(context, expected):
    assert context.sr.entity_guid == expected


@then('entity_name is "{expected}"')
def step_check_name(context, expected):
    assert context.sr.entity_name == expected


@given('a TaxonomyNode "{code}" with parent "{parent}"')
def step_taxonomy_node(context, code, parent):
    from atelier.governance.sync import TaxonomyNode
    context.node = TaxonomyNode(code=code, label=code, parent_code=parent)


@then('the ClassificationTag has super_types ["{expected}"]')
def step_tag_super_types(context, expected):
    tag = context.node.to_classification_tag()
    assert tag.super_types == [expected], f"Got {tag.super_types}"


@given('qualified name for "{table}" cluster "{cluster}"')
def step_qualified_name(context, table, cluster):
    from atelier.governance.atlas import QualifiedName
    context.qn = QualifiedName.from_table_name(table, cluster=cluster)


@then('table qualified name is "{expected}"')
def step_table_qn(context, expected):
    assert context.qn.for_table() == expected


@then('column qualified name for "{col}" is "{expected}"')
def step_column_qn(context, col, expected):
    assert context.qn.for_column(col) == expected
