"""Step definitions for Agent SDK smoke tests."""

from behave import given, when, then


KEYSTONE_AGENTS = [
    ("classifier", "Column Classifier",
     "Zero-shot classification of table columns using LLM reasoning",
     "classifier"),
    ("evidence-fuser", "Evidence Fuser",
     "Combines LLM, embedding, and SVM evidence via Dempster-Shafer fusion",
     "evidence_fuser"),
    ("viz-director", "Visualization Director",
     "Curates embedding projections and manages interactive exploration",
     "visualization_director"),
]


@given("the database is bootstrapped")
def step_db_bootstrapped(context):
    from atelier.db.dao import AtelierDao
    context.dao = AtelierDao()
    if not context.dao.list_agents():
        for aid, name, desc, role in KEYSTONE_AGENTS:
            context.dao.upsert_agent(aid, name, desc, role)


@then("at least {count:d} agents are registered")
def step_agents_registered(context, count):
    agents = context.dao.list_agents()
    assert len(agents) >= count, \
        f"Expected at least {count} agents, got {len(agents)}: {[a['id'] for a in agents]}"


@given("config with overrides")
def step_config_with_overrides(context):
    from atelier.config import AtelierConfig
    # Build a clean config from *only* the table values — no HOCON, no env
    # vars — so credential-matrix scenarios test exact combos in isolation.
    overrides = {}
    for row in context.table:
        overrides[row["field"]] = row["value"]
    context.cfg = AtelierConfig(**overrides)


@then("the config has_bedrock is true")
def step_has_bedrock_true(context):
    assert context.cfg.has_bedrock, "Expected has_bedrock=True"


@then("the config has_bedrock is false")
def step_has_bedrock_false(context):
    assert not context.cfg.has_bedrock, "Expected has_bedrock=False"


@then("the config has_anthropic is true")
def step_has_anthropic_true(context):
    assert context.cfg.has_anthropic, "Expected has_anthropic=True"


@then("the config has_anthropic is false")
def step_has_anthropic_false(context):
    assert not context.cfg.has_anthropic, "Expected has_anthropic=False"


@when("I validate all credentials")
def step_validate_credentials(context):
    from atelier.agents import validate_credentials
    from atelier.config import load_config
    cfg = load_config()
    context.cred_result = validate_credentials(cfg)


@then("at least one provider is valid")
def step_any_provider_valid(context):
    result = context.cred_result
    assert result.get("any_valid"), \
        f"No valid providers: {result}"


@when("I run the SDK smoke test")
def step_run_smoke_test(context):
    from atelier.agents import run_smoke_test
    from atelier.config import load_config
    cfg = load_config()
    context.smoke_result = run_smoke_test(cfg)


@then("the result contains a session_id")
def step_result_has_session_id(context):
    result = context.smoke_result
    assert result.get("success"), \
        f"Smoke test failed: {result.get('error')}"
    assert result.get("session_id"), \
        f"No session_id in result: {result}"
