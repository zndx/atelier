# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Domain-specific step definitions for gateway API endpoint validation."""

from behave import then


# ── Valid FSM states ─────────────────────────────────────────────────

_KNOWN_FSM_STATES = {
    "IDLE", "LOADING_VOCAB", "DISCOVERING", "SAMPLING",
    "GENERATING_SYNTH", "TRAINING", "LLM_SWEEP", "VALIDATING",
    "CLASSIFYING", "FUSING", "EVALUATING", "CONVERGED", "ERROR",
}


# ── Agent assertions ─────────────────────────────────────────────────


@then('every agent should have "{f1}" and "{f2}" and "{f3}"')
def step_agents_have_fields(context, f1, f2, f3):
    agents = context.response_json.get("agents", [])
    for agent in agents:
        for field in (f1, f2, f3):
            assert field in agent, (
                f"Agent {agent.get('id', '?')} missing field '{field}'"
            )


@then('the agent roles should include "{r1}" and "{r2}"')
def step_agent_roles_include(context, r1, r2):
    agents = context.response_json.get("agents", [])
    roles = {a.get("role") for a in agents}
    for role in (r1, r2):
        assert role in roles, (
            f"Role '{role}' not found in agent roles: {roles}"
        )


# ── Skill assertions ─────────────────────────────────────────────────


@then('every skill should have "{f1}" and "{f2}" and "{f3}"')
def step_skills_have_fields(context, f1, f2, f3):
    skills = context.response_json.get("skills", [])
    for skill in skills:
        for field in (f1, f2, f3):
            assert field in skill, (
                f"Skill {skill.get('id', '?')} missing field '{field}'"
            )


# ── Data source assertions ───────────────────────────────────────────


@then('a source with display_name containing "{text}" should exist')
def step_source_name_contains(context, text):
    sources = context.response_json.get("sources", [])
    matches = [
        s for s in sources
        if text.lower() in (s.get("display_name") or "").lower()
    ]
    assert matches, (
        f"No source with display_name containing '{text}'. "
        f"Found: {[s.get('display_name') for s in sources]}"
    )


# ── FSM state assertions ─────────────────────────────────────────────


@then('the response JSON "{key}" should be a known FSM state')
def step_known_fsm_state(context, key):
    state = context.response_json.get(key)
    assert state in _KNOWN_FSM_STATES, (
        f"'{state}' is not a known FSM state. "
        f"Valid: {sorted(_KNOWN_FSM_STATES)}"
    )
