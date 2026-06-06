"""Step definitions for AMP deployment lifecycle."""

import os

from behave import when, then, given

from features.deployment.step_defs.helpers import (
    PROJECT_ROOT,
    parse_shell_host_logic,
    parse_install_deps_root_dir,
)


# ── AMP metadata ────────────────────────────────────────────────


@when("I parse the AMP metadata")
@given("the AMP metadata is loaded")
def step_parse_amp_metadata(context):
    import yaml
    path = PROJECT_ROOT / ".project-metadata.yaml"
    context.amp_metadata = yaml.safe_load(path.read_text())


@then('it has a "{field}" field')
def step_amp_has_field(context, field):
    assert field in context.amp_metadata, \
        f"Missing field '{field}' in AMP metadata"


@then('it has a "{section}" section')
def step_amp_has_section(context, section):
    assert section in context.amp_metadata, \
        f"Missing section '{section}' in AMP metadata"
    assert context.amp_metadata[section], \
        f"Section '{section}' is empty"


@then('a "{task_type}" task with entity_label "{label}" exists')
def step_amp_task_with_label(context, task_type, label):
    tasks = context.amp_metadata.get("tasks", [])
    found = any(
        t.get("type") == task_type and t.get("entity_label") == label
        for t in tasks
    )
    assert found, \
        f"No {task_type} task with entity_label '{label}' found"


@then('a "{task_type}" task exists')
def step_amp_task_exists(context, task_type):
    tasks = context.amp_metadata.get("tasks", [])
    found = any(t.get("type") == task_type for t in tasks)
    assert found, f"No {task_type} task found"


# ── Application modality (HOST binding) ────────────────────────


@given('CDSW_APP_PORT is set to "{value}"')
def step_set_cdsw_app_port(context, value):
    context.cdsw_app_port = value


@given("CDSW_APP_PORT is not set")
def step_unset_cdsw_app_port(context):
    context.cdsw_app_port = None


@when("I parse bin/start-app.sh for the HOST variable")
def step_parse_host_variable(context):
    context.parsed_host = parse_shell_host_logic(
        "bin/start-app.sh",
        cdsw_app_port=getattr(context, "cdsw_app_port", None),
    )


@then('HOST is "{expected}"')
def step_host_is(context, expected):
    assert context.parsed_host == expected, \
        f"HOST: expected '{expected}', got '{context.parsed_host}'"


# ── Studio modality (IS_COMPOSABLE) ────────────────────────────


@when('I set IS_COMPOSABLE to "{value}"')
def step_set_composable(context, value):
    context.is_composable = value


@when("IS_COMPOSABLE is not set")
def step_unset_composable(context):
    context.is_composable = None


@when("I parse scripts/install_deps.py for root_dir")
def step_parse_root_dir(context):
    context.parsed_root_dir = parse_install_deps_root_dir(
        is_composable=getattr(context, "is_composable", None),
    )


@then('root_dir is "{expected}"')
def step_root_dir_is(context, expected):
    assert context.parsed_root_dir == expected, \
        f"root_dir: expected '{expected}', got '{context.parsed_root_dir}'"
