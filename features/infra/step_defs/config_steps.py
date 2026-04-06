"""Step definitions for pipeline configuration lifecycle."""

import tempfile
from pathlib import Path

from behave import given, when, then

from atelier.config import (
    load_config,
    materialize_config,
    validate_materialized_config,
)


@when("I load the config with no overrides")
@given("I load the config with no overrides")
def step_load_config_defaults(context):
    context.cfg = load_config()


@when("I load the config with overrides")
def step_load_config_overrides(context):
    overrides = {}
    for row in context.table:
        field = row["field"]
        value = row["value"]
        # Coerce numeric strings
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        overrides[field] = value
    context.cfg = load_config(overrides=overrides)


@then('the config has {field} = {value:d}')
def step_config_int_field(context, field, value):
    actual = getattr(context.cfg, field)
    assert actual == value, f"{field}: expected {value}, got {actual}"


@then('the config has {field} starting with "{prefix}"')
def step_config_field_startswith(context, field, prefix):
    actual = getattr(context.cfg, field)
    assert actual.startswith(prefix), \
        f"{field}: expected to start with '{prefix}', got '{actual}'"


@when("I materialize the config to a temporary path")
def step_materialize_config(context):
    tmp = tempfile.NamedTemporaryFile(suffix=".env", delete=False)
    tmp.close()
    context.materialized_path = tmp.name
    if not hasattr(context, "_temp_files"):
        context._temp_files = []
    context._temp_files.append(tmp.name)
    materialize_config(context.cfg, tmp.name)


@when("I validate the materialized config")
def step_validate_config(context):
    context.validation_errors = validate_materialized_config(
        context.materialized_path
    )


@then("validation returns no errors")
def step_no_validation_errors(context):
    errors = context.validation_errors
    assert not errors, f"Validation errors: {errors}"
