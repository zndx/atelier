"""Step definitions for the settings overlay feature."""

from behave import given, when, then


@given("the config overlay is empty")
def step_overlay_empty(context):
    from atelier.config_overlay import clear_overlay
    clear_overlay()
    context.overlay_error = None


def _coerce(value: str):
    """Parse a feature-file literal — number if numeric, else stripped string."""
    v = value.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


@when('I set overlay "{key}" to {value}')
def step_set_overlay(context, key, value):
    from atelier.config_overlay import set_overlay
    parsed = _coerce(value)
    try:
        set_overlay({key: parsed})
        context.overlay_error = None
    except ValueError as exc:
        context.overlay_error = exc


@when("I clear the overlay")
def step_clear_overlay(context):
    from atelier.config_overlay import clear_overlay
    clear_overlay()


@when("I apply the overlay to a loaded config")
def step_apply_overlay(context):
    from atelier.config import load_config
    from atelier.config_overlay import apply_to_config
    base = load_config()
    context.base_config = base
    context.result_config = apply_to_config(base)


@then("the resulting config is unchanged")
def step_unchanged(context):
    assert context.result_config is context.base_config, (
        "Empty overlay should short-circuit (return the same object)"
    )


@then('the resulting config has {attr} equal to "{expected}"')
def step_attr_equals_str(context, attr, expected):
    actual = getattr(context.result_config, attr)
    assert actual == expected, f"{attr}: expected {expected!r}, got {actual!r}"


@then("the resulting config has {attr} equal to {expected:f}")
def step_attr_equals_float(context, attr, expected):
    actual = float(getattr(context.result_config, attr))
    assert abs(actual - expected) < 1e-9, (
        f"{attr}: expected {expected}, got {actual}"
    )


@then("a ValueError is raised by the overlay")
def step_raised(context):
    assert context.overlay_error is not None, "Expected a ValueError but none was raised"
    assert isinstance(context.overlay_error, ValueError)
