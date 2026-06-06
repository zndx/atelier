"""Step definitions for the settings overlay feature."""

from behave import given, when, then


@given("the config overlay is empty")
def step_overlay_empty(context):
    from atelier.config_overlay import clear_overlay
    clear_overlay()
    context.overlay_error = None


def _coerce(value: str):
    """Parse a feature-file literal — bool / number / quoted string."""
    v = value.strip()
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
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


@then("the resulting config has {attr} equal to {expected:d}")
def step_attr_equals_int(context, attr, expected):
    actual = getattr(context.result_config, attr)
    assert actual == expected, f"{attr}: expected {expected!r}, got {actual!r}"


@then('the resulting config has {attr} equal to bool {expected}')
def step_attr_equals_bool(context, attr, expected):
    want = expected.strip().lower() == "true"
    actual = getattr(context.result_config, attr)
    assert actual is want, f"{attr}: expected {want!r}, got {actual!r}"


@then("a ValueError is raised by the overlay")
def step_raised(context):
    assert context.overlay_error is not None, "Expected a ValueError but none was raised"
    assert isinstance(context.overlay_error, ValueError)


@then("every SETTINGS_METADATA entry declares a tab group")
def step_every_entry_has_group(context):
    from atelier.config_overlay import SETTINGS_METADATA
    valid_groups = {"convergence", "evidence", "sampling", "training", "llm_system"}
    bad: list[str] = []
    for key, meta in SETTINGS_METADATA.items():
        group = meta.get("group")
        if group not in valid_groups:
            bad.append(f"{key} → group={group!r}")
    assert not bad, "entries missing a valid tab group:\n  " + "\n  ".join(bad)
