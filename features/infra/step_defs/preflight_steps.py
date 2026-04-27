# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for preflight validation."""

from pathlib import Path

from behave import given, when, then

from atelier.config import AtelierConfig
from atelier.preflight import run_preflight


@when("I run preflight checks")
def step_run_preflight(context):
    from atelier.config import load_config
    context.preflight_result = run_preflight(load_config())


@when("I run preflight with a nonexistent config path")
def step_run_preflight_missing(context):
    from atelier.config import load_config
    cfg = load_config()
    context.preflight_result = run_preflight(
        cfg, env_path=Path("/tmp/nonexistent_atelier_config.env"),
    )


@given("config with no LLM credentials")
def step_config_no_creds(context):
    context.cfg_no_creds = AtelierConfig()  # defaults, no API keys


@when("I run preflight checks against that config")
def step_run_preflight_custom(context):
    context.preflight_result = run_preflight(context.cfg_no_creds)


@then("the result has no denies")
def step_no_denies(context):
    denies = context.preflight_result.denies
    assert not denies, f"Expected no denies, got: {[d.name for d in denies]}"


@then('the result has a deny for "{check_name}"')
def step_has_deny(context, check_name):
    deny_names = [c.name for c in context.preflight_result.denies]
    assert check_name in deny_names, \
        f"Expected deny '{check_name}', got: {deny_names}"


@then('the result has a warning for "{check_name}"')
def step_has_warning(context, check_name):
    warn_names = [c.name for c in context.preflight_result.warnings]
    assert check_name in warn_names, \
        f"Expected warning '{check_name}', got: {warn_names}"
