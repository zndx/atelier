# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for CAI runtime profile validation."""

from behave import when, then, given
from pathlib import Path

from features.deployment.step_defs.helpers import (
    PROJECT_ROOT,
    try_import,
    compile_python,
    file_exists,
    file_is_executable,
)


# ── Import checks ──────────────────────────────────────────────


@when('I import "{module_name}"')
def step_import_module(context, module_name):
    if not hasattr(context, "imports"):
        context.imports = {}
    mod, err = try_import(module_name)
    context.imports[module_name] = (mod, err)
    context.last_import = (module_name, mod, err)


@then("no ImportError is raised")
def step_no_import_error(context):
    for name, (mod, err) in context.imports.items():
        assert err is None, f"Import '{name}' failed: {err}"


@then("atelier.__version__ is defined")
def step_atelier_version(context):
    mod, err = context.imports.get("atelier", (None, None))
    assert mod is not None, "atelier not imported"
    assert hasattr(mod, "__version__"), "atelier.__version__ not defined"
    assert mod.__version__, "atelier.__version__ is empty"


# ── File existence and executability ───────────────────────────


@then('the file "{path}" is executable')
def step_file_is_executable(context, path):
    assert file_is_executable(path), \
        f"File not found or not executable: {path}"


# ── Config checks (reuses platform steps for loading) ──────────


@then("no exception is raised")
def step_no_exception(context):
    # If we got here, the previous step didn't raise
    pass


@then("the config has {field} > {value:d}")
def step_config_gt(context, field, value):
    actual = getattr(context.cfg, field)
    assert actual > value, f"{field}: expected > {value}, got {actual}"


# ── Compile checks ─────────────────────────────────────────────


@when('I compile "{script}" with py_compile')
def step_compile_script(context, script):
    ok, err = compile_python(script)
    context.compile_ok = ok
    context.compile_error = err


@then("no SyntaxError is raised")
def step_no_syntax_error(context):
    assert context.compile_ok, \
        f"Compilation failed: {context.compile_error}"


# ── Migration parsing ──────────────────────────────────────────


@given('migration files exist in "{path}"')
def step_migrations_exist(context, path):
    migrations_dir = PROJECT_ROOT / path
    if not migrations_dir.exists():
        # No migrations yet is OK — skip remaining steps
        context.migrations = []
        return
    context.migrations = sorted(migrations_dir.glob("*.sql"))


@when("I parse each migration for UP/DOWN blocks")
def step_parse_migrations(context):
    from atelier.db.bootstrap import _extract_up_block
    context.migration_up_blocks = {}
    for f in context.migrations:
        content = f.read_text()
        up = _extract_up_block(content)
        context.migration_up_blocks[f.name] = up


@then("every migration has a valid UP block")
def step_all_migrations_have_up(context):
    if not context.migrations:
        # No migrations exist yet — this is fine
        return
    for name, up in context.migration_up_blocks.items():
        assert up is not None, \
            f"Migration '{name}' has no valid -- migrate:up block"
