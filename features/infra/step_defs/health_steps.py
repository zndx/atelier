# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for platform health checks."""

import json
import urllib.request

from behave import when, then, given
from pathlib import Path

from features.infra.step_defs.helpers import get_config, pg_engine


# ── PostgreSQL ──────────────────────────────────────────────────


@when("I connect to the atelier database")
def step_connect_pg(context):
    cfg = get_config()
    engine = pg_engine(cfg.db_url)
    from sqlalchemy import text
    with engine.connect() as conn:
        context.pg_connected = True
        # Check extensions
        result = conn.execute(text(
            "SELECT extname FROM pg_extension"
        ))
        context.pg_extensions = [row[0] for row in result]
        # Check tables
        result = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
        ))
        context.pg_tables = [row[0] for row in result]
    engine.dispose()


@then("the connection succeeds")
def step_pg_connected(context):
    assert context.pg_connected, "PostgreSQL connection failed"


@then('extension "{ext}" is loaded')
def step_pg_extension(context, ext):
    assert ext in context.pg_extensions, \
        f"Extension '{ext}' not found. Loaded: {context.pg_extensions}"


@then('table "{table}" exists')
def step_pg_table(context, table):
    assert table in context.pg_tables, \
        f"Table '{table}' not found. Tables: {context.pg_tables}"


# ── Qdrant ──────────────────────────────────────────────────────


@when("I check the Qdrant health endpoint")
def step_qdrant_health(context):
    cfg = get_config()
    url = f"http://{cfg.qdrant_host}:{cfg.qdrant_http_port}/healthz"
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        context.qdrant_status = resp.status
    except Exception as e:
        context.qdrant_status = None
        context.qdrant_error = str(e)


@then("the response status is {status:d}")
def step_response_status(context, status):
    actual = context.qdrant_status
    assert actual == status, \
        f"Expected status {status}, got {actual}"


# ── PGlite ──────────────────────────────────────────────────────


@then('the file "{path}" exists')
def step_file_exists(context, path):
    full = context.project_root / path
    assert full.exists(), f"File not found: {full}"


@given('the file "{path}" exists')
def step_given_file_exists(context, path):
    full = context.project_root / path
    assert full.exists(), f"File not found: {full}"
    context.current_file = full


@then('it declares dependency "{dep}"')
def step_pkg_json_dep(context, dep):
    data = json.loads(context.current_file.read_text())
    deps = data.get("dependencies", {})
    assert dep in deps, \
        f"Dependency '{dep}' not in package.json. Found: {list(deps.keys())}"
