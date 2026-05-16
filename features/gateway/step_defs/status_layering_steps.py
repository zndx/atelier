# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step defs for features/gateway/status_layering.feature.

The seeder scenarios use an in-memory SQLite AtelierDao rather than
the devenv PostgreSQL so the suite stays tier-0 (no infrastructure
dependency).  The _seed_classify_data_source helper is replayed here
against the ephemeral DAO to exercise the same logic without
requiring the full gateway lifespan to run.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile

from behave import given, when, then


def _patch_env(context, key: str, value: str | None) -> None:
    """Set an env var for the scenario and restore on teardown."""
    if not hasattr(context, "_patched_env"):
        context._patched_env = {}
    if key not in context._patched_env:
        context._patched_env[key] = os.environ.get(key)
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
    if not any(cb.__name__ == "_restore_env" for cb in getattr(context, "_cleanups", [])):
        context.add_cleanup(_restore_env, context)


def _restore_env(context) -> None:
    for key, prior in getattr(context, "_patched_env", {}).items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


# ── cfg-seeded defaults ────────────────────────────────────────


@given('ATELIER_CLASSIFY_CONNECTION is "{value}" in config')
def step_set_conn(context, value):
    _patch_env(context, "ATELIER_CLASSIFY_CONNECTION", value)


@given('ATELIER_CLASSIFY_DATABASE is "{value}" in config')
def step_set_db(context, value):
    _patch_env(context, "ATELIER_CLASSIFY_DATABASE", value)


@when("I inspect the fsm_start resolution source")
def step_inspect_fsm_start(context):
    from atelier.gateway import fsm_start
    context.src = inspect.getsource(fsm_start)


@then("the resolution reads cfg.classify_connection_name as default")
def step_resolution_uses_conn(context):
    assert "classify_connection_name" in context.src, context.src[:400]


@then("the resolution reads cfg.classify_database as default")
def step_resolution_uses_db(context):
    assert "classify_database" in context.src, context.src[:400]


@then("the resolution comment documents the precedence ladder")
def step_resolution_documents_ladder(context):
    assert "precedence" in context.src.lower() or "ladder" in context.src.lower()


# ── load_annotations_from_hive whitelist ───────────────────────


@when('I call load_annotations_from_hive with database "{db}"')
def step_call_hive_loader(context, db):
    from atelier.classify.taxonomy import load_annotations_from_hive
    context.hive_error = None
    try:
        load_annotations_from_hive(None, "conn", db, hierarchical=True)
    except ValueError as exc:
        context.hive_error = exc
    except RuntimeError:
        # Expected when cml.data_v1 isn't on the path (tier-0) — the
        # whitelist check ran first and passed, so cml import happens.
        context.hive_error = None


@then("a ValueError is raised mentioning unsafe")
def step_assert_unsafe_raised(context):
    assert isinstance(context.hive_error, ValueError)
    assert "unsafe" in str(context.hive_error).lower()


@then("no whitelist-violation error is raised before the cml import")
def step_assert_no_whitelist_error(context):
    assert context.hive_error is None


# ── _load_vocabulary env-default Hive fallback ─────────────────


@given(
    '_load_vocabulary receives connection_name "{conn}" and database "{db}" '
    "with no vocab_uri"
)
def step_set_vocab_args(context, conn, db):
    context.conn, context.db = conn, db


@then("the env-default Hive branch is present in the function source")
def step_env_default_branch_present(context):
    from atelier.classify import pipeline
    src = inspect.getsource(pipeline._load_vocabulary)
    # The env-default branch is the (connection_name, database) path
    # that resolves the operator's annotations table.  Cache-aware
    # since the silent-fallback removal: it delegates to
    # _load_domain_annotations, which itself wraps
    # load_annotations_from_hive with a cache check.
    assert "connection_name and database" in src
    assert "_load_domain_annotations" in src


@then("the universal fixture is not silently substituted on resolution failure")
def step_no_silent_universal_fallback(context):
    """The cure for the bel_threshold-2026-05-15T22:42:57Z sweep regression.

    `_load_vocabulary` previously fell through to `load_universal_vocabulary`
    when the env-default Hive load raised — which silently substituted a
    29-leaf generic vocabulary that did not align with any operator's
    domain annotations.  Removing that fallback is a deliberate
    correctness fix; reinstating it would re-open the same hours-of-
    compute-wasted-on-wrong-vocab failure mode.

    Guard the property at the source level: any future caller of
    `load_universal_vocabulary` from inside `_load_vocabulary`'s body
    must justify itself, and this test should be revisited.
    """
    from atelier.classify import pipeline
    src = inspect.getsource(pipeline._load_vocabulary)
    # The docstring may mention `load_universal_vocabulary` to explain
    # why the function deliberately does NOT call it — that's fine.
    # What we forbid is an actual *call*: the substring followed by `(`.
    assert "load_universal_vocabulary(" not in src, (
        "_load_vocabulary must not call load_universal_vocabulary() as a "
        "silent fallback; raise instead so callers see misconfiguration."
    )


# ── Seeder scenarios (SQLite-backed DAO) ───────────────────────


def _fresh_sqlite_dao(context):
    """Build an AtelierDao backed by a temp SQLite file for this scenario.

    The production default carries PG-specific engine kwargs; we bypass
    by setting ATELIER_DB_URL to sqlite: before load_config and passing
    engine_args={} to skip the PG connect_timeout.  Schema is created
    from the SQLAlchemy Base.
    """
    from atelier.db.dao import AtelierDao
    from atelier.db.model import Base
    # Hard-pin SQLite for the scenario so the gateway seeder (which
    # calls AtelierDao() with no args) picks up the same engine URL.
    path = tempfile.mktemp(suffix=".db", prefix="atelier_bdd_")
    _patch_env(context, "ATELIER_DB_URL", f"sqlite:///{path}")
    dao = AtelierDao(engine_url=f"sqlite:///{path}", engine_args={})
    Base.metadata.create_all(dao.engine)
    return dao


@when("_seed_classify_data_source runs against the in-process gateway")
def step_run_seeder(context):
    context.dao = _fresh_sqlite_dao(context)
    from atelier.gateway import _seed_classify_data_source
    _seed_classify_data_source()


@then('a data_source row exists with id "{source_id}"')
def step_assert_row_exists(context, source_id):
    row = context.dao.get_data_source(source_id)
    assert row is not None, f"no row with id {source_id!r}"
    context.row = row


@then('its source_uri is "{source_uri}"')
def step_assert_source_uri(context, source_uri):
    assert context.row["source_uri"] == source_uri, context.row


@then('its vocab_uri is "{vocab_uri}"')
def step_assert_vocab_uri(context, vocab_uri):
    assert context.row["vocab_uri"] == vocab_uri, context.row


@then("its metadata carries seeded_from_env true")
def step_assert_metadata(context):
    meta = json.loads(context.row.get("metadata") or "{}")
    assert meta.get("seeded_from_env") is True, meta


@given(
    'a data_source row with id "{source_id}" and a custom vocab_uri'
)
def step_create_custom_row(context, source_id):
    # Pre-seed with a custom vocab_uri the operator would have set
    # via the UI; the idempotent seeder must not stomp on this.
    context.dao = _fresh_sqlite_dao(context)
    context.dao.get_or_create_data_source(
        source_id=source_id,
        source_type="hive",
        display_name="custom",
        source_uri="test-conn/hive_poc",
        vocabulary_mode="hive",
        vocab_uri="custom.annotations",  # ← operator edit
        metadata=json.dumps({"seeded_from_env": True}),
    )
    context.source_id = source_id
    # Env must still be set so the seeder would try to write.
    _patch_env(context, "ATELIER_CLASSIFY_CONNECTION", "test-conn")
    _patch_env(context, "ATELIER_CLASSIFY_DATABASE", "hive_poc")


@when("_seed_classify_data_source runs a second time")
def step_run_seeder_again(context):
    from atelier.gateway import _seed_classify_data_source
    _seed_classify_data_source()


@then("the row's custom vocab_uri is preserved")
def step_row_custom_preserved(context):
    row = context.dao.get_data_source(context.source_id)
    assert row is not None
    assert row["vocab_uri"] == "custom.annotations", row


# ── Web Terminal provider-routing scenarios ───────────────────
#
# The Web Terminal picker lets the operator override cfg.agent_model
# per-session.  _build_sdk_env must honor the selection when deciding
# whether to set CLAUDE_CODE_USE_BEDROCK — without this, a direct-API
# pick on a CAI deploy (which has both Bedrock creds and an
# ANTHROPIC_API_KEY) still force-routes through Bedrock.

_BEDROCK_ARN = (
    "arn:aws:bedrock:us-east-1:440464140575:inference-profile/"
    "us.anthropic.claude-opus-4-6-v1"
)


@given("a cfg with both Bedrock ARN default and ANTHROPIC_API_KEY")
def step_mixed_cred_cfg(context):
    from atelier.config import AtelierConfig
    context.cfg = AtelierConfig(
        anthropic_api_key="sk-ant-bdd",
        aws_access_key_id="AKIAEXAMPLE",
        aws_secret_access_key="SECRETEXAMPLE",
        aws_region="us-east-1",
        agent_model=_BEDROCK_ARN,
    )


@when('_build_sdk_env is called with selected_model "{model}"')
def step_build_env_with_model(context, model):
    from atelier.agents.client import _build_sdk_env
    context.env = _build_sdk_env(context.cfg, selected_model=model)


@when("_build_sdk_env is called with the Bedrock ARN as selected_model")
def step_build_env_with_arn(context):
    from atelier.agents.client import _build_sdk_env
    context.env = _build_sdk_env(context.cfg, selected_model=_BEDROCK_ARN)


@then("the env dict omits CLAUDE_CODE_USE_BEDROCK")
def step_no_bedrock_flag(context):
    assert "CLAUDE_CODE_USE_BEDROCK" not in context.env, context.env


@then('the env dict sets CLAUDE_CODE_USE_BEDROCK to "{value}"')
def step_bedrock_flag_set(context, value):
    got = context.env.get("CLAUDE_CODE_USE_BEDROCK")
    assert got == value, f"expected {value!r}, got {got!r}; env={context.env}"
