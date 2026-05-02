# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for the adaptive Settings focus feature."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from behave import given, when, then


def _ensure_snapshot(context) -> None:
    """Lazily create a snapshot payload keyed on a per-scenario temp dir."""
    if getattr(context, "_focus_tmp", None) is None:
        context._focus_tmp = tempfile.TemporaryDirectory()
        context._focus_dir = Path(context._focus_tmp.name)
        context._focus_resolved = {}
        context._focus_defaults = {}
        context._focus_skip_snapshot = False
        context._focus_result = None
        # Register cleanup — the after_scenario hook drains _cleanups.
        # `getattr(..., [])` returns a throwaway empty list if the
        # attribute isn't set yet, silently dropping the cleanup, so
        # initialize explicitly.
        if not hasattr(context, "_cleanups"):
            context._cleanups = []
        context._cleanups.append(context._focus_tmp.cleanup)
        # Also reset _focus_tmp → None after the scenario so the next
        # one rebuilds from a clean slate.
        tmp = context  # close-over
        def _reset() -> None:
            tmp._focus_tmp = None
            tmp._focus_dir = None
            tmp._focus_resolved = {}
            tmp._focus_defaults = {}
            tmp._focus_skip_snapshot = False
            tmp._focus_result = None
        context._cleanups.append(_reset)


def _write_snapshot(context) -> None:
    _ensure_snapshot(context)
    if not context._focus_resolved:
        # Empty snapshot → use metadata defaults directly
        from atelier.config_overlay import SETTINGS_METADATA
        for key, meta in SETTINGS_METADATA.items():
            context._focus_resolved[key] = meta.get("default")
            context._focus_defaults[key] = meta.get("default")
    snap = {
        "run_id": "test",
        "timestamp": "2026-04-17T00:00:00+00:00",
        "overlay_at_start": {},
        "resolved_values": context._focus_resolved,
        "default_values": context._focus_defaults,
    }
    (context._focus_dir / "settings_snapshot.json").write_text(json.dumps(snap))


def _set_resolved(context, key, current, default):
    _ensure_snapshot(context)
    from atelier.config_overlay import SETTINGS_METADATA
    if not context._focus_resolved:
        for k, meta in SETTINGS_METADATA.items():
            context._focus_resolved[k] = meta.get("default")
            context._focus_defaults[k] = meta.get("default")
    context._focus_resolved[key] = current
    context._focus_defaults[key] = default


def _parse_literal(s: str):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s


@given('a settings snapshot where {key} is {current} (default {default})')
def step_snapshot_drift(context, key, current, default):
    _set_resolved(context, key, _parse_literal(current), _parse_literal(default))


@given('the overlay resolves {key} to {current} (default {default})')
def step_overlay_resolves(context, key, current, default):
    _set_resolved(context, key, _parse_literal(current), _parse_literal(default))


@given("a settings snapshot at defaults")
def step_snapshot_defaults(context):
    _ensure_snapshot(context)
    # _write_snapshot will populate from SETTINGS_METADATA defaults


@given('an overwatch report with focus keys "{keys}"')
def step_overwatch_focus(context, keys):
    _ensure_snapshot(context)
    key_list = [k.strip() for k in keys.split(",")]
    payload = {"focus_keys": key_list}
    md = (
        "# Overwatch report\n\n"
        "Some prose.\n\n"
        "```focus\n" + json.dumps(payload) + "\n```\n"
    )
    (context._focus_dir / "overwatch.md").write_text(md)


@given("an overwatch report with a malformed focus block")
def step_overwatch_malformed(context):
    _ensure_snapshot(context)
    md = (
        "# Overwatch report\n\n"
        "```focus\n"
        "{this is not json\n"
        "```\n"
    )
    (context._focus_dir / "overwatch.md").write_text(md)


@given("a results directory with no settings_snapshot.json")
def step_no_snapshot(context):
    _ensure_snapshot(context)
    # Deliberately skip _write_snapshot call later
    context._focus_skip_snapshot = True


@when("I compute focus for that run")
def step_compute_focus(context):
    from atelier.classify.focus import compute_focus
    if not getattr(context, "_focus_skip_snapshot", False):
        _write_snapshot(context)
    context._focus_result = compute_focus("test", context._focus_dir)


@then('the focus source is "{source}"')
def step_focus_source(context, source):
    actual = context._focus_result.get("source")
    assert actual == source, f"source: expected {source!r}, got {actual!r}"


@then('the deterministic focus includes "{key}"')
def step_determ_includes(context, key):
    keys = context._focus_result.get("deterministic") or []
    assert key in keys, f"deterministic={keys} missing {key!r}"


@then('the deterministic focus does not include "{key}"')
def step_determ_excludes(context, key):
    keys = context._focus_result.get("deterministic") or []
    assert key not in keys, f"deterministic={keys} unexpectedly includes {key!r}"


@then('the overwatch focus includes "{key}"')
def step_overwatch_includes(context, key):
    keys = context._focus_result.get("from_overwatch") or []
    assert key in keys, f"from_overwatch={keys} missing {key!r}"


@then('the overwatch focus does not include "{key}"')
def step_overwatch_excludes(context, key):
    keys = context._focus_result.get("from_overwatch") or []
    assert key not in keys, f"from_overwatch={keys} unexpectedly includes {key!r}"


@then("the overwatch focus is empty")
def step_overwatch_empty(context):
    keys = context._focus_result.get("from_overwatch") or []
    assert keys == [], f"expected empty from_overwatch, got {keys}"


@then('the merged focus includes "{key}"')
def step_merged_includes(context, key):
    keys = context._focus_result.get("focus_keys") or []
    assert key in keys, f"focus_keys={keys} missing {key!r}"


@then("the merged focus is empty")
def step_merged_empty(context):
    keys = context._focus_result.get("focus_keys") or []
    assert keys == [], f"expected empty focus_keys, got {keys}"
