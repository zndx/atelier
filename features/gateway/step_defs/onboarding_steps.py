# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for onboarding flow — archive/unarchive lifecycle."""

import json
import logging
import urllib.request
import urllib.error

from behave import given, when, then

log = logging.getLogger(__name__)


def _gateway_url(context, path):
    from atelier.config import load_config
    if not hasattr(context, "_gateway_port"):
        cfg = load_config()
        context._gateway_port = cfg.gateway_port
    return f"http://localhost:{context._gateway_port}{path}"


@given("all data sources are archived for a clean slate")
def step_archive_all_sources(context):
    """Archive every data source to simulate a fresh installation.

    Stores the list of archived source IDs so teardown can restore them.
    """
    url = _gateway_url(context, "/api/data-sources?include_archived=true")
    resp = urllib.request.urlopen(url, timeout=10)
    body = json.loads(resp.read().decode())
    sources = body.get("sources", [])

    context._archived_source_ids = []
    for source in sources:
        if source.get("is_archived"):
            continue  # already archived
        sid = source["id"]
        archive_url = _gateway_url(context, f"/api/data-sources/{sid}/archive")
        req = urllib.request.Request(archive_url, data=None, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            context._archived_source_ids.append(sid)
            log.info("Archived source: %s (%s)", sid, source.get("display_name"))
        except urllib.error.HTTPError as e:
            log.warning("Failed to archive %s: %s", sid, e)

    # Register cleanup to unarchive everything after the scenario
    context.add_cleanup(_unarchive_all, context)


def _unarchive_all(context):
    """Restore all sources that were archived during setup."""
    for sid in getattr(context, "_archived_source_ids", []):
        url = _gateway_url(context, f"/api/data-sources/{sid}/unarchive")
        req = urllib.request.Request(url, data=None, method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            log.info("Restored source: %s", sid)
        except Exception as e:
            log.warning("Failed to unarchive %s: %s", sid, e)


@then('the response JSON "{key}" should be an empty list')
def step_json_key_empty_list(context, key):
    assert context.response_json is not None, "Response is not JSON"
    val = context.response_json.get(key)
    assert isinstance(val, list), f"'{key}' is not a list: {type(val)}"
    assert len(val) == 0, (
        f"'{key}' has {len(val)} items, expected empty list. "
        f"Items: {val[:3]}..."
    )


@when("I fetch the latest dataset's parquet data")
def step_fetch_latest_parquet(context):
    """GET the parquet data endpoint for the most recently listed dataset."""
    datasets = context.response_json.get("datasets", [])
    assert datasets, "No datasets available to fetch parquet from"
    latest = datasets[0]  # Already sorted desc by created_at
    dataset_id = latest["id"]
    url = _gateway_url(context, f"/api/datasets/{dataset_id}/data")
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        context.parquet_response_status = resp.status
        context.parquet_response_size = len(resp.read())
        log.info(
            "Parquet fetch OK: dataset=%s, size=%d bytes",
            dataset_id, context.parquet_response_size,
        )
    except urllib.error.HTTPError as e:
        context.parquet_response_status = e.code
        context.parquet_response_size = 0
        log.warning("Parquet fetch failed: dataset=%s, status=%d", dataset_id, e.code)


@then("the parquet response status should be {code:d}")
def step_parquet_status(context, code):
    actual = getattr(context, "parquet_response_status", None)
    assert actual == code, (
        f"Parquet response status is {actual}, expected {code}"
    )
