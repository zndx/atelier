"""Step definitions for the CAI Data Platform unified-surface feature."""

from __future__ import annotations

from pathlib import Path

from behave import given, when, then


@given("the gateway has seeded local filesystem sources")
def step_seed_filesystem_sources(context):
    """Run the gateway's seeders so the DB has the three filesystem rows."""
    from atelier.gateway import (
        _seed_sample_source,
        _seed_synth_source,
        _seed_meta_tagging_source,
    )
    _seed_sample_source()
    _seed_synth_source()
    _seed_meta_tagging_source()


@given("a local filesystem mount with an annotations.csv")
def step_mount_with_annotations(context):
    """Resolve the meta-tagging mount or skip when unavailable."""
    from atelier.classify.meta_tagging_source import resolve_meta_tagging_mount
    mount = resolve_meta_tagging_mount()
    if mount is None:
        context.scenario.skip("No filesystem mount with annotations.csv available")
        return
    context.fs_mount = Path(mount)


@when('I GET "{path}"')
def step_http_get(context, path):
    from fastapi.testclient import TestClient
    from atelier.gateway import app
    client = TestClient(app)
    r = client.get(path)
    context.http_response = r.json()


@when("I load annotations from the filesystem path")
def step_load_annotations(context):
    from atelier.classify.taxonomy import load_annotations_from_filesystem
    context.cs = load_annotations_from_filesystem(context.fs_mount / "annotations.csv")


@when("I resolve vocabulary for a file:// vocab_uri against the mount")
def step_resolve_vocab_uri(context):
    """Exercise pipeline._load_vocabulary's file:// branch directly."""
    from atelier.classify.pipeline import _load_vocabulary
    from atelier.config import load_config
    cfg = load_config()
    build_dir = Path("build")
    vocab_uri = f"file://{(context.fs_mount / 'annotations.csv').resolve()}"
    context.cs = _load_vocabulary(cfg, build_dir, None, vocab_uri=vocab_uri)


@then("the response lists at least one filesystem platform")
def step_has_filesystem_platform(context):
    platforms = context.http_response.get("platforms") or []
    fs = [p for p in platforms if p.get("kind") == "filesystem"]
    assert fs, f"expected >=1 filesystem platform; got {platforms}"


@then('every filesystem platform\'s source_uri starts with "file://"')
def step_filesystem_scheme(context):
    platforms = context.http_response.get("platforms") or []
    bad = [
        p for p in platforms
        if p.get("kind") == "filesystem"
        and not (p.get("source_uri") or "").startswith("file://")
    ]
    assert not bad, f"non-file:// filesystem entries: {bad}"


@then('every hive platform\'s source_uri starts with "hive://"')
def step_hive_scheme(context):
    platforms = context.http_response.get("platforms") or []
    bad = [
        p for p in platforms
        if p.get("kind") == "hive"
        and not (p.get("source_uri") or "").startswith("hive://")
    ]
    assert not bad, f"non-hive:// hive entries: {bad}"


@then("the stats response is ok")
def step_stats_ok(context):
    body = context.http_response
    assert body.get("ok") is True, f"not ok: {body}"


@then("the stats response includes an annotation_count > 0")
def step_stats_annotations(context):
    body = context.http_response
    count = body.get("annotation_count")
    assert isinstance(count, int) and count > 0, (
        f"annotation_count not positive int: {count!r}"
    )


@then('the stats response\'s vocab_uri ends with "annotations.csv"')
def step_stats_vocab_uri_path(context):
    body = context.http_response
    uri = body.get("vocab_uri") or ""
    assert uri.endswith("annotations.csv"), f"unexpected vocab_uri: {uri!r}"


@then("the resulting category set has at least {n:d} categories")
def step_cs_has_n_categories(context, n):
    assert context.cs is not None
    cats = list(context.cs.categories)
    assert len(cats) >= n, f"only {len(cats)} categories (< {n})"


@then("the resolved category set has at least {n:d} categories")
def step_resolved_cs_has_n_categories(context, n):
    step_cs_has_n_categories(context, n)
