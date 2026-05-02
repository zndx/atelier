# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for OOTB synth source feature.

Fabricates minimal synth corpora in a per-scenario temp dir and patches
``resolve_synth_mount``'s path constants so the scenarios exercise the
real resolver logic without depending on build/ artifacts.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

from behave import given, when, then


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _make_dir_mount(base: Path, table_count: int) -> Path:
    """Build an uncompressed synth dir under ``base``. Returns mount path."""
    tables = base / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    reference_labels: dict[str, str] = {}
    for i in range(table_count):
        tname = f"table_{i}"
        header = ["id", "email", "note"]
        rows = [
            [f"{i}-{j}", f"u{j}@example.com", f"row {j}"]
            for j in range(3)
        ]
        _write_csv(tables / f"{tname}.csv", header, rows)
        reference_labels[f"{tname}.email"] = "ICE.SENSITIVE.PID.EMAIL"
        reference_labels[f"{tname}.id"] = "ICE.NONSENSITIVE.DESIGNATIVE.IDENTIFIER"
    (base / "reference_labels.json").write_text(json.dumps(reference_labels))
    return base


def _make_zip_mount(zip_path: Path, table_count: int) -> Path:
    """Build a synth zip at ``zip_path``. Returns the zip path."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    reference_labels: dict[str, str] = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i in range(table_count):
            tname = f"ztable_{i}"
            header = ["id", "ssn", "note"]
            rows = [
                [f"{i}-{j}", f"{100 + j:03d}-{10 + i}-{1000 + j}", f"row {j}"]
                for j in range(3)
            ]
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(header)
            w.writerows(rows)
            zf.writestr(f"data/synth/tables/{tname}.csv", buf.getvalue())
            reference_labels[f"{tname}.ssn"] = "ICE.SENSITIVE.PID.IDENTITY.SSN"
            reference_labels[f"{tname}.id"] = "ICE.NONSENSITIVE.DESIGNATIVE.IDENTIFIER"
        zf.writestr("data/synth/reference_labels.json", json.dumps(reference_labels))
    return zip_path


def _install_mount_paths(context, dir_path: Path | None, zip_path: Path | None):
    """Patch sampler._SYNTH_DIR / _SYNTH_ZIP so resolve_synth_mount() uses fakes."""
    from atelier.classify import sampler
    patches = [
        mock.patch.object(sampler, "_SYNTH_DIR", dir_path or Path("/nonexistent/dir")),
        mock.patch.object(sampler, "_SYNTH_ZIP", zip_path or Path("/nonexistent.zip")),
    ]
    for p in patches:
        p.start()
        context.synth_patches.append(p)


def _ensure_scenario_state(context):
    if getattr(context, "synth_tmpdir", None) is not None:
        return
    context.synth_tmpdir = tempfile.TemporaryDirectory()
    context.synth_patches = []
    context.mount_dir = None
    context.mount_zip = None

    if not hasattr(context, "_cleanups"):
        context._cleanups = []

    def _cleanup():
        for p in context.synth_patches:
            try:
                p.stop()
            except Exception:
                pass
        context.synth_tmpdir.cleanup()
        context.synth_tmpdir = None
        context.synth_patches = []
        context.mount_dir = None
        context.mount_zip = None

    context._cleanups.append(_cleanup)


# ── Givens ─────────────────────────────────────────────────────────


@given("a synth directory mount with {n:d} tables")
def step_dir_mount(context, n):
    _ensure_scenario_state(context)
    base = Path(context.synth_tmpdir.name) / "dir-mount"
    context.mount_dir = _make_dir_mount(base, n)
    # Refresh patches so both paths reflect current state
    for p in context.synth_patches:
        p.stop()
    context.synth_patches = []
    _install_mount_paths(context, context.mount_dir, context.mount_zip)


@given("a synth zip mount with {n:d} tables")
@given("only a synth zip mount with {n:d} tables")
def step_zip_mount(context, n):
    _ensure_scenario_state(context)
    zip_path = Path(context.synth_tmpdir.name) / "zip-mount.zip"
    context.mount_zip = _make_zip_mount(zip_path, n)
    for p in context.synth_patches:
        p.stop()
    context.synth_patches = []
    _install_mount_paths(context, context.mount_dir, context.mount_zip)


@given("no synth artifacts present")
def step_no_artifacts(context):
    _ensure_scenario_state(context)
    _install_mount_paths(context, None, None)


# ── Whens ──────────────────────────────────────────────────────────


@when("I resolve the synth mount")
def step_resolve(context):
    from atelier.classify.sampler import resolve_synth_mount
    context.resolved = resolve_synth_mount()


@when("I load the synth source from that mount")
def step_load(context):
    from atelier.classify.sampler import load_synth_source, resolve_synth_mount
    context.resolved = resolve_synth_mount()
    context.samples = load_synth_source()


@when("I read synth source stats from that mount")
def step_stats(context):
    from atelier.classify.sampler import synth_source_stats
    context.stats = synth_source_stats()


# ── Thens ──────────────────────────────────────────────────────────


@then("the resolver returns the directory path")
def step_returns_dir(context):
    assert context.resolved == context.mount_dir, (
        f"expected {context.mount_dir}, got {context.resolved}"
    )


@then("the resolver returns the zip path")
def step_returns_zip(context):
    assert context.resolved == context.mount_zip, (
        f"expected {context.mount_zip}, got {context.resolved}"
    )


@then("the resolver returns None")
def step_returns_none(context):
    assert context.resolved is None, f"expected None, got {context.resolved}"


@then("I get {n:d} tables")
def step_n_tables(context, n):
    assert len(context.samples) == n, (
        f"expected {n} tables, got {len(context.samples)}"
    )


@then("each column has its curated reference attached")
def step_reference_attached(context):
    hits = sum(
        1 for t in context.samples for c in t.columns if c.reference_code
    )
    assert hits > 0, "no columns had curated reference mapped"


@then("every column belongs to its parent table")
def step_columns_belong(context):
    for t in context.samples:
        for c in t.columns:
            assert c.table_name == t.name, (
                f"column {c.name} has table_name={c.table_name!r}, expected {t.name!r}"
            )


@then('the stats report mount kind "{kind}"')
def step_mount_kind(context, kind):
    assert context.stats.get("mount_kind") == kind, (
        f"expected {kind}, got {context.stats.get('mount_kind')}"
    )


@then("the stats report {n:d} tables")
def step_stats_tables(context, n):
    assert context.stats["table_count"] == n, (
        f"expected {n} tables, got {context.stats['table_count']}"
    )
