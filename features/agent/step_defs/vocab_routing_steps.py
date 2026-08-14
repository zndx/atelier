"""Step definitions for vocabulary routing scenarios."""

import tempfile
from pathlib import Path

from behave import given, when, then


# ── Stub helpers ───────────────────────────────────────────────────


class _HiveStub:
    """Records (connection_name, database) call args; returns or raises."""

    def __init__(self, return_value=None, raise_exc=None):
        self.calls = []
        self.return_value = return_value
        self.raise_exc = raise_exc

    def __call__(self, cfg, connection_name, database, *, hierarchical=True):
        self.calls.append({
            "connection_name": connection_name,
            "database": database,
            "hierarchical": hierarchical,
        })
        if self.raise_exc is not None:
            raise self.raise_exc
        return self.return_value


class _UniversalStub:
    """Tracks invocation of the universal-vocabulary fallback."""

    def __init__(self):
        self.called = False

    def __call__(self, *args, **kwargs):
        self.called = True
        from atelier.classify.taxonomy import HierarchicalCategorySet
        return HierarchicalCategorySet(
            name="stub-universal", categories=[], all_categories=[],
        )


def _ensure_cleanups(context):
    if not hasattr(context, "_cleanups"):
        context._cleanups = []


def _patch(context, module, attr, replacement):
    original = getattr(module, attr)
    setattr(module, attr, replacement)
    _ensure_cleanups(context)
    context._cleanups.append(lambda: setattr(module, attr, original))


def _make_stub_cs(name="stub", n=3):
    """Build a fixture CategorySet whose JSON round-trip survives the
    legacy-mode filter in ``_build_category_set_from_records`` (numeric
    leading-digit codes are required when ``parent_code`` is None)."""
    from atelier.classify.taxonomy import (
        HierarchicalCategorySet,
        ReferenceCategory,
    )
    cats = [
        ReferenceCategory(
            code=f"9.9.{i}",
            label=f"Stub {i}",
            embedding_text=f"Stub {i} | test category {i}",
            description=f"Stub category {i}",
            parent_code="9.9",
        )
        for i in range(n)
    ]
    return HierarchicalCategorySet(
        name=f"stub-{name}",
        categories=cats,
        all_categories=cats,
    )


def _install_universal_stub(context):
    """Spy on load_universal_vocabulary so scenarios can assert no fallback.

    The pipeline no longer imports the universal loader at module level —
    the silent-fallback path was excised (fail-loud vocabulary resolution),
    and remaining callers import from ``atelier.classify.taxonomy`` at call
    time.  Patch the taxonomy module so the spy still observes any caller.
    """
    if getattr(context, "universal_stub", None) is not None:
        return
    import atelier.classify.taxonomy as taxonomy_mod
    stub = _UniversalStub()
    _patch(context, taxonomy_mod, "load_universal_vocabulary", stub)
    context.universal_stub = stub


def _build_dir(context):
    """Per-scenario tempdir used as build_dir for _load_vocabulary."""
    if getattr(context, "_vocab_build_tmp", None) is None:
        context._vocab_build_tmp = tempfile.TemporaryDirectory()
        _ensure_cleanups(context)
        context._cleanups.append(context._vocab_build_tmp.cleanup)
    return Path(context._vocab_build_tmp.name)


# ── OOTB-sample scenario ──────────────────────────────────────────


@when('I resolve vocabulary for source "ootb-sample"')
def step_resolve_ootb(context):
    from atelier.classify.taxonomy import load_sample_vocabulary
    context.vocab = load_sample_vocabulary(hierarchical=True)


@then("the vocabulary has {count:d} categories")
def step_vocab_has_n_categories(context, count):
    assert len(context.vocab.categories) == count, (
        f"Expected {count} categories, got {len(context.vocab.categories)}"
    )


@then('all codes start with "ICE."')
def step_all_codes_start_ice(context):
    # The unified tagging vocabulary includes the bare root "ICE" —
    # parents (root included) are taggable alongside leaves.
    for cat in context.vocab.categories:
        assert cat.code == "ICE" or cat.code.startswith("ICE."), (
            f"Code {cat.code} doesn't start with ICE."
        )


# ── Hive source setup ─────────────────────────────────────────────


@given("domain annotations with {count:d} leaf codes")
def step_domain_annotations(context, count):
    from atelier.classify.taxonomy import ReferenceCategory
    context.domain_categories = [
        ReferenceCategory(
            code=f"1.1.{i}",
            label=f"Category {i}",
            embedding_text=f"Category {i} | test category number {i}",
            description=f"Test category number {i}",
            parent_code="1.1" if i > 0 else "1",
        )
        for i in range(count)
    ]


@given('a hive source with vocab_uri "{uri}"')
def step_hive_source_with_uri(context, uri):
    context.vocab_uri = uri


@given("a hive source with no vocab_uri")
def step_hive_source_no_uri(context):
    context.vocab_uri = None


@given("load_annotations_from_hive is stubbed to return a non-empty CategorySet")
def step_stub_hive_returns(context):
    import atelier.classify.pipeline as pipeline_mod
    stub = _HiveStub(return_value=_make_stub_cs(name="domain", n=3))
    _patch(context, pipeline_mod, "load_annotations_from_hive", stub)
    context.hive_stub = stub
    _install_universal_stub(context)


@given('load_annotations_from_hive is stubbed to raise RuntimeError "{message}"')
def step_stub_hive_raises(context, message):
    import atelier.classify.pipeline as pipeline_mod
    stub = _HiveStub(raise_exc=RuntimeError(message))
    _patch(context, pipeline_mod, "load_annotations_from_hive", stub)
    context.hive_stub = stub
    _install_universal_stub(context)


@given("the annotations cache directory is empty")
def step_cache_empty(context):
    build_dir = _build_dir(context)
    cache_dir = build_dir / "data" / "annotations"
    if cache_dir.exists():
        for child in cache_dir.iterdir():
            child.unlink()


# ── Resolution ────────────────────────────────────────────────────


def _resolve_vocab(context, vocab_uri, connection_name=None):
    """Drive _load_vocabulary, capturing result or exception on context."""
    from atelier.classify.pipeline import _load_vocabulary
    from atelier.config import load_config
    cfg = load_config()
    build_dir = _build_dir(context)
    context.last_vocab_uri = vocab_uri
    try:
        context.vocab = _load_vocabulary(
            cfg,
            build_dir,
            connection_name,
            vocab_uri=vocab_uri,
        )
        context.raised = None
    except Exception as exc:  # noqa: BLE001 — tests assert on the type
        context.vocab = None
        context.raised = exc


@when("I resolve vocabulary for the hive source")
def step_resolve_hive(context):
    _resolve_vocab(context, getattr(context, "vocab_uri", None))


@when('I resolve vocabulary for a hive source with vocab_uri "{uri}" on connection "{conn}"')
def step_resolve_hive_explicit(context, uri, conn):
    _resolve_vocab(context, uri, connection_name=conn)


@when("I attempt to resolve vocabulary with vocab_uri")
def step_resolve_with_vocab_uri(context):
    """Legacy step retained for any external callers; routes through _resolve_vocab."""
    _resolve_vocab(context, "nonexistent.table")


@when('I parse the vocab_uri "{uri}"')
def step_parse_vocab_uri(context, uri):
    from atelier.classify.pipeline import _parse_hive_vocab_uri
    try:
        context.parsed = _parse_hive_vocab_uri(uri)
        context.parser_error = None
    except ValueError as exc:
        context.parsed = None
        context.parser_error = exc


# ── Assertions ────────────────────────────────────────────────────


@then("a RuntimeError is raised")
def step_runtime_error_raised(context):
    assert context.raised is not None, "Expected RuntimeError but none was raised"
    assert isinstance(context.raised, RuntimeError), (
        f"Expected RuntimeError, got {type(context.raised).__name__}"
    )


@then('a RuntimeError is raised whose message mentions "{needle}"')
def step_runtime_error_message(context, needle):
    assert isinstance(context.raised, RuntimeError), (
        f"Expected RuntimeError, got {type(context.raised).__name__ if context.raised else None}"
    )
    assert needle in str(context.raised), (
        f"Expected RuntimeError message to mention {needle!r}, got: {context.raised}"
    )


@then('the raised exception\'s __cause__ message contains "{needle}"')
def step_cause_contains(context, needle):
    cause = getattr(context.raised, "__cause__", None)
    assert cause is not None, "Expected raised exception to have a chained __cause__"
    assert needle in str(cause), (
        f"Expected __cause__ message to contain {needle!r}, got: {cause}"
    )


@then('a ValueError is raised mentioning "{needle}"')
def step_value_error_mentions(context, needle):
    assert context.parser_error is not None, (
        f"Expected ValueError, but parser succeeded with {context.parsed!r}"
    )
    assert isinstance(context.parser_error, ValueError), (
        f"Expected ValueError, got {type(context.parser_error).__name__}"
    )
    msg = str(context.parser_error)
    assert needle in msg, (
        f"Expected ValueError message to mention {needle!r}, got: {msg}"
    )


@then('load_annotations_from_hive was called with database "{database}"')
def step_hive_called_with_db(context, database):
    seen = [c["database"] for c in context.hive_stub.calls]
    assert database in seen, (
        f"Expected load_annotations_from_hive to be called with database "
        f"{database!r}; recorded calls: {seen}"
    )


@then('load_annotations_from_hive was not called with database "{database}"')
def step_hive_not_called_with_db(context, database):
    seen = [c["database"] for c in context.hive_stub.calls]
    assert database not in seen, (
        f"Did not expect load_annotations_from_hive to be called with "
        f"database {database!r}; recorded calls: {seen}"
    )


@then("load_annotations_from_hive was called twice")
def step_hive_called_twice(context):
    n = len(context.hive_stub.calls)
    assert n == 2, f"Expected 2 calls to load_annotations_from_hive, got {n}: {context.hive_stub.calls}"


@then("the universal fallback was not used")
def step_universal_not_used(context):
    stub = getattr(context, "universal_stub", None)
    assert stub is not None, (
        "Universal-vocabulary spy not installed — add a hive stub Given step "
        "before this assertion."
    )
    assert not stub.called, "load_universal_vocabulary was called (silent fallback)"


@then("the resolved vocabulary matches the stub's return")
def step_vocab_matches_stub(context):
    expected = context.hive_stub.return_value
    assert context.vocab is not None, "No vocabulary resolved"
    expected_codes = sorted(c.code for c in expected.categories)
    actual_codes = sorted(c.code for c in context.vocab.categories)
    assert actual_codes == expected_codes, (
        f"Resolved vocabulary categories {actual_codes!r} did not match "
        f"stub's return {expected_codes!r}"
    )


@then('the annotations cache contains "{filename}"')
def step_cache_contains(context, filename):
    cache_dir = _build_dir(context) / "data" / "annotations"
    target = cache_dir / filename
    assert target.exists(), (
        f"Expected cache file {target} to exist; cache_dir contents: "
        f"{[p.name for p in cache_dir.iterdir()] if cache_dir.exists() else 'cache_dir missing'}"
    )


# ── Hierarchy / batch sizing (unchanged) ──────────────────────────


@given("domain annotations with hierarchical dot-codes")
def step_hierarchical_domain(context):
    from atelier.classify.taxonomy import ReferenceCategory
    context.domain_categories = [
        ReferenceCategory(code="1", label="Root", embedding_text="Root", parent_code=None),
        ReferenceCategory(code="1.1", label="Level 1", embedding_text="Level 1", parent_code="1"),
        ReferenceCategory(code="1.1.1", label="Leaf A", embedding_text="Leaf A", parent_code="1.1"),
        ReferenceCategory(code="1.1.2", label="Leaf B", embedding_text="Leaf B", parent_code="1.1"),
        ReferenceCategory(code="1.2", label="Level 1b", embedding_text="Level 1b", parent_code="1"),
        ReferenceCategory(code="1.2.1", label="Leaf C", embedding_text="Leaf C", parent_code="1.2"),
    ]


@when("I build a DST frame from the domain vocabulary")
def step_build_frame(context):
    from atelier.classify.taxonomy import HierarchicalCategorySet
    from atelier.classify.belief import FrameOfDiscernment
    cats = context.domain_categories
    leaves = [c for c in cats if not any(
        other.parent_code == c.code for other in cats
    )]
    vocab = HierarchicalCategorySet(name="test", categories=leaves, all_categories=cats)
    context.frame = FrameOfDiscernment(vocab)


@then("internal nodes exist for parent codes")
def step_internal_nodes_exist(context):
    assert len(context.frame.internal_nodes) > 0, "No internal nodes in frame"


@given("a vocabulary with {count:d} categories")
def step_vocab_with_n(context, count):
    context.category_count = count


@then("the estimated safe batch size is less than {limit:d}")
def step_batch_less_than(context, limit):
    from atelier.classify.bootstrap import _estimate_safe_batch_size
    batch = _estimate_safe_batch_size(context.category_count)
    assert batch < limit, f"Expected batch < {limit}, got {batch}"


@then("the estimated safe batch size is {expected:d}")
def step_batch_equals(context, expected):
    from atelier.classify.bootstrap import _estimate_safe_batch_size
    batch = _estimate_safe_batch_size(context.category_count)
    assert batch == expected, f"Expected batch {expected}, got {batch}"
