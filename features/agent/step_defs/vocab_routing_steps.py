"""Step definitions for vocabulary routing scenarios."""

from behave import given, when, then


@when('I resolve vocabulary for source "ootb-sample"')
def step_resolve_ootb(context):
    from atelier.classify.taxonomy import load_sample_vocabulary
    context.vocab = load_sample_vocabulary(hierarchical=True)


@then("the vocabulary has {count:d} leaves")
def step_vocab_has_n_leaves(context, count):
    assert len(context.vocab.categories) == count, (
        f"Expected {count} leaves, got {len(context.vocab.categories)}"
    )


@then('all leaf codes start with "ICE."')
def step_all_codes_start_ice(context):
    for cat in context.vocab.categories:
        assert cat.code.startswith("ICE."), f"Code {cat.code} doesn't start with ICE."


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


@when("I resolve vocabulary for the hive source")
def step_resolve_hive(context):
    from atelier.classify.taxonomy import HierarchicalCategorySet
    # For tier-0, we don't have a real hive connection.
    # Build the vocab directly from domain_categories.
    cats = getattr(context, "domain_categories", [])
    context.vocab = HierarchicalCategorySet(
        name="test-domain",
        categories=cats,
        all_categories=cats,
    )


@when("I attempt to resolve vocabulary with vocab_uri")
def step_resolve_with_vocab_uri(context):
    import tempfile
    from atelier.classify.pipeline import _load_vocabulary
    from atelier.config import load_config
    from pathlib import Path
    cfg = load_config()
    # Use a temp dir with no cached annotations to ensure the load fails
    with tempfile.TemporaryDirectory() as tmp:
        try:
            _load_vocabulary(cfg, Path(tmp), None, vocab_uri="nonexistent.table")
            context.raised = None
        except RuntimeError as e:
            context.raised = e


@then("a RuntimeError is raised")
def step_runtime_error_raised(context):
    assert context.raised is not None, "Expected RuntimeError but none was raised"
    assert isinstance(context.raised, RuntimeError)


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
