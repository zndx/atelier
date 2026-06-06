"""Step definitions for synthetic data generation BDD scenarios."""

import json
import tempfile
from pathlib import Path

from behave import when, then


@when("I generate synthetic data with {n:d} variants per category")
def step_generate_synth(context, n):
    from atelier.classify.synth import generate_synth_tables
    context.synth_dir = Path(tempfile.mkdtemp())
    context.synth_result = generate_synth_tables(
        context.category_set, context.synth_dir,
        variants_per_category=n, seed=42,
    )


@then("every leaf category should have at least {n:d} columns")
def step_check_leaf_coverage(context, n):
    from atelier.classify.synth_generators import GENERATORS

    # Collect all curated reference labels
    all_reference = {}
    for table in context.synth_result:
        all_reference.update(table.get("reference_labels", {}))

    # Count per category
    counts: dict[str, int] = {}
    for code in all_reference.values():
        counts[code] = counts.get(code, 0) + 1

    # Check every generator-supported leaf has at least n columns
    for code in GENERATORS:
        # Only check codes that exist in the loaded vocabulary
        has_code = any(c.code == code for c in context.category_set.categories)
        if has_code:
            actual = counts.get(code, 0)
            assert actual >= n, (
                f"Category {code} has {actual} columns, expected >= {n}"
            )


@then("the curated reference should map each column to a valid category code")
def step_check_curated_reference_valid(context):
    ref_path = context.synth_dir / "reference_labels.json"
    assert ref_path.exists(), "reference_labels.json not found"

    with open(ref_path) as f:
        reference_labels = json.load(f)

    valid_codes = {c.code for c in context.category_set.categories}
    for col_name, code in reference_labels.items():
        assert code in valid_codes, (
            f"Column {col_name} has invalid code {code}"
        )


@when("I generate synthetic data with seed {seed:d}")
def step_generate_with_seed(context, seed):
    from atelier.classify.synth import generate_synth_tables
    outdir = Path(tempfile.mkdtemp())

    result = generate_synth_tables(
        context.category_set, outdir,
        variants_per_category=5, seed=seed,
    )

    ref_path = outdir / "reference_labels.json"
    with open(ref_path) as f:
        reference_labels = json.load(f)

    if not hasattr(context, "synth_runs"):
        context.synth_runs = []
    context.synth_runs.append(reference_labels)


@when("I generate synthetic data again with seed {seed:d}")
def step_generate_again_with_seed(context, seed):
    step_generate_with_seed(context, seed)


@then("both outputs should be identical")
def step_check_deterministic(context):
    assert len(context.synth_runs) == 2, "Expected 2 synth runs"
    ref1, ref2 = context.synth_runs
    assert ref1 == ref2, "Outputs differ between runs with same seed"
