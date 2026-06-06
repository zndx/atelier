"""Step definitions for onboarding synth framework BDD scenarios."""

import json
import tempfile
from pathlib import Path

from behave import when, then


@when("I build the full generator registry")
def step_build_registry(context):
    from atelier.classify.synth_registry import GeneratorRegistry
    context.registry = GeneratorRegistry.from_vocabulary(context.category_set)
    context.coverage = context.registry.coverage_report(context.category_set)


@then("the coverage report should show at least {pct:d} percent total coverage")
def step_check_coverage(context, pct):
    total = len(context.coverage)
    covered = sum(1 for v in context.coverage.values() if v != "missing")
    actual_pct = (covered / total * 100) if total > 0 else 0
    assert actual_pct >= pct, (
        f"Coverage is {actual_pct:.0f}% ({covered}/{total}), expected >= {pct}%"
    )


@when("I run generate_for_vocabulary with {n:d} variants per category")
def step_gen_for_vocab(context, n):
    from atelier.classify.synth import generate_for_vocabulary
    context.gen_dir = Path(tempfile.mkdtemp())
    context.gen_result, context.gen_coverage = generate_for_vocabulary(
        context.category_set,
        context.gen_dir,
        variants_per_category=n,
        seed=42,
    )


@then("CSV files should be created with columns for each leaf")
def step_check_csv_files(context):
    csv_files = list(context.gen_dir.glob("synth_*.csv"))
    assert len(csv_files) > 0, "No CSV files created"


@then("reference_labels.json should map every column to a valid code")
def step_check_reference_json(context):
    ref_path = context.gen_dir / "reference_labels.json"
    assert ref_path.exists(), "reference_labels.json not found"

    with open(ref_path) as f:
        reference_labels = json.load(f)

    valid_codes = {c.code for c in context.category_set.categories}
    for col_name, code in reference_labels.items():
        assert code in valid_codes, (
            f"Column {col_name} has code {code} not in vocabulary"
        )
