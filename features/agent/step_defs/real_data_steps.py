"""Step definitions for real data classification baseline scenarios."""

import os
from pathlib import Path

from behave import given, when, then


_DEFAULT_DATA_DIR = Path("~/local/tmp/meta-tagging").expanduser()


def _data_dir() -> Path:
    return Path(os.environ.get("ATELIER_REAL_DATA_DIR", str(_DEFAULT_DATA_DIR)))


@given("the real data directory is available")
def step_real_data_available(context):
    d = _data_dir()
    if not d.exists() or not (d / "annotations.csv").exists():
        context.scenario.skip("Real data directory not available")
        return
    context.real_data_dir = d


@when("I parse real CSVs from the data directory")
def step_parse_real_csvs(context):
    from atelier.classify.real_data_loader import load_real_samples, load_real_vocabulary

    context.real_vocab = load_real_vocabulary(context.real_data_dir)
    context.real_samples = load_real_samples(context.real_data_dir)
    context.real_total_cols = sum(len(t.columns) for t in context.real_samples)


@then("at least {n:d} target columns should be extracted")
def step_min_columns(context, n):
    assert context.real_total_cols >= n, (
        f"Only {context.real_total_cols} columns extracted, expected >= {n}"
    )


@then("every ground truth code should exist in the vocabulary")
def step_gt_codes_in_vocab(context):
    all_codes = set(context.real_vocab.all_by_code.keys())
    missing = []
    for ts in context.real_samples:
        for col in ts.columns:
            if col.ground_truth and col.ground_truth not in all_codes:
                missing.append(col.ground_truth)
    assert not missing, (
        f"{len(missing)} ground truth codes not in vocabulary: {missing[:10]}"
    )


@when("I generate template-based synthetic training data")
def step_generate_template_synth(context):
    import tempfile
    from atelier.classify.real_data_loader import (
        extract_value_templates,
        load_real_vocabulary,
    )
    from atelier.classify.synth import generate_synth_tables

    vocab = load_real_vocabulary(context.real_data_dir)
    templates = extract_value_templates(context.real_data_dir)
    out_dir = Path(tempfile.mkdtemp(prefix="atelier_real_synth_"))

    generate_synth_tables(
        vocab,
        out_dir,
        value_templates=templates,
        variants_per_category=10,
        seed=42,
    )

    # Count distinct categories from ground_truth.json
    import json
    gt_path = out_dir / "ground_truth.json"
    assert gt_path.exists(), f"ground_truth.json not found in {out_dir}"
    with open(gt_path) as f:
        gt = json.load(f)
    context.synth_categories = set(gt.values())


@then("at least {n:d} categories should have synthetic columns")
def step_min_synth_categories(context, n):
    actual = len(context.synth_categories)
    assert actual >= n, (
        f"Only {actual} categories covered by synthetic data, expected >= {n}"
    )


@when("I run the real data ML train-eval cycle")
def step_run_real_ml(context):
    import tempfile
    from atelier.classify.train_eval_cycle import run_real_data_eval

    work_dir = Path(tempfile.mkdtemp(prefix="atelier_real_eval_"))
    context.real_eval_result = run_real_data_eval(
        context.real_data_dir,
        mode="ml_only",
        variants_per_category=30,
        seed=42,
        work_dir=work_dir,
    )
    context.real_eval_work_dir = work_dir


@then("at least {n:d} columns should be classified")
def step_min_classified(context, n):
    total = context.real_eval_result["total_columns"]
    assert total >= n, f"Only {total} columns classified, expected >= {n}"


@then("the hierarchical accuracy should exceed {threshold:g}")
def step_hierarchical_accuracy(context, threshold):
    acc = context.real_eval_result["hierarchical_accuracy"]
    assert acc > float(threshold), (
        f"Hierarchical accuracy {acc:.4f} <= {threshold}"
    )


@then("the evaluation report should be written to the work directory")
def step_eval_report_written(context):
    results_dir = Path(context.real_eval_result["results_dir"])
    report_path = results_dir / "evaluation_report.json"
    assert report_path.exists(), f"Evaluation report not found at {report_path}"
