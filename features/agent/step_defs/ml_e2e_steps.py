"""Step definitions for the ML end-to-end training cycle BDD scenarios."""

from behave import when, then


@when("I run the synth-train-eval cycle")
def step_run_cycle(context):
    from atelier.classify.train_eval_cycle import run_synth_train_eval
    context.cycle_result = run_synth_train_eval(
        context.category_set,
        variants_per_category=30,
        seed=42,
        catboost_iterations=500,
    )


@then("the cycle should complete successfully")
def step_cycle_complete(context):
    r = context.cycle_result
    assert r["total_columns"] > 0, "No columns classified"
    assert r["accuracy"] is not None, "No accuracy computed"


@then("both CatBoost and SVM models should be trained")
def step_models_trained(context):
    trained = context.cycle_result["models_trained"]
    assert "catboost" in trained, f"CatBoost not in {trained}"
    assert "svm" in trained, f"SVM not in {trained}"


@then("the pipeline should use at least {n:d} evidence sources")
def step_min_sources(context, n):
    sources = context.cycle_result["evidence_sources_fired"]
    assert len(sources) >= n, (
        f"Only {len(sources)} sources fired: {sources}, expected >= {n}"
    )


@then("the overall accuracy should exceed {threshold:g}")
def step_accuracy_threshold(context, threshold):
    acc = context.cycle_result["accuracy"]
    assert acc > float(threshold), (
        f"Accuracy {acc:.4f} <= {threshold}"
    )


@then("at least {pct:d}% of columns should have CatBoost evidence")
def step_catboost_coverage(context, pct):
    total = context.cycle_result["total_columns"]
    count = context.cycle_result["catboost_evidence_count"]
    actual_pct = (count / total * 100) if total > 0 else 0
    assert actual_pct >= pct, (
        f"CatBoost coverage {actual_pct:.1f}% < {pct}% ({count}/{total})"
    )


@then("at least {pct:d}% of columns should have SVM evidence")
def step_svm_coverage(context, pct):
    total = context.cycle_result["total_columns"]
    count = context.cycle_result["svm_evidence_count"]
    actual_pct = (count / total * 100) if total > 0 else 0
    assert actual_pct >= pct, (
        f"SVM coverage {actual_pct:.1f}% < {pct}% ({count}/{total})"
    )
