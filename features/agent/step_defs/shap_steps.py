# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for SHAP per-item explanation BDD scenarios."""

from behave import when, then


# ── CatBoost TreeSHAP via synth-train-eval ──────────────────────


@when("I run the synth-train-eval cycle with SHAP enabled")
def step_run_synth_train_eval_with_shap(context):
    from atelier.classify.train_eval_cycle import run_synth_train_eval
    from atelier.classify.shap_explanations import run_catboost_shap
    from atelier.classify.embedding import embed_texts
    from atelier.classify import ml_inference
    import numpy as np
    import tempfile
    from pathlib import Path

    category_set = context.category_set
    work_dir = Path(tempfile.mkdtemp(prefix="atelier_shap_"))

    # Run synth-train-eval to get trained models
    cycle_result = run_synth_train_eval(
        category_set, seed=42,
        catboost_iterations=500,
        work_dir=work_dir,
    )

    # Now load the freshly trained CatBoost model and run TreeSHAP
    ml_inference.reset()
    ml_inference.configure_paths(
        catboost_path=work_dir / "models" / "catboost.cbm",
    )

    try:
        from atelier.classify.ml_inference import get_catboost
        from atelier.classify.sampler import load_fixture_samples
        from atelier.classify.features import extract_features

        cb = get_catboost()
        assert cb is not None, "CatBoost model not loaded"

        # Build feature matrix for mock data
        all_features = []
        for ts in load_fixture_samples():
            for col in ts.columns:
                all_features.append(extract_features(
                    column_name=col.name,
                    column_type=col.column_type,
                    values=col.values,
                    siblings=col.siblings,
                    source_table=col.table_name,
                    total_count=col.total_count,
                    null_count=col.null_count,
                ))

        texts = [f.to_embedding_text() for f in all_features]
        X_eval = np.array(embed_texts(texts))

        # Get predictions
        proba_list = [cb.predict_proba_single(X_eval[i]) for i in range(len(all_features))]
        classes = cb._classes
        predicted_indices = np.array([
            classes.index(max(p, key=p.get)) if p else 0
            for p in proba_list
        ])

        shap_result = run_catboost_shap(
            cb._model, X_eval, predicted_indices,
            emb_dim=X_eval.shape[1],
        )

        # Store results as classification-like dicts with SHAP columns
        shap_records = shap_result.to_records(k=3)
        context.shap_classifications = shap_records
        context.shap_result = shap_result
    finally:
        ml_inference.reset()


@then("each classification should have shap_top1_name and shap_top1_value")
def step_check_shap_columns(context):
    for i, rec in enumerate(context.shap_classifications):
        assert "shap_top1_name" in rec, f"Item {i} missing shap_top1_name"
        assert "shap_top1_value" in rec, f"Item {i} missing shap_top1_value"


@then("shap values should be non-zero for at least {pct:d}% of columns")
def step_check_shap_nonzero(context, pct):
    total = len(context.shap_classifications)
    nonzero = sum(
        1 for rec in context.shap_classifications
        if abs(rec.get("shap_top1_value", 0.0)) > 1e-9
    )
    actual_pct = nonzero / total * 100 if total else 0
    assert actual_pct >= pct, (
        f"Only {actual_pct:.0f}% of columns have non-zero SHAP values, expected >= {pct}%"
    )


# ── Embedding PermutationSHAP ────────────────────────────────────


@when("I run SHAP analysis with embedding permutation method on mock data")
def step_run_embedding_shap(context):
    from atelier.classify.shap_explanations import run_embedding_shap
    from atelier.classify.sampler import load_fixture_samples
    from atelier.classify.features import extract_features

    category_set = context.category_set
    all_features = []
    for ts in load_fixture_samples():
        for col in ts.columns:
            all_features.append(extract_features(
                column_name=col.name,
                column_type=col.column_type,
                values=col.values,
                siblings=col.siblings,
                source_table=col.table_name,
                total_count=col.total_count,
                null_count=col.null_count,
            ))

    context.shap_result = run_embedding_shap(
        all_features, category_set, n_permutations=8,
    )
    context.shap_all_features = all_features


@then("the SHAP result should have explanations for all classified columns")
def step_check_shap_count(context):
    expected = len(context.shap_all_features)
    assert context.shap_result.n_items == expected, (
        f"SHAP has {context.shap_result.n_items} items, expected {expected}"
    )


@then("each item should have top-3 feature attributions")
def step_check_top3(context):
    for i in range(context.shap_result.n_items):
        top = context.shap_result.top_features(i, k=3)
        assert len(top) == 3, f"Item {i} has {len(top)} top features, expected 3"


@then("column_name or sample_values should appear in most top-3 lists")
def step_check_common_features(context):
    appearances = 0
    for i in range(context.shap_result.n_items):
        top = context.shap_result.top_features(i, k=3)
        names = {name for name, _ in top}
        if "column_name" in names or "sample_values" in names:
            appearances += 1

    ratio = appearances / context.shap_result.n_items if context.shap_result.n_items else 0
    # "most" = > 50%
    assert ratio > 0.5, (
        f"column_name/sample_values in top-3 for only {ratio:.0%} of items"
    )
