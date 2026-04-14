"""Step definitions for SAGE feature importance BDD scenarios."""

import numpy as np
from behave import when, then


@when("I run SAGE with {n:d} permutations on mock data")
def step_run_sage(context, n):
    from atelier.classify.features import extract_features
    from atelier.classify.sage import run_sage_analysis
    from atelier.classify.sampler import load_fixture_samples

    category_set = context.category_set
    code_to_idx = {
        c.code: i for i, c in enumerate(category_set.categories)
    }

    all_features = []
    gt_indices = []
    for ts in load_fixture_samples():
        for col in ts.columns:
            if col.ground_truth and col.ground_truth in code_to_idx:
                features = extract_features(
                    column_name=col.name,
                    column_type=col.column_type,
                    values=col.values,
                    siblings=col.siblings,
                    source_table=col.table_name,
                    total_count=col.total_count,
                    null_count=col.null_count,
                )
                all_features.append(features)
                gt_indices.append(code_to_idx[col.ground_truth])

    context.sage_result = run_sage_analysis(
        all_features=all_features,
        ground_truth_indices=np.array(gt_indices),
        category_set=category_set,
        n_permutations=n,
        detect_convergence=False,
    )


@then("the SAGE result should have {n:d} feature importance values")
def step_sage_count(context, n):
    result = context.sage_result
    assert len(result.importance_values) == n, (
        f"Expected {n} importance values, got {len(result.importance_values)}"
    )
    assert len(result.feature_names) == n


@then("{feature_name} should have non-zero importance")
def step_sage_nonzero(context, feature_name):
    result = context.sage_result
    idx = result.feature_names.index(feature_name)
    val = result.importance_values[idx]
    assert abs(val) > 1e-6, (
        f"Feature {feature_name} has zero importance: {val}"
    )


@then("{feature_name} should be in the top {k:d} by absolute importance")
def step_sage_top_k(context, feature_name, k):
    ranked = context.sage_result.ranked()
    top_names = [name for name, _ in ranked[:k]]
    assert feature_name in top_names, (
        f"{feature_name} not in top {k}: {top_names} "
        f"(full ranking: {ranked})"
    )
