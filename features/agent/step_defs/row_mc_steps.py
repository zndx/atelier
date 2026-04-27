# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for row-level Monte Carlo sampling scenarios."""

from behave import given, when, then


@then("row MC config should load with default values")
def step_row_mc_config_defaults(context):
    from atelier.config import load_config
    from atelier.classify.row_sampler import RowMCConfig

    cfg = load_config()
    row_mc = RowMCConfig.from_cfg(cfg)

    assert row_mc.enabled is False, f"Expected disabled, got {row_mc.enabled}"
    assert row_mc.k == 10, f"Expected k=10, got {row_mc.k}"
    assert row_mc.strategy == "stratified", f"Expected stratified, got {row_mc.strategy}"
    assert row_mc.iterations == 3, f"Expected iterations=3, got {row_mc.iterations}"
    assert row_mc.max_iterations == 5, f"Expected max_iterations=5, got {row_mc.max_iterations}"
    assert row_mc.adaptive_escalation is True


@given("the OOTB sample source is loaded")
def step_load_ootb_sample(context):
    from atelier.classify.sampler import load_sample_source
    context.sample_tables = load_sample_source()
    assert len(context.sample_tables) > 0, "No sample tables loaded"


@then("each column should have all_values with more entries than values")
def step_check_all_values(context):
    for table in context.sample_tables:
        for col in table.columns:
            assert len(col.all_values) >= len(col.values), (
                f"{col.name}: all_values ({len(col.all_values)}) < values ({len(col.values)})"
            )
            # OOTB CSVs have 100 rows; values should be first 5
            assert len(col.values) <= 5, (
                f"{col.name}: values has {len(col.values)} entries, expected <= 5"
            )
            if col.total_count > 5:
                assert len(col.all_values) > len(col.values), (
                    f"{col.name}: all_values should have more entries than values "
                    f"when total_count={col.total_count}"
                )


@given("a column with {n:d} distinct values in its reservoir")
def step_column_with_reservoir(context, n):
    from atelier.classify.sampler import ColumnSample
    context.test_column = ColumnSample(
        name="test_col",
        column_type="VARCHAR",
        values=[f"value_{i}" for i in range(min(5, n))],
        all_values=[f"value_{i}" for i in range(n)],
        total_count=n,
        null_count=0,
        table_name="test_table",
    )


@when("I select stratified row samples for {n_iter:d} iterations with k={k:d}")
def step_select_stratified_samples(context, n_iter, k):
    from atelier.classify.row_sampler import select_row_sample

    context.row_samples = []
    for iteration in range(n_iter):
        # Make a copy of values before selection (since it mutates in-place)
        select_row_sample(
            context.test_column, k=k, strategy="stratified", iteration=iteration,
        )
        context.row_samples.append(list(context.test_column.values))


@then("each iteration should produce a different value subset")
def step_different_subsets(context):
    for i in range(len(context.row_samples)):
        for j in range(i + 1, len(context.row_samples)):
            assert set(context.row_samples[i]) != set(context.row_samples[j]), (
                f"Iterations {i} and {j} produced identical subsets"
            )


@then("each subset should contain {k:d} values")
def step_subset_size(context, k):
    for i, sample in enumerate(context.row_samples):
        assert len(sample) == k, (
            f"Iteration {i}: expected {k} values, got {len(sample)}"
        )


@given("a column with {n:d} values in its reservoir")
def step_column_small_reservoir(context, n):
    from atelier.classify.sampler import ColumnSample
    context.test_column = ColumnSample(
        name="test_col",
        column_type="VARCHAR",
        values=[f"value_{i}" for i in range(n)],
        all_values=[f"value_{i}" for i in range(n)],
        total_count=n,
        null_count=0,
        table_name="test_table",
    )


@when("I select a row sample with k={k:d}")
def step_select_sample_passthrough(context, k):
    from atelier.classify.row_sampler import select_row_sample
    context.selected = select_row_sample(
        context.test_column, k=k, strategy="stratified", iteration=0,
    )


@then("all {n:d} values should be returned unchanged")
def step_passthrough_check(context, n):
    assert len(context.selected) == n, (
        f"Expected {n} values, got {len(context.selected)}"
    )
    assert context.selected == context.test_column.all_values


@given("a bootstrap state with varying labels across row iterations")
def step_create_mixed_state(context):
    from atelier.classify.bootstrap import BootstrapState
    state = BootstrapState()
    # Unstable: different label each iteration (stability = 1/4 = 0.25)
    state.row_labels_history["unstable_col"] = [
        "ICE.A", "ICE.B", "ICE.C", "ICE.D",
    ]
    # Stable: same label every iteration
    state.row_labels_history["stable_col"] = [
        "ICE.X", "ICE.X", "ICE.X", "ICE.X",
    ]
    context.bootstrap_state = state


@then("the row stability for the unstable column should be below {threshold:g}")
def step_unstable_row_stability(context, threshold):
    from atelier.classify.bootstrap import row_stability
    stability = row_stability(context.bootstrap_state, "unstable_col")
    assert stability < threshold, (
        f"unstable_col stability {stability} >= {threshold}"
    )


@then("the row stability for the stable column should be {expected:g}")
def step_stable_row_stability(context, expected):
    from atelier.classify.bootstrap import row_stability
    stability = row_stability(context.bootstrap_state, "stable_col")
    assert abs(stability - expected) < 0.01, (
        f"stable_col stability {stability} != {expected}"
    )
