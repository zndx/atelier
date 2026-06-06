"""Step definitions for Monte Carlo sampling BDD scenarios."""

from behave import given, when, then


@given("the fixture corpus with {n:d} columns")
def step_fixture_corpus(context, n):
    from atelier.classify.sampler import load_fixture_samples
    from atelier.classify.taxonomy import load_universal_vocabulary
    from atelier.classify.belief import FrameOfDiscernment

    context.category_set = load_universal_vocabulary(hierarchical=True)
    context.frame = FrameOfDiscernment(context.category_set)
    samples = load_fixture_samples()
    all_cols = {c.name: c for ts in samples for c in ts.columns}
    # Limit to first n columns if needed
    context.mc_column_names = list(all_cols.keys())[:n]
    context.mc_samples = {k: all_cols[k] for k in context.mc_column_names}
    context.mc_total = len(context.mc_column_names)


@given("an MC config with min_corpus_size {threshold:d}")
def step_mc_config_threshold(context, threshold):
    from atelier.classify.monte_carlo import MCConfig
    context.mc_cfg = MCConfig(min_corpus_size=threshold)


@when("I run pre-classification on the corpus")
def step_pre_classify(context):
    from atelier.classify.monte_carlo import pre_classify
    context.pre_results = pre_classify(
        context.mc_column_names,
        context.mc_samples,
        context.category_set,
        context.frame,
        has_embeddings=True,
    )


@when("I stratify the pre-classified columns")
def step_stratify(context):
    from atelier.classify.monte_carlo import stratify
    context.strata = stratify(context.pre_results, context.mc_cfg)


@when("I select the MC sample")
def step_select_sample(context):
    from atelier.classify.monte_carlo import select_sample
    context.mc_plan = select_sample(
        context.strata, context.mc_cfg, total=context.mc_total,
    )


@then("every column should have a pre-classification")
def step_all_pre_classified(context):
    assert len(context.pre_results) == context.mc_total, (
        f"Expected {context.mc_total} pre-classifications, "
        f"got {len(context.pre_results)}"
    )


@then("there should be at least {n:d} strata")
def step_min_strata(context, n):
    assert len(context.strata) >= n, (
        f"Expected >= {n} strata, got {len(context.strata)}"
    )


@then("the MC plan should be passthrough")
def step_plan_passthrough(context):
    assert context.mc_plan.is_passthrough, (
        f"Expected passthrough, got sampled={len(context.mc_plan.sampled_columns)}, "
        f"total={context.mc_plan.total_columns}"
    )


@then("the MC plan should NOT be passthrough")
def step_plan_not_passthrough(context):
    assert not context.mc_plan.is_passthrough, (
        f"Expected active MC, but all {context.mc_plan.total_columns} "
        "columns were sampled (passthrough)"
    )


@then("sampled columns should be a subset of all columns")
def step_sampled_subset(context):
    all_cols = set(context.mc_column_names)
    assert context.mc_plan.sampled_columns.issubset(all_cols), (
        "Sampled columns contain names not in corpus"
    )


@then("sampled + propagation should cover all columns")
def step_full_coverage(context):
    covered = context.mc_plan.sampled_columns | context.mc_plan.propagation_columns
    all_cols = set(context.mc_column_names)
    assert covered == all_cols, (
        f"Coverage gap: {all_cols - covered} columns not in sampled or propagation"
    )


@then("MC config should load from HOCON with default values")
def step_mc_config_defaults(context):
    from atelier.config import load_config
    from atelier.classify.monte_carlo import MCConfig
    cfg = load_config()
    mc = MCConfig.from_cfg(cfg)
    assert mc.min_corpus_size == 200
    # sample_fraction=1.00 is the thesis-aligned default (commit cc68385)
    # — MC passthrough mode where every column goes to the LLM.
    assert mc.sample_fraction == 1.00
    assert mc.min_per_stratum == 3
    assert mc.max_sampled_columns == 500
    assert mc.propagation_threshold == 0.85
    assert mc.propagation_discount == 0.30
