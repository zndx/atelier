# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""GPU SAGE parity / sanity step definitions (requires CUDA)."""

from __future__ import annotations

from behave import given, when, then


@given("the OOTB sample corpus loaded as ColumnFeatures")
def step_load_ootb(context):
    from atelier.classify.embedding import configure, _get_model
    from atelier.classify.features import extract_features
    from atelier.classify.sampler import load_sample_source
    from atelier.classify.taxonomy import load_sample_vocabulary

    configure(device="auto", batch_size=32, shard_threshold=200_000)
    _ = _get_model().encode(["warm"], normalize_embeddings=True)

    samples = load_sample_source()
    vocab = load_sample_vocabulary(hierarchical=True)
    feats = []
    for ts in samples:
        for col in ts.columns:
            feats.append(extract_features(
                column_name=col.name, column_type=col.column_type,
                values=col.values, siblings=col.siblings,
                source_table=col.table_name, total_count=col.total_count,
                null_count=col.null_count, distinct_count=col.distinct_count,
            ))
    context.features = feats
    context.vocab = vocab
    assert len(feats) > 0, "No features loaded"


@when("I run gpu_sage with 32 permutations")
def step_run_gpu_sage(context):
    import numpy as np
    from atelier.classify.gpu_importance import gpu_sage

    gt = np.zeros(len(context.features), dtype=np.int64)
    context.sage_result = gpu_sage(
        context.features, gt, context.vocab,
        n_permutations=32, chunk_size=16, detect_convergence=False,
    )


@then("the result has one importance value per feature")
def step_one_per_feature(context):
    from atelier.classify.features import FEATURE_NAMES
    assert len(context.sage_result.importance_values) == len(FEATURE_NAMES), (
        f"expected {len(FEATURE_NAMES)} values, got "
        f"{len(context.sage_result.importance_values)}"
    )


@then("the elapsed time is under {max_seconds:d} seconds")
def step_under_budget(context, max_seconds):
    elapsed = context.sage_result.elapsed_seconds
    assert elapsed < max_seconds, (
        f"SAGE took {elapsed:.1f}s, expected < {max_seconds}s"
    )


@then('the top feature by absolute importance is one of "{allowed}"')
def step_top_feature(context, allowed):
    candidates = [c.strip() for c in allowed.split(",")]
    pairs = list(zip(
        context.sage_result.feature_names,
        context.sage_result.importance_values,
    ))
    pairs.sort(key=lambda p: -abs(p[1]))
    top = pairs[0][0]
    assert top in candidates, (
        f"top feature was {top!r}; expected one of {candidates}. "
        f"full ranking: {pairs}"
    )
