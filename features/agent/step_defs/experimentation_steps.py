# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for the user experimentation journey.

Exercises the critical phase transition: orienting → experimentation.
Uses the committed mock customer taxonomy fixture — earlier versions
of this file branched on a local meta-tagging UAT mount, but that
corpus's hand-curated ICE-mapping overlay was excised when the
LLM-mediated alignment landed in ``ontology_alignment.py``.  Refer
to that module's docstring for the new vocabulary-translation path.
"""

import logging
from pathlib import Path

from behave import given, when, then

log = logging.getLogger(__name__)


# ── Background: load user taxonomy ──────────────────────────────────


@given("the user taxonomy is loaded")
def step_load_user_taxonomy(context):
    """Load the committed mock customer taxonomy fixture."""
    from atelier.classify.taxonomy import (
        compose_vocabularies,
        load_annotations_from_json,
        load_universal_vocabulary,
    )

    universal = load_universal_vocabulary(hierarchical=True)
    fixtures_dir = Path(__file__).resolve().parent.parent.parent.parent
    fixtures_dir = fixtures_dir / "src" / "atelier" / "classify" / "fixtures"
    domain = load_annotations_from_json(
        fixtures_dir / "mock_user_taxonomy.json", hierarchical=True,
    )
    composed = compose_vocabularies(universal, domain)
    context.taxonomy_source = "mock"
    context.user_vocab = domain
    context.universal_vocab = universal
    context.composed_vocab = composed
    log.info("Using mock customer taxonomy fixture")


# ── Vocabulary composition ──────────────────────────────────────────


@then("the composed vocabulary should have more leaves than universal alone")
def step_composed_has_more_leaves(context):
    universal_leaves = set(context.universal_vocab.leaf_codes)
    composed_leaves = set(context.composed_vocab.leaf_codes)

    # Mock taxonomy adds new domain leaves
    added = composed_leaves - universal_leaves
    assert len(added) > 0, (
        f"Composed has {len(composed_leaves)} leaves, "
        f"universal has {len(universal_leaves)} — no domain leaves added"
    )
    log.info(
        "Vocabulary: %d universal + %d domain = %d composed leaves "
        "(source: %s)",
        len(universal_leaves), len(added), len(composed_leaves),
        context.taxonomy_source,
    )


@then("every custom leaf should be reachable from the ICE root")
def step_custom_leaves_reachable(context):
    composed = context.composed_vocab

    # Mock taxonomy: check that NEW domain leaves (not in universal)
    # are reachable from ICE root.
    universal_codes = set(context.universal_vocab.all_by_code.keys())
    domain_leaves = [
        code for code in composed.leaf_codes
        if code not in universal_codes
    ]
    unreachable = []
    for code in domain_leaves:
        ancestors = composed.ancestors(code)
        if not ancestors or "ICE" not in ancestors:
            unreachable.append(code)
    assert not unreachable, (
        f"{len(unreachable)} domain leaves not reachable from ICE root: "
        f"{unreachable[:10]}"
    )


# ── Generator coverage ──────────────────────────────────────────────


@when("I build a generator registry for the composed vocabulary")
def step_build_registry(context):
    from atelier.classify.synth_registry import GeneratorRegistry
    context.registry = GeneratorRegistry.from_vocabulary(context.composed_vocab)


@then("at least {pct:d} percent of custom leaf categories should have generators")
def step_generator_coverage(context, pct):
    composed = context.composed_vocab
    report = context.registry.coverage_report(composed)

    # Check the NEW domain leaves added by the mock taxonomy on top of
    # the universal base — those are the only leaves the registry
    # needs to cover at this level for an experimentation run.
    universal_codes = set(context.universal_vocab.leaf_codes)
    domain_leaves = [
        code for code in composed.leaf_codes
        if code not in universal_codes
    ]
    assert domain_leaves, "No domain leaves found in composed vocabulary"
    covered = sum(
        1 for code in domain_leaves
        if report.get(code, "missing") != "missing"
    )
    actual_pct = (covered / len(domain_leaves)) * 100
    log.info(
        "Generator coverage: %d/%d domain leaves (%.0f%%)",
        covered, len(domain_leaves), actual_pct,
    )
    assert actual_pct >= pct, (
        f"Only {actual_pct:.0f}% of domain leaves have generators "
        f"({covered}/{len(domain_leaves)}), expected >= {pct}%"
    )


# ── Pipeline vocabulary consistency ─────────────────────────────────


@when("I run the classification pipeline with the composed vocabulary and mock LLM")
def step_run_pipeline_composed(context):
    from atelier.config import load_config
    from atelier.classify.pipeline import run_classification_pipeline
    from atelier.classify.fsm import AgentFSM
    from atelier.classify.sampler import load_fixture_samples

    samples = load_fixture_samples()

    # Build curated reference labels from fixtures
    reference_labels = {}
    for ts in samples:
        for col in ts.columns:
            if col.reference_code:
                reference_labels[col.name] = col.reference_code

    cfg = load_config()
    fsm = AgentFSM()

    # Use the mock LLM that returns curated reference labels
    from features.agent.step_defs.bootstrap_steps import _MockLLMBackend
    mock_backend = _MockLLMBackend(reference_labels)

    context.pipeline_result = run_classification_pipeline(
        cfg, fsm,
        samples=samples,
        llm_backend=mock_backend,
        category_set=context.composed_vocab,
    )


# "the pipeline should reach CONVERGED state" is defined in classification_steps.py


@then("every predicted code should exist in the composed vocabulary")
def step_codes_in_vocab(context):
    all_codes = set(context.composed_vocab.all_by_code.keys())
    classifications = context.pipeline_result.get("classifications", [])
    invalid = []
    for c in classifications:
        code = c.get("predicted_code")
        if code and code not in all_codes:
            invalid.append(code)
    assert not invalid, (
        f"{len(invalid)} predicted codes not in composed vocabulary: "
        f"{invalid[:10]}"
    )


@then("the belief paths should trace to roots in the universal hierarchy")
def step_belief_paths_to_roots(context):
    classifications = context.pipeline_result.get("classifications", [])
    universal_codes = set(context.universal_vocab.all_by_code.keys())
    broken = []
    for c in classifications:
        bp = c.get("belief_path", [])
        if bp:
            # The last entry in belief_path is the root-most ancestor
            root_code = bp[-1].get("code", "")
            if root_code not in universal_codes:
                broken.append(c.get("column_name", "?"))
    assert not broken, (
        f"{len(broken)} classifications have belief paths that don't "
        f"trace to universal root: {broken[:10]}"
    )
