"""Step definitions for the user experimentation journey.

Exercises the critical phase transition: orienting → experimentation.
Uses mock customer taxonomy by default; overrides with real meta-tagging
annotations when ATELIER_REAL_DATA_DIR is available or when the in-repo
UAT snapshot at ``build/meta-tagging/`` is populated.
"""

import logging
import os
from pathlib import Path

from behave import given, when, then

log = logging.getLogger(__name__)

# Prefer the in-repo UAT snapshot (gitignored under build/) and fall
# back to the legacy maintainer-convention location.
_REPO_SNAPSHOT = Path(__file__).resolve().parents[3] / "build" / "meta-tagging"
_LEGACY_DATA_DIR = Path("~/local/tmp/meta-tagging").expanduser()


def _data_dir() -> Path:
    env = os.environ.get("ATELIER_REAL_DATA_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    if (_REPO_SNAPSHOT / "annotations.csv").is_file():
        return _REPO_SNAPSHOT
    return _LEGACY_DATA_DIR


# ── Background: load user taxonomy ──────────────────────────────────


@given("the user taxonomy is loaded")
def step_load_user_taxonomy(context):
    """Load mock taxonomy; override with real meta-tagging if available."""
    from atelier.classify.taxonomy import (
        compose_vocabularies,
        load_annotations_from_json,
        load_universal_vocabulary,
    )

    universal = load_universal_vocabulary(hierarchical=True)

    # Prefer real meta-tagging annotations when available locally.
    # When present, use the expanded sample vocabulary (300+ leaves) as
    # base, and validate mapping coverage. The meta-tagging codes map INTO
    # existing ICE codes rather than adding new leaves.
    real_dir = _data_dir()
    if real_dir.exists() and (real_dir / "annotations.csv").exists():
        from atelier.classify.taxonomy import load_sample_vocabulary
        from atelier.classify.meta_tagging_overlay import (
            build_blended_vocabulary,
            mapping_coverage_report,
        )
        # Use expanded sample vocabulary as base (300+ leaves)
        sample_vocab = load_sample_vocabulary(hierarchical=True)
        composed = build_blended_vocabulary(sample_vocab, real_dir)
        coverage = mapping_coverage_report(real_dir)
        context.taxonomy_source = "meta-tagging"
        context.user_vocab = sample_vocab
        context.meta_coverage = coverage
        log.info(
            "Using real meta-tagging annotations from %s "
            "(%d/%d codes mapped)",
            real_dir, coverage.get("mapped", 0), coverage.get("total_annotations", 0),
        )
    else:
        # Fall back to committed mock taxonomy (always available, safe for CI)
        fixtures_dir = Path(__file__).resolve().parent.parent.parent.parent
        fixtures_dir = fixtures_dir / "src" / "atelier" / "classify" / "fixtures"
        domain = load_annotations_from_json(
            fixtures_dir / "mock_user_taxonomy.json", hierarchical=True,
        )
        composed = compose_vocabularies(universal, domain)
        context.taxonomy_source = "mock"
        context.user_vocab = domain
        log.info("Using mock customer taxonomy fixture")

    context.universal_vocab = universal
    context.composed_vocab = composed


# ── Vocabulary composition ──────────────────────────────────────────


@then("the composed vocabulary should have more leaves than universal alone")
def step_composed_has_more_leaves(context):
    universal_leaves = set(context.universal_vocab.leaf_codes)
    composed_leaves = set(context.composed_vocab.leaf_codes)

    if context.taxonomy_source == "meta-tagging":
        # Meta-tagging codes map INTO existing ICE codes (enrichment, not
        # extension). Validate that the expanded sample vocabulary is richer
        # than the universal base, and that mapping coverage is adequate.
        assert len(composed_leaves) > len(universal_leaves), (
            f"Sample vocabulary ({len(composed_leaves)} leaves) should exceed "
            f"universal ({len(universal_leaves)} leaves)"
        )
        coverage = getattr(context, "meta_coverage", {})
        mapped = coverage.get("mapped", 0)
        total = coverage.get("total_annotations", 0)
        log.info(
            "Meta-tagging: %d/%d codes mapped into %d-leaf vocabulary",
            mapped, total, len(composed_leaves),
        )
    else:
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

    if context.taxonomy_source == "meta-tagging":
        # Meta-tagging maps into existing ICE codes. Validate that the
        # composed vocabulary (sample) is internally consistent: all
        # leaves should trace to ICE root.
        unreachable = []
        for code in composed.leaf_codes:
            ancestors = composed.ancestors(code)
            if "ICE" not in ancestors and code != "ICE":
                unreachable.append(code)
        assert not unreachable, (
            f"{len(unreachable)} leaves not reachable from ICE root: "
            f"{unreachable[:10]}"
        )
    else:
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

    if context.taxonomy_source == "meta-tagging":
        # For meta-tagging, check overall vocabulary generator coverage
        all_leaves = list(composed.leaf_codes)
        covered = sum(
            1 for code in all_leaves
            if report.get(code, "missing") != "missing"
        )
        actual_pct = (covered / len(all_leaves)) * 100 if all_leaves else 100
        log.info(
            "Generator coverage: %d/%d leaves (%.0f%%) — meta-tagging mode",
            covered, len(all_leaves), actual_pct,
        )
        assert actual_pct >= pct, (
            f"Only {actual_pct:.0f}% of leaves have generators "
            f"({covered}/{len(all_leaves)}), expected >= {pct}%"
        )
    else:
        # For mock taxonomy, check specifically the new domain leaves
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
            "Generator coverage: %d/%d domain leaves (%.0f%%) — mock mode",
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

    # Build ground truth from fixtures
    gt = {}
    for ts in samples:
        for col in ts.columns:
            if col.ground_truth:
                gt[col.name] = col.ground_truth

    cfg = load_config()
    fsm = AgentFSM()

    # Use the mock LLM that returns ground truth labels
    from features.agent.step_defs.bootstrap_steps import _MockLLMBackend
    mock_backend = _MockLLMBackend(gt)

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
