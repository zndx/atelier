# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.

"""Wiring tests for ClassifyBackedEnrichmentGenerator (P4 — DST pivot).

Covers the wiring without spending on real LLM calls:

  - Model resolver: per-backend apex selection + Bedrock loud-fail.
  - Generator: prompt assembly, JSON parsing (clean + fenced + prose),
    GenerationResult shape, name provenance.
  - Default-on discipline: the script's default generator is the
    provider-co-located one, not the stub.

The real-LLM smoke (one tag, real backend) is intentionally a
separate test gated on credentials — this file stays free of network
or credential dependencies.
"""

from __future__ import annotations

import json
from unittest import mock

import pytest

from atelier.config import AtelierConfig
from atelier.enrichment.llm_generator import (
    ClassifyBackedEnrichmentGenerator,
    DeterministicStubGenerator,
    _parse_enrichment_json,
)
from atelier.enrichment.model_resolver import (
    EnrichmentModelError,
    resolve_enrichment_model,
)


# ── Model resolver ────────────────────────────────────────────────


def test_resolver_anthropic_apex():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "anthropic"
    backend, model = resolve_enrichment_model(cfg)
    assert backend == "anthropic"
    assert model == "claude-opus-4-7"


def test_resolver_bedrock_without_override_raises():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "bedrock"
    with pytest.raises(EnrichmentModelError) as exc_info:
        resolve_enrichment_model(cfg)
    # Remediation hint must be operator-actionable
    assert "ATELIER_ENRICHMENT_MODEL" in str(exc_info.value)
    assert "inference-profile" in str(exc_info.value)


def test_resolver_bedrock_with_override_succeeds():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "bedrock"
    cfg.enrichment_model_override = "arn:aws:bedrock:us-east-1:1:ip/claude-opus-4-v1:0"
    backend, model = resolve_enrichment_model(cfg)
    assert backend == "bedrock"
    assert model == cfg.enrichment_model_override


def test_resolver_openai_compatible_falls_through_to_classify_model():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "openai_compatible"
    cfg.classify_llm_model = "glm-4.7"
    backend, model = resolve_enrichment_model(cfg)
    assert backend == "openai_compatible"
    assert model == "glm-4.7"


def test_resolver_override_wins_over_apex():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "anthropic"
    cfg.enrichment_model_override = "claude-sonnet-4-6"
    backend, model = resolve_enrichment_model(cfg)
    assert backend == "anthropic"
    assert model == "claude-sonnet-4-6"


# ── JSON parsing ──────────────────────────────────────────────────


def test_parse_clean_json():
    text = '{"prototype_values": ["x"], "name_hints": ["a"]}'
    obj, notes = _parse_enrichment_json(text)
    assert obj["prototype_values"] == ["x"]
    assert notes == ""


def test_parse_fenced_json():
    text = '```json\n{"prototype_values": ["x"]}\n```'
    obj, notes = _parse_enrichment_json(text)
    assert obj["prototype_values"] == ["x"]


def test_parse_json_with_trailing_prose():
    text = '{"prototype_values": ["x"]} — here is some prose after'
    obj, notes = _parse_enrichment_json(text)
    assert obj["prototype_values"] == ["x"]
    assert "fallback" in notes  # records that the fallback path was used


def test_parse_raises_on_no_json():
    with pytest.raises(ValueError):
        _parse_enrichment_json("no json here at all")


# ── Generator end-to-end with mocked backend client ───────────────


def _sample_taxonomy_row() -> dict:
    return {
        "code": "1.1.1",
        "label": "ISBN-13",
        "abbrev": "ISBN13",
        "description": "13-digit International Standard Book Number",
        "parent_code": "1.1",
        "common_names": "isbn,book_id",
    }


def _sample_taxonomy_context(*, is_leaf: bool = True) -> dict:
    return {
        "is_leaf": is_leaf,
        "expected_parent_path": ["Identifiers", "Publication Identifiers", "ISBN"],
        "siblings": [
            {"code": "1.1.2", "label": "ISBN-10"},
            {"code": "1.1.3", "label": "ISSN"},
        ],
        "valid_tag_codes": {"1.1.1", "1.1.2", "1.1.3"},
    }


def test_generator_assembles_prompts_and_parses_response():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "anthropic"

    canned_json = json.dumps({
        "prototype_values": ["978-3-16-148410-0", "9780306406157"],
        "value_patterns": [{"kind": "regex", "expr": r"^\d{3}-?\d-\d{3}-\d{5}-\d$"}],
        "name_hints": ["isbn", "isbn13", "book_id"],
        "anti_examples": [
            {"value": "0306406152", "confusable_tag": "1.1.2",
             "reason": "10-digit ISBN-10 lacks the 978/979 prefix"}
        ],
        "parent_path": ["Identifiers", "Publication Identifiers", "ISBN"],
    })

    with mock.patch(
        "atelier.enrichment.backend_client.generate_text",
        return_value=canned_json,
    ) as mock_generate:
        gen = ClassifyBackedEnrichmentGenerator(cfg)
        result = gen.generate(
            _sample_taxonomy_row(),
            taxonomy_context=_sample_taxonomy_context(is_leaf=True),
        )

    # Backend got the expected call
    assert mock_generate.call_count == 1
    call_kwargs = mock_generate.call_args.kwargs
    assert call_kwargs["backend"] == "anthropic"
    assert call_kwargs["model"] == "claude-opus-4-7"
    assert call_kwargs["reasoning_budget"] == 16384
    assert "LEAF tag" in call_kwargs["system_prompt"]
    assert "ISBN-13" in call_kwargs["user_prompt"]
    # Anti-example candidates should mention the siblings
    assert "ISBN-10" in call_kwargs["user_prompt"]

    # Result shape
    assert result.enrichment["prototype_values"] == [
        "978-3-16-148410-0", "9780306406157",
    ]
    assert result.attempts == 1
    # Generator records provenance as {backend}:{model}
    assert gen.name == "anthropic:claude-opus-4-7"


def test_generator_parent_framing_for_internal_nodes():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "anthropic"

    canned_json = json.dumps({
        "prototype_values": ["generic_payment_card_number"],
        "value_patterns": [{"kind": "regex", "expr": r".+"}],
        "name_hints": ["payment_card"],
        "anti_examples": [],
        "parent_path": ["Financial Sensitive Information", "Payment Card Information"],
    })

    with mock.patch(
        "atelier.enrichment.backend_client.generate_text",
        return_value=canned_json,
    ) as mock_generate:
        gen = ClassifyBackedEnrichmentGenerator(cfg)
        ctx = _sample_taxonomy_context(is_leaf=False)
        ctx["children"] = [
            {"code": "1.1.1.1.1.1.1", "label": "Payment Card Number"},
            {"code": "1.1.1.1.1.1.2", "label": "Card Verification Value"},
        ]
        gen.generate(_sample_taxonomy_row(), taxonomy_context=ctx)

    call_kwargs = mock_generate.call_args.kwargs
    # Parent framing kicks in
    assert "INTERNAL (parent)" in call_kwargs["system_prompt"]
    # Children appear in the user prompt so the model knows what specializations exist
    assert "Payment Card Number" in call_kwargs["user_prompt"]
    assert "Card Verification Value" in call_kwargs["user_prompt"]


def test_generator_provides_verifier_feedback_on_retry():
    cfg = AtelierConfig()
    cfg.classify_llm_backend = "anthropic"

    canned_first = json.dumps({"prototype_values": ["x"], "parent_path": []})
    canned_retry = json.dumps({
        "prototype_values": ["978-3-16-148410-0"],
        "parent_path": ["Identifiers", "Publication Identifiers", "ISBN"],
        "value_patterns": [], "name_hints": ["isbn"], "anti_examples": [],
    })

    from atelier.enrichment.llm_generator import GenerationResult

    with mock.patch(
        "atelier.enrichment.backend_client.generate_text",
        return_value=canned_retry,
    ) as mock_generate:
        gen = ClassifyBackedEnrichmentGenerator(cfg)
        prior = GenerationResult(
            enrichment=json.loads(canned_first), attempts=1, notes="",
        )
        gen.generate(
            _sample_taxonomy_row(),
            taxonomy_context=_sample_taxonomy_context(is_leaf=True),
            prior_attempt=prior,
            verifier_feedback={
                # Matches ReportResult.to_dict(): failed checks live under
                # "details", each as {name, detail, ...} (verifiers.py).
                "details": [
                    {"name": "parent_path_consistent",
                     "detail": "parent_path does not match taxonomy hierarchy"}
                ]
            },
        )

    user_prompt = mock_generate.call_args.kwargs["user_prompt"]
    # The retry must include the verifier feedback so the model can correct
    assert "parent_path_consistent" in user_prompt
    assert "previous attempt failed" in user_prompt.lower()


# ── Default-on discipline ─────────────────────────────────────────


def test_enrich_script_defaults_to_backend_generator():
    """The CLI default must construct the provider-co-located generator,
    not the deterministic stub.  Stub-default would silently degrade
    real runs to non-LLM enrichment, contradicting the no-silent-DST-
    degradation discipline."""
    import argparse
    from scripts import enrich_annotations  # type: ignore[import-not-found]

    # Reconstruct just the argparse to inspect default — avoid actually
    # running main() since it requires a source-json and a live Qdrant.
    parser = argparse.ArgumentParser()
    # Re-invoke the same add_argument block by parsing an empty arg list
    # against a probe parser would be brittle; instead inspect the script
    # by reading argv defaults via a known-good invocation.
    parsed = parser.parse_known_args([])  # noop probe
    del parsed

    # Direct introspection: the default on the --generator argument is
    # what we care about.  Construct the script-level parser and look
    # at its action defaults.
    _orig_parse = argparse.ArgumentParser.parse_args

    captured = {}

    def _capture(self, argv=None, namespace=None):
        captured["defaults"] = {
            a.dest: a.default for a in self._actions
        }
        # Short-circuit by raising SystemExit so main() doesn't proceed
        raise SystemExit(0)

    try:
        argparse.ArgumentParser.parse_args = _capture  # type: ignore[method-assign]
        with pytest.raises(SystemExit):
            enrich_annotations.main([
                "--taxonomy-id", "x", "--augmentation-version", "y",
                "--source-json", "/nonexistent",
            ])
    finally:
        argparse.ArgumentParser.parse_args = _orig_parse  # type: ignore[method-assign]

    assert captured["defaults"]["generator"] == "backend", (
        "Default generator must be the provider-co-located backend, not the stub"
    )


# ── DeterministicStubGenerator remains usable for tests ───────────


def test_stub_generator_still_works():
    """The stub must stay buildable + functional for tests that don't
    want to mock LLM calls.  Removing it would break CI."""
    gen = DeterministicStubGenerator()
    result = gen.generate(
        _sample_taxonomy_row(),
        taxonomy_context=_sample_taxonomy_context(),
    )
    assert result.enrichment["prototype_values"]
    assert result.enrichment["name_hints"]
    assert gen.name == "stub:deterministic"
