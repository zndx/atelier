# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for Governance Cost Model BDD scenarios.

Validates the Type-II-aversion preamble + ICE-only vocabulary
sensitivity map injected into the LLM system prompt by
``llm_backend.build_system_prompt``.  See
``docs/src/architecture/dst-evidence-independence.md`` for the
academic framing (Elkan 2001 cost-sensitive classification + privacy
regime conventions) and ``src/atelier/classify/fixtures/PROVENANCE.md``
for the public-source attribution that gates the sensitivity-map
activation.
"""

from behave import given, when, then


@when("I build the system prompt against the universal vocabulary")
def step_build_prompt_universal(context):
    from atelier.classify.llm_backend import build_category_tree, build_system_prompt
    from atelier.classify.taxonomy import load_universal_vocabulary
    vocab = load_universal_vocabulary(hierarchical=True)
    table = build_category_tree(vocab)
    context.system_prompt = build_system_prompt(table, category_set=vocab)


@when("I build the system prompt against that non-ICE vocabulary")
def step_build_prompt_non_ice(context):
    from atelier.classify.llm_backend import build_category_tree, build_system_prompt
    table = build_category_tree(context.numeric_vocab)
    context.system_prompt = build_system_prompt(table, category_set=context.numeric_vocab)


@then('the system prompt contains "{needle}"')
def step_system_prompt_contains(context, needle):
    assert needle in context.system_prompt, (
        f"Expected {needle!r} in system prompt; got prompt of length "
        f"{len(context.system_prompt)} (first 200 chars: "
        f"{context.system_prompt[:200]!r})"
    )


@then('the system prompt does not contain "{needle}"')
def step_system_prompt_not_contains(context, needle):
    assert needle not in context.system_prompt, (
        f"Did not expect {needle!r} in system prompt"
    )


@then('the system prompt mentions at least one of "{a}", "{b}", or "{c}"')
def step_system_prompt_mentions_one_of(context, a, b, c):
    found = [n for n in (a, b, c) if n in context.system_prompt]
    assert found, (
        f"Expected at least one of {(a, b, c)!r} in system prompt; "
        f"none present"
    )


@then("_sensitive_subtree_summary returns the empty string for that vocabulary")
def step_summary_empty(context):
    from atelier.classify.llm_backend import _sensitive_subtree_summary
    result = _sensitive_subtree_summary(context.numeric_vocab)
    assert result == "", (
        f"Expected empty string from _sensitive_subtree_summary on a "
        f"non-ICE vocabulary; got {result!r}"
    )
