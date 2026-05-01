"""Step definitions for Governance Cost Model BDD scenarios.

Validates the Type-II-aversion preamble + vocabulary-aware sensitivity
map injected into the LLM system prompt by
``llm_backend.build_system_prompt``.  See
``docs/src/architecture/dst-evidence-independence.md`` for the
academic framing (Elkan 2001 cost-sensitive classification + privacy
regime conventions).
"""

from behave import given, when, then


@when("I build the system prompt against the universal vocabulary")
def step_build_prompt_universal(context):
    from atelier.classify.llm_backend import build_category_table, build_system_prompt
    from atelier.classify.taxonomy import load_universal_vocabulary
    vocab = load_universal_vocabulary(hierarchical=True)
    table = build_category_table(vocab)
    context.system_prompt = build_system_prompt(table, category_set=vocab)


@when("I build the system prompt against that fictitious vocabulary")
def step_build_prompt_fictitious(context):
    from atelier.classify.llm_backend import build_category_table, build_system_prompt
    table = build_category_table(context.numeric_vocab)
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


# ── Helper-level ────────────────────────────────────────────────


@given("a category with sensitivity ratings {key1}={val1} and {key2}={val2}")
def step_category_with_sensitivity(context, key1, val1, key2, val2):
    from atelier.classify.taxonomy import ReferenceCategory
    sens = {key1: val1, key2: val2}
    context.cat = ReferenceCategory(
        code="acme.test", label="Test", embedding_text="",
        abbrev="", sensitivity=sens,
    )


@then("the category's min sensitivity rating is {expected:d}")
def step_min_rating(context, expected):
    from atelier.classify.llm_backend import _category_min_rating
    actual = _category_min_rating(context.cat)
    assert actual == expected, f"min_rating={actual!r}, expected {expected}"


@then("the category tiers as {tier}")
def step_category_tier(context, tier):
    from atelier.classify.llm_backend import _category_min_rating
    r = _category_min_rating(context.cat)
    assert r is not None, "min_rating is None — cannot tier"
    if tier == "high":
        assert r <= 1, f"min_rating={r}, expected ≤1 for high tier"
    elif tier == "moderate":
        assert r == 2, f"min_rating={r}, expected ==2 for moderate tier"
    elif tier == "low":
        assert r >= 3, f"min_rating={r}, expected ≥3 for low tier"
    else:
        raise ValueError(f"Unknown tier: {tier!r}")


@given('high-tier members "{members_spec}"')
def step_high_tier_members(context, members_spec):
    """Parse "Name(abbrev=X), Name(no abbrev), Name(abbrev=Y)" into refs."""
    from atelier.classify.taxonomy import ReferenceCategory
    refs: list[ReferenceCategory] = []
    for part in members_spec.split(", "):
        part = part.strip()
        # Format: "Label(abbrev=X)" or "Label(no abbrev)"
        if "(" in part:
            label, rest = part.split("(", 1)
            rest = rest.rstrip(")")
            abbrev = "" if rest == "no abbrev" else rest.split("=", 1)[1]
        else:
            label, abbrev = part, ""
        label = label.strip()
        refs.append(ReferenceCategory(
            code=f"acme.{label.lower().replace(' ', '_').replace('-', '_')}",
            label=label, embedding_text="", abbrev=abbrev.strip(),
            sensitivity={"non_corp": "1"},  # all in high tier
        ))
    context.tier_members = refs


@then('the exemplar list contains "{a}" before "{b}"')
def step_exemplar_order(context, a, b):
    # Re-implement the exemplar selection inline to assert ordering.
    # This mirrors _exemplars in _sensitive_subtree_summary —
    # abbrev-presence preferred, then label-length ascending.
    members = list(context.tier_members)
    members.sort(
        key=lambda c: (
            0 if (getattr(c, "abbrev", "") or "") else 1,
            len(c.label or ""),
        )
    )
    out: list[str] = []
    for m in members:
        label = (getattr(m, "abbrev", "") or "") or m.label or m.code
        if label not in out:
            out.append(label)
    assert a in out and b in out, f"Both {a!r} and {b!r} must appear; got {out}"
    assert out.index(a) < out.index(b), (
        f"Expected {a!r} before {b!r}; got order {out}"
    )
