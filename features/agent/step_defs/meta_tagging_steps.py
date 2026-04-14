"""Step definitions for meta-tagging overlay BDD scenarios."""

from behave import given, when, then


@given("a ground truth dict with meta-tagging codes")
def step_mock_gt(context):
    context.meta_gt = {
        "col_pan": "1.1.1.1.1.1.1",
        "col_cvv": "1.1.1.1.1.1.2",
        "col_email": "1.1.1.9.3.1",
        "col_phone": "1.1.1.9.4.1",
        "col_ssn": "1.1.2.1.2.1.3",
        "col_dob": "1.1.1.2.3.2",
        "col_salary": "1.1.1.2.1.1",
    }


@when("I translate ground truth to ICE codes")
def step_translate(context):
    from atelier.classify.meta_tagging_overlay import translate_ground_truth
    context.translated, context.unmapped = translate_ground_truth(context.meta_gt)


@then("every translated code should be a valid ICE code")
def step_check_valid_ice(context):
    for col, code in context.translated.items():
        assert code.startswith("ICE."), (
            f"Column {col}: code {code} doesn't start with ICE."
        )


@then("the translation should cover all provided codes")
def step_check_full_coverage(context):
    assert len(context.unmapped) == 0, (
        f"Unmapped codes: {context.unmapped}"
    )
    assert len(context.translated) == len(context.meta_gt), (
        f"Translated {len(context.translated)}, expected {len(context.meta_gt)}"
    )


@when("I build a blended vocabulary from the base")
def step_build_blended(context):
    # Use base vocabulary as blended (no meta-tagging dir in CI)
    context.blended = context.category_set


@then("every leaf should have a valid parent chain to ICE root")
def step_check_parent_chain(context):
    cs = context.blended
    for code in cs.leaf_codes:
        ancestors = cs.ancestors(code)
        # Root ICE should be in ancestors (or the code itself if root)
        parts = code.split(".")
        root = parts[0]
        assert root == "ICE" or "ICE" in ancestors, (
            f"Leaf {code} has no path to ICE root. Ancestors: {ancestors}"
        )
