# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for the artifact set vocabulary + compatibility tier-0 BDD.

Pure-Python checks against the helpers in
:mod:`atelier.classify.artifact_set`.  No DB, no devenv stack.
"""

from __future__ import annotations

from behave import given, when, then


def _parse_codes(s: str) -> list[str]:
    return [c.strip() for c in s.split(",") if c.strip()]


@given('a class list with codes "{codes}"')
def step_class_list(context, codes):
    """First class-list slot for tier-0 signature scenarios.

    Uses two explicit slots (``primary``/``secondary``) rather than a
    list to avoid behave context-layering surprises where attributes
    set during one scenario carry into the next when behave's user mode
    layers don't pop them at scenario close.
    """
    if getattr(context, "primary_class_list", None) is None:
        context.primary_class_list = _parse_codes(codes)
    else:
        context.secondary_class_list = _parse_codes(codes)


@given('an artifact class list "{codes}"')
def step_artifact_classes(context, codes):
    context.artifact_classes = _parse_codes(codes)


@given('a candidate class list "{codes}"')
def step_candidate_classes(context, codes):
    context.candidate_classes = _parse_codes(codes)


@when("I compute the vocab signature")
def step_compute_signature(context):
    from atelier.classify.artifact_set import compute_vocab_signature
    context.vocab_signature = compute_vocab_signature(context.primary_class_list)


@when("I compute the signatures of both lists")
def step_compute_both(context):
    from atelier.classify.artifact_set import compute_vocab_signature
    context.vocab_signature_a = compute_vocab_signature(context.primary_class_list)
    context.vocab_signature_b = compute_vocab_signature(context.secondary_class_list)


@when("I check compatibility")
def step_check_compat(context):
    from atelier.classify.artifact_set import check_compatibility
    context.compat_report = check_compatibility(
        context.artifact_classes, context.candidate_classes,
    )


@then("the signature should be a 64-character hex string")
def step_signature_shape(context):
    sig = context.vocab_signature
    assert isinstance(sig, str), f"expected str, got {type(sig)}"
    assert len(sig) == 64, f"expected length 64, got {len(sig)}"
    int(sig, 16)  # raises if non-hex


@then("the two signatures should be equal")
def step_signatures_equal(context):
    a, b = context.vocab_signature_a, context.vocab_signature_b
    assert a == b, f"signatures differ: {a} vs {b}"


@then('the compatibility status should be "{status}"')
def step_status(context, status):
    actual = context.compat_report.status
    assert actual == status, f"expected status {status!r}, got {actual!r}"


@then("the compatibility report should report {n:d} missing codes")
def step_missing(context, n):
    actual = len(context.compat_report.missing_codes)
    assert actual == n, (
        f"expected {n} missing, got {actual}: "
        f"{context.compat_report.missing_codes}"
    )


@then("the compatibility report should report {n:d} extra codes")
def step_extra(context, n):
    actual = len(context.compat_report.extra_codes)
    assert actual == n, (
        f"expected {n} extra, got {actual}: "
        f"{context.compat_report.extra_codes}"
    )
