# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Step definitions for naming audit and file content checks."""

import os
from behave import when, then

from features.deployment.step_defs.helpers import PROJECT_ROOT


@when('I search {directory} for "{text}"')
def step_search_directory(context, directory, text):
    """Search recursively for text in a directory."""
    search_dir = PROJECT_ROOT / directory
    context.search_matches = []
    if not search_dir.exists():
        return
    for root, _dirs, files in os.walk(search_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if text in content:
                    context.search_matches.append(fpath)
            except (OSError, UnicodeDecodeError):
                pass


@then("no matches are found")
def step_no_matches(context):
    matches = getattr(context, "search_matches", [])
    assert len(matches) == 0, \
        f"Expected no matches, found {len(matches)}: {matches}"


@then("at least one match is found")
def step_has_matches(context):
    matches = getattr(context, "search_matches", [])
    assert len(matches) > 0, "Expected at least one match, found none"


@then('it contains "{text}"')
def step_file_contains(context, text):
    """Check that the current file (from a Given step) contains text."""
    content = context.current_file.read_text()
    assert text in content, \
        f"File does not contain '{text}': {context.current_file}"
