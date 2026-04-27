# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Overwatch — autonomous pipeline monitoring and code evolution.

Requires direct Anthropic API access (not Bedrock). When enabled,
a dedicated Claude Opus instance monitors classification runs and
can propose code improvements via GitHub App PRs.

Guard rail: ``AtelierConfig.has_overwatch`` gates on both
``overwatch.enabled=true`` AND ``has_anthropic`` (real API key).
"""
