"""Overwatch — autonomous pipeline monitoring and code evolution.

Requires direct Anthropic API access (not Bedrock). When enabled,
a dedicated Claude Opus instance monitors classification runs and
can propose code improvements via GitHub App PRs.

Guard rail: ``AtelierConfig.has_overwatch`` gates on both
``overwatch.enabled=true`` AND ``has_anthropic`` (real API key).
"""
