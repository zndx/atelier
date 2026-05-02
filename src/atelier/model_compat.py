# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Tiny cross-cutting helpers for Claude model capability detection.

Kept deliberately dependency-free — both the Agent SDK code paths
(``terminal.py``, ``overwatch/agent.py``) and the direct-API
classification backends (``classify/llm_backend.py``) import from here,
and none of them want to pull a heavy transitive.
"""

from __future__ import annotations

import re


# Matches any Opus id that implies the 4.7-era (adaptive-thinking) API.
# We match both dotted and dashed variants as well as future ``4-8`` /
# ``5-x`` identifiers — they're expected to continue using the new
# thinking shape.  Bedrock ARNs embed the dashed form, so this works
# for both direct Anthropic and Bedrock route.
_OPUS_ADAPTIVE_RE = re.compile(
    r"claude-opus-(?:4[-.](?:7|8|9)|[5-9]|\d{2,})",
    re.IGNORECASE,
)


def requires_adaptive_thinking(model: str | None) -> bool:
    """Return True when *model* only accepts the new thinking shape.

    Opus 4.7+ (and all future Opus ≥ 5) reject the legacy
    ``thinking.type=enabled`` + ``budget_tokens`` pair.  They require
    ``thinking.type=adaptive`` alongside ``output_config.effort``
    (``low`` / ``medium`` / ``high`` / ``max``).

    Earlier Opus builds (4.6 on Bedrock today, 4.5 historically) still
    accept the legacy shape and are incompatible with the adaptive
    route on current deployments — so treat anything *below* 4.7 as
    legacy until upstream catches up.
    """
    if not model:
        return False
    return bool(_OPUS_ADAPTIVE_RE.search(model))
