# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Focus-list computation for the adaptive Settings page.

After a pipeline run completes, the Settings page shows a "Focus"
section listing the 3–7 controls most relevant to this run. Two
signal sources contribute:

1. **Deterministic rules** — any control whose resolved value drifts
   meaningfully from its HOCON default (> 10% for floats, any change
   for enums / bools / ints). Always computed; always available.

2. **Overwatch extract** — if the Overwatch agent produced a markdown
   report, we look for a fenced ``focus`` JSON block the agent was
   asked to emit. The union of the two sets is the final focus list.

Missing or malformed overwatch output degrades gracefully to the
deterministic-only path — the UI remains useful even when overwatch
is disabled (Bedrock-only deployments) or when the agent prose
skipped the JSON block.

Output schema (written to ``build/results/{run_id}/focus_settings.json``)::

    {
        "run_id": "a22f1f10",
        "computed_at": "2026-04-17T14:23:00+00:00",
        "source": "hybrid" | "rules" | "overwatch",
        "focus_keys": ["classify_monte_carlo_sample_fraction", ...],
        "deterministic": ["..."],   # subset of focus_keys
        "from_overwatch": ["..."],  # subset of focus_keys
    }

Consumers (gateway ``/api/settings/focus``, UI) union the two
subsets — they're reported separately so the UI can mark which
signal originated each entry if it wants.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Relative drift threshold for float controls — below this, a
# divergence from default is treated as noise and not flagged.
_FLOAT_DRIFT_REL = 0.10


def compute_focus(
    run_id: str,
    results_dir: Path | str,
) -> dict[str, Any]:
    """Compute and persist the focus list for a completed run.

    Reads ``settings_snapshot.json`` for the run's resolved/default
    values and (optionally) ``overwatch.md`` for a fenced ``focus``
    JSON block. Writes the result to ``focus_settings.json`` and
    returns the dict.

    Never raises — any error downgrades to an empty focus list with
    a ``source: "error"`` tag so the gateway falls back to the
    starter set. Callers that want to surface errors should inspect
    the returned dict.
    """
    results_dir = Path(results_dir)
    out: dict[str, Any] = {
        "run_id": run_id,
        "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "error",
        "focus_keys": [],
        "deterministic": [],
        "from_overwatch": [],
    }

    try:
        snap_path = results_dir / "settings_snapshot.json"
        if not snap_path.exists():
            log.warning("compute_focus: no settings_snapshot.json in %s", results_dir)
            out["source"] = "no_snapshot"
            _persist(results_dir, out)
            return out

        snap = json.loads(snap_path.read_text())
        deterministic = sorted(_deterministic_flags(snap))

        from_overwatch: list[str] = []
        overwatch_path = results_dir / "overwatch.md"
        if overwatch_path.exists():
            from_overwatch = sorted(_parse_overwatch_focus(overwatch_path))

        merged = sorted(set(deterministic) | set(from_overwatch))
        if deterministic and from_overwatch:
            source = "hybrid"
        elif from_overwatch:
            source = "overwatch"
        else:
            source = "rules"

        out.update({
            "source": source,
            "focus_keys": merged,
            "deterministic": deterministic,
            "from_overwatch": from_overwatch,
        })
    except Exception as exc:
        log.warning("compute_focus failed (non-fatal): %s", exc)
        out["error"] = str(exc)

    _persist(results_dir, out)
    return out


def _persist(results_dir: Path, payload: dict[str, Any]) -> None:
    path = results_dir / "focus_settings.json"
    try:
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    except Exception as exc:
        log.warning("focus_settings.json write failed: %s", exc)


def _deterministic_flags(snap: dict[str, Any]) -> list[str]:
    """Flag controls whose resolved value diverges from default."""
    from atelier.config_overlay import SETTINGS_METADATA
    resolved = snap.get("resolved_values") or {}
    defaults = snap.get("default_values") or {}

    flagged: list[str] = []
    for key, meta in SETTINGS_METADATA.items():
        if key not in resolved:
            continue
        cur = resolved[key]
        base = defaults.get(key, meta.get("default"))
        if _diverges(meta.get("type"), cur, base):
            flagged.append(key)
    return flagged


def _diverges(kind: str | None, current: Any, baseline: Any) -> bool:
    """Return True when ``current`` is meaningfully different from ``baseline``."""
    if baseline is None or current is None:
        return False
    if kind == "float":
        try:
            b = float(baseline)
            c = float(current)
        except (TypeError, ValueError):
            return False
        if b == 0:
            return abs(c) > 1e-6
        return abs((c - b) / b) > _FLOAT_DRIFT_REL
    if kind == "int":
        return int(current) != int(baseline)
    if kind in ("choice", "switch"):
        return current != baseline
    return current != baseline


# Overwatch emits a fenced code block tagged ``focus`` (per the
# prompt addendum). Matches ``focus`` opener and a closing fence.
_FOCUS_BLOCK_RE = re.compile(
    r"```focus\s*\n(?P<payload>.*?)\n```",
    re.DOTALL | re.IGNORECASE,
)


def _parse_overwatch_focus(overwatch_path: Path) -> list[str]:
    """Extract the ``focus_keys`` list from an overwatch.md file.

    Tolerant of missing / malformed blocks — returns empty list in
    either case and logs a debug message. Only keys that are known
    to SETTINGS_METADATA survive; unknown keys are dropped.
    """
    from atelier.config_overlay import SETTINGS_METADATA
    try:
        text = overwatch_path.read_text()
    except OSError as exc:
        log.debug("overwatch.md unreadable: %s", exc)
        return []

    match = _FOCUS_BLOCK_RE.search(text)
    if not match:
        return []

    try:
        payload = json.loads(match.group("payload"))
    except json.JSONDecodeError as exc:
        log.debug("overwatch focus block not valid JSON: %s", exc)
        return []

    keys = payload.get("focus_keys") if isinstance(payload, dict) else None
    if not isinstance(keys, list):
        log.debug("overwatch focus block missing focus_keys list")
        return []

    return [k for k in keys if isinstance(k, str) and k in SETTINGS_METADATA]
