"""In-memory runtime config overlay for the settings page.

Settings applied here survive for the current gateway process.
For persistence across restarts, operators update ``config/base.conf``
or set environment variables (which feed HOCON ``${?VAR}`` substitution).

This is honest UX — the settings page explicitly labels these as
session-level, with a disclaimer that they reset on restart.

Validation is enforced here (ranges, allowed values) so that the
gateway PATCH endpoint and the pipeline integration share a single
source of truth.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

log = logging.getLogger(__name__)


# ── Parameter metadata ─────────────────────────────────────────────
#
# Each entry describes a tunable parameter: its semantic description,
# valid range or allowed values, and a caption template for the UI.
# The settings page fetches this via GET /api/settings.

SETTINGS_METADATA: dict[str, dict[str, Any]] = {
    "classify_fusion_strategy": {
        "hocon_path": "classify.fusion_strategy",
        "label": "Fusion Strategy",
        "description": "DST combination rule for evidence fusion",
        "type": "choice",
        "choices": ["dempster", "yager"],
        "default": "dempster",
        "captions": {
            "dempster": "Dempster: [m1 ⊕ m2] ÷ (1−K) — normalizes conflict",
            "yager": "Yager: [m1 ⊕ m2], conflict → Θ — preserves ignorance",
        },
    },
    "classify_llm_discount": {
        "hocon_path": "classify.llm.discount",
        "label": "LLM Discount",
        "description": "Mass allocated to ignorance from LLM evidence",
        "type": "float",
        "min": 0.05,
        "max": 0.20,
        "step": 0.01,
        "default": 0.10,
        "caption_template": "{value_pct}% of LLM mass allocated to ignorance.",
    },
    "classify_discount_cosine": {
        "hocon_path": "classify.discounts.cosine",
        "label": "Cosine Discount",
        "description": "Mass allocated to ignorance from embedding similarity",
        "type": "float",
        "min": 0.20,
        "max": 0.45,
        "step": 0.01,
        "default": 0.30,
        "caption_template": "{value_pct}% of cosine mass allocated to ignorance.",
    },
    "classify_bootstrap_gap_threshold": {
        "hocon_path": "classify.bootstrap.gap_threshold",
        "label": "Gap Threshold",
        "description": "Belief-gap convergence target: mean(Pl − Bel)",
        "type": "float",
        "min": 0.08,
        "max": 0.25,
        "step": 0.01,
        "default": 0.15,
        "caption_template": "Converge when mean(Pl − Bel) < {value}.",
    },
    "classify_bootstrap_bel_floor": {
        "hocon_path": "classify.bootstrap.bel_floor",
        "label": "Belief Floor",
        "description": "Minimum belief for a prediction to be 'settled'",
        "type": "float",
        "min": 0.40,
        "max": 0.70,
        "step": 0.01,
        "default": 0.50,
        "caption_template": "A prediction is 'settled' when Bel ≥ {value}.",
    },
}


# ── Overlay state ──────────────────────────────────────────────────

_overlay: dict[str, Any] = {}


def get_overlay() -> dict[str, Any]:
    """Return a copy of the current overlay (empty = all defaults)."""
    return dict(_overlay)


def clear_overlay() -> None:
    """Remove all overlay entries — reverts to HOCON/env defaults."""
    _overlay.clear()


def set_overlay(updates: dict[str, Any]) -> dict[str, Any]:
    """Validate and apply overlay updates.

    Raises ValueError on invalid keys or out-of-range values.
    Returns the resulting overlay (merged with prior state).
    """
    for key, value in updates.items():
        _validate(key, value)
    _overlay.update(updates)
    log.info("Config overlay updated: %s", sorted(updates.keys()))
    return dict(_overlay)


def _validate(key: str, value: Any) -> None:
    """Raise ValueError if key is unknown or value is out of range."""
    if key not in SETTINGS_METADATA:
        raise ValueError(f"Unknown setting: {key!r}")
    meta = SETTINGS_METADATA[key]
    kind = meta["type"]

    if kind == "choice":
        if value not in meta["choices"]:
            raise ValueError(
                f"{key}: invalid choice {value!r}, expected one of "
                f"{meta['choices']}"
            )
    elif kind == "float":
        if not isinstance(value, (int, float)):
            raise ValueError(f"{key}: expected number, got {type(value).__name__}")
        fv = float(value)
        if fv < meta["min"] or fv > meta["max"]:
            raise ValueError(
                f"{key}: {fv} out of range [{meta['min']}, {meta['max']}]"
            )
    else:
        raise ValueError(f"{key}: unsupported type {kind}")


def apply_to_config(cfg):  # type: ignore[no-untyped-def]
    """Return a copy of cfg with overlay values applied.

    No-op when overlay is empty.  Callers that need the overlay
    (e.g. the classification pipeline) should wrap their load_config()
    result with this function.
    """
    if not _overlay:
        return cfg
    return dataclasses.replace(cfg, **_overlay)
