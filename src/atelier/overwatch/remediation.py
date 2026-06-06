"""Remediation invariants — Bedrock-only classifier, tunable scaffolding.

Overwatch is allowed to adapt the scaffolding around classification
(sampling, batching, discounts, curated reference, GPU knobs) but it must
NOT change which provider actually performs the classification
operations.  That invariant is hard: even the autonomous tier cannot
flip the classifier onto a direct-Anthropic backend or a third-party
OpenAI-compatible endpoint, because downstream CAI deployments must
continue to classify through Bedrock Claude endpoints only.

This module provides:

* :data:`PROVIDER_AFFECTING_KEYS` — the frozen set of overlay keys a
  supervisor overlay must never touch.
* :class:`RemediationError` — structured rejection.
* :func:`enforce_bedrock_only` — the invariant gate used by every
  controlled CLI the agent is allowed to run.
* :class:`OverlayProposal` — the shape of a supervisor proposal that
  :mod:`atelier.overwatch.write_proposal` validates + persists.
"""

from __future__ import annotations

import dataclasses
from typing import Any


# Keys that influence which provider / endpoint performs classification.
# These are rejected even if they fall within SETTINGS_METADATA ranges
# because the overlay cannot smuggle the classifier off Bedrock.
PROVIDER_AFFECTING_KEYS: frozenset[str] = frozenset({
    "classify_llm_backend",
    "classify_llm_api_key",
    "classify_llm_base_url",
    "classify_llm_model",
    "classify_subagent_model",
    # Agent / subagent model pins — belong to the agent host, not the
    # classifier, but overwatch-initiated changes here would also
    # violate the scaffolding/classifier separation.
    "agent_model",
    # Note: overwatch_model is no longer a config field — Overwatch
    # follows the WTA selection.  Keeping the WTA selection out of
    # remediation overlays preserves operator-only control.
})


class RemediationError(ValueError):
    """Supervisor proposal violates a remediation invariant."""


@dataclasses.dataclass
class OverlayProposal:
    """A supervisor proposal to overlay specific AtelierConfig keys.

    Structured as a dataclass so callers get a stable JSON shape across
    the CLI boundary, and so ``enforce_bedrock_only`` has something
    concrete to validate.
    """

    run_id: str
    overlay: dict[str, Any]
    rationale: str = ""
    expected_effect: str = ""
    trigger: str = ""  # what caused overwatch to propose this (e.g. "fsm_error", "post_mortem")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "overlay": dict(self.overlay),
            "rationale": self.rationale,
            "expected_effect": self.expected_effect,
            "trigger": self.trigger,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OverlayProposal":
        if "run_id" not in data:
            raise RemediationError("proposal missing run_id")
        if "overlay" not in data:
            raise RemediationError("proposal missing overlay")
        if not isinstance(data["overlay"], dict):
            raise RemediationError("proposal overlay must be an object")
        return cls(
            run_id=str(data["run_id"]),
            overlay=dict(data["overlay"]),
            rationale=str(data.get("rationale", "")),
            expected_effect=str(data.get("expected_effect", "")),
            trigger=str(data.get("trigger", "")),
        )


def enforce_bedrock_only(overlay: Any) -> None:
    """Reject overlays that would change the classifier provider.

    Raises :class:`RemediationError` on any forbidden key.  Safe to
    call on empty overlays (no-op).  Invariant enforcement is
    deliberately independent of the rest of overlay validation so the
    reason a proposal was rejected is unambiguous.

    The parameter is typed ``Any`` because this function is the trust
    boundary for JSON coming in through the CLI — the shape check is
    load-bearing even though the dataclass annotation claims ``dict``.
    """
    if not isinstance(overlay, dict):
        raise RemediationError("overlay must be a dict")

    violations = sorted(k for k in overlay if k in PROVIDER_AFFECTING_KEYS)
    if violations:
        raise RemediationError(
            "Bedrock-only invariant violated: overlay touches classifier "
            f"provider routing ({', '.join(violations)}). Overwatch may "
            "adapt scaffolding (sampling, batching, discounts, GPU knobs, "
            "curated reference) but classification must continue to route "
            "through the Bedrock Claude endpoint configured at deploy time."
        )


def validate_proposal(proposal: OverlayProposal) -> list[str]:
    """Validate a proposal against SETTINGS_METADATA + the Bedrock invariant.

    Returns the list of overlay keys that passed validation.  Raises
    :class:`RemediationError` on any violation — partial application is
    intentionally not supported so operators never see a half-applied
    proposal.
    """
    enforce_bedrock_only(proposal.overlay)

    # Late import to keep this module light and avoid a config_overlay
    # import at package import time (the CLIs use this module directly).
    try:
        from atelier.config_overlay import SETTINGS_METADATA, _validate
    except Exception as exc:
        raise RemediationError(f"config_overlay unavailable: {exc}") from exc

    accepted: list[str] = []
    for key, value in proposal.overlay.items():
        if key not in SETTINGS_METADATA:
            raise RemediationError(
                f"unknown setting in overlay: {key!r}. Supervisor may only "
                f"overlay keys that appear in SETTINGS_METADATA (these are "
                f"the same controls the operator can tune in the UI)."
            )
        try:
            _validate(key, value)
        except ValueError as exc:
            raise RemediationError(str(exc)) from exc
        accepted.append(key)

    return accepted
