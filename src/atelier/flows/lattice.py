"""Signals lattice helpers for Metaflow steps.

LLM Complete (thinking / instruct, OIP) lands with ClassificationFlow.
This module owns datastore fail-closed today.
"""

from __future__ import annotations

from typing import Mapping

from atelier.flows.platform_metaflow import (
    GURU_NOPLATFORM,
    PlatformMetaflowError,
    require_signals_metaflow as _require,
)

GURU_NOLATTICE = "#GR.00000002.NOLATTICE"
GURU_NOCAP = "#GR.00000003.NOCAP"
GURU_UNHEALTHY = "#EP.00000016.NOTREADY"
GURU_WRONGMODEL = "#EP.00000018.WRONGMODEL"

# Operating profiles on the resident Qwen — not a referee capability.
COMPLETE_CAPABILITIES = ("thinking", "instruct")


def require_signals_metaflow(environ: Mapping[str, str] | None = None) -> None:
    _require(environ)


def assert_complete_capability(capability: str) -> str:
    cap = (capability or "").strip().lower()
    if cap not in COMPLETE_CAPABILITIES:
        raise RuntimeError(
            f"{GURU_NOCAP} Complete capability must be thinking or instruct "
            f"(got {capability!r}); referee is not a lattice LLM."
        )
    return cap


__all__ = (
    "COMPLETE_CAPABILITIES",
    "GURU_NOLATTICE",
    "GURU_NOCAP",
    "GURU_NOPLATFORM",
    "GURU_UNHEALTHY",
    "GURU_WRONGMODEL",
    "PlatformMetaflowError",
    "assert_complete_capability",
    "require_signals_metaflow",
)
