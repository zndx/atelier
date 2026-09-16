"""AtelierFlow — Metaflow FlowSpec with platform discovery.

YK claims for classification follow the AgentRTC catalogue pattern
(embedding for ColBERT-Zero, light for ModernBERT NHSVM fit). This
base does not steal thinking GPUs.
"""

from __future__ import annotations

from metaflow import FlowSpec


class AtelierFlow(FlowSpec):
    """Base flow. Subclasses declare YK needs on the catalogue, not as heavy."""

    # Documentation of the MaxSim encoder claim (ColBERT-Zero). Catalogue
    # claims[] are the SoR; this is the FlowSpec-side reminder.
    gpu_tokens: int = 0
    model: str = ""
