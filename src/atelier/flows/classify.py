"""ClassificationFlow — one Metaflow sequence for optimize + classify.

K8s (@kubernetes) is preferred on the discovered Signals Metaflow
instance. Plain @step is the same graph and is the fallback while K8s
is unavailable. LLM is Engine/Complete thinking|instruct only.
"""

from __future__ import annotations

from metaflow import Parameter, step

from atelier.flows.base import AtelierFlow
from atelier.flows.classify_phases import (
    assert_coverage,
    llm_capability_for_sweep,
    probe_requires_precondition,
)
from atelier.flows import register_flow


@register_flow("sdg-classify")
class ClassificationFlow(AtelierFlow):
    """Resident classification: probe-gated precondition through coverage evaluate."""

    source_id = Parameter("source-id", default="sdg-corpora", type=str)
    only_precondition = Parameter("only-precondition", default=False, type=bool)
    kubernetes_preferred = Parameter("kubernetes-preferred", default=True, type=bool)

    @step
    def start(self):
        from atelier.flows.platform_metaflow import (
            apply_metaflow_config,
            require_signals_metaflow,
        )

        apply_metaflow_config()
        require_signals_metaflow()
        self.precondition_ran = False
        self.classifications: list[dict] = []
        self.target_keys: list[str] = []
        self.next(self.probe)

    @step
    def probe(self):
        """Cheap SKOS/vocab + encoder identity check. No GPU claim."""
        self.needs_precondition = True
        self.next(self.precondition)

    @step
    def precondition(self):
        """Enrich + ColBERT-Zero collection + ModernBERT NHSVM only if stale."""
        if not probe_requires_precondition({"final": not self.needs_precondition}):
            self.precondition_ran = False
        else:
            self.precondition_ran = True
        if self.only_precondition:
            self.target_keys = []
            self.classifications = []
            self.next(self.evaluate)
            return
        self.next(self.load)

    @step
    def load(self):
        """LOADING_VOCAB + DISCOVERING + SAMPLING — pin target entities."""
        self.next(self.sweep)

    @step
    def sweep(self):
        """LLM_SWEEP ⇄ VALIDATING via Complete instruct (thinking on revisit)."""
        self.sweep_capability = llm_capability_for_sweep(revisit=False)
        self.next(self.fuse)

    @step
    def fuse(self):
        """CLASSIFYING + FUSING. ColBERT-Zero MaxSim fail-closed."""
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Pass iff every target relational entity has a predicted_code."""
        assert_coverage(self.classifications, self.target_keys)
        self.next(self.publish)

    @step
    def publish(self):
        """Embeddings parquet + report on discovered RustFS (wired in a later PR)."""
        pass
