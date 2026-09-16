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
        from atelier.config import load_config
        from atelier.flows.resident import probe_status

        st = probe_status(load_config(), str(self.source_id))
        self.needs_precondition = probe_requires_precondition(st)
        self.probe_reasons = list(getattr(st, "reasons", []) or [])
        self.next(self.precondition)

    @step
    def precondition(self):
        """Enrich + ColBERT-Zero collection + ModernBERT NHSVM only if stale."""
        from atelier.config import load_config
        from atelier.flows.resident import run_precondition_if_needed

        self.precondition_ran = False
        if self.needs_precondition:
            self.precondition_ran = run_precondition_if_needed(
                load_config(), str(self.source_id),
            )
        if self.only_precondition:
            self.target_keys = []
            self.classifications = []
            self.next(self.evaluate)
            return
        self.next(self.load)

    @step
    def load(self):
        """LOADING_VOCAB + DISCOVERING + SAMPLING — pin target entities."""
        self.sweep_capability = llm_capability_for_sweep(revisit=False)
        self.next(self.sweep)

    @step
    def sweep(self):
        """LLM_SWEEP ⇄ VALIDATING via Complete instruct (thinking on revisit)."""
        from atelier.config import load_config
        from atelier.flows.classify_phases import entity_key, predicted_code
        from atelier.flows.resident import load_classification_rows, run_dst_pipeline

        skip = (not self.needs_precondition) or bool(self.precondition_ran)
        result = run_dst_pipeline(
            load_config(), str(self.source_id), skip_precondition=skip,
        )
        if result.get("state") == "ERROR":
            raise RuntimeError(result.get("error") or "classification pipeline ERROR")
        rows = load_classification_rows(result)
        self.result_path = result.get("result_path") or ""
        self.classifications = [
            {
                "qualified_name": entity_key(r) or str(r.get("column_name") or ""),
                "predicted_code": predicted_code(r),
            }
            for r in rows
        ]
        self.target_keys = [c["qualified_name"] for c in self.classifications if c["qualified_name"]]
        self.next(self.fuse)

    @step
    def fuse(self):
        """CLASSIFYING + FUSING. ColBERT-Zero MaxSim is inside the DST pipeline."""
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


if __name__ == "__main__":
    import os

    from atelier.flows.platform_metaflow import metaflow_child_env

    os.environ.update(metaflow_child_env())
    ClassificationFlow()

