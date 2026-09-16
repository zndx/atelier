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

    # Empty → current sample id (sdg-corpora/<pin>_<profile>). Bare
    # ``sdg-corpora`` is the full corpus (later), never the default.
    source_id = Parameter("source-id", default="", type=str)
    # collection:<slug>, table:<name>, phase:precondition|maxsim|nhsvm
    target = Parameter("target", default="", type=str)
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
        from atelier.sdg.sample import CORPUS_SOURCE_ID, current_sample_source_id

        sid = str(self.source_id or "").strip()
        if not sid:
            sid = current_sample_source_id()
        if not sid:
            raise RuntimeError(
                "no sdg-corpora sample id; run `just sdg-sample` "
                "(do not default to the full corpus source-id "
                f"{CORPUS_SOURCE_ID!r})"
            )
        # Parameters are immutable; persist the resolved sample id.
        self.sample_source_id = sid
        from atelier.sdg.targets import resolve_sample_target

        spec = resolve_sample_target(sid, str(self.target or ""))
        self.target_raw = spec.raw
        self.target_phase = spec.phase
        self.target_collection = spec.collection
        self.target_tables = list(spec.tables)
        self.target_keys = list(spec.entity_keys)
        self.target_hard_gate = spec.hard_gate
        self.target_notes = list(spec.notes)
        self.precondition_ran = False
        self.classifications: list[dict] = []
        self.next(self.probe)

    @step
    def probe(self):
        """Cheap SKOS/vocab + encoder identity check. No GPU claim."""
        from atelier.config import load_config
        from atelier.flows.resident import probe_status

        st = probe_status(load_config(), str(self.sample_source_id))
        self.needs_precondition = probe_requires_precondition(st)
        self.probe_reasons = list(getattr(st, "reasons", []) or [])
        self.next(self.precondition)

    @step
    def precondition(self):
        """Enrich + ColBERT-Zero collection + ModernBERT NHSVM only if stale."""
        from atelier.config import load_config
        from atelier.flows.resident import run_precondition_if_needed

        from atelier.sdg.targets import resolve_sample_target

        spec = resolve_sample_target(
            str(self.sample_source_id), str(getattr(self, "target_raw", "") or self.target or ""),
        )
        self.precondition_ran = False
        if self.needs_precondition or spec.precondition_stages:
            self.precondition_ran = run_precondition_if_needed(
                load_config(), str(self.sample_source_id),
                stages=spec.precondition_stages,
            )
        if self.only_precondition or spec.stop_after_precondition:
            self.classifications = []
            self.next(self.end)
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
            load_config(), str(self.sample_source_id), skip_precondition=skip,
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
        declared = list(getattr(self, "target_keys", []) or [])
        if declared:
            self.target_keys = declared
        else:
            self.target_keys = [
                c["qualified_name"] for c in self.classifications if c["qualified_name"]
            ]
        self.next(self.fuse)

    @step
    def fuse(self):
        """CLASSIFYING + FUSING. ColBERT-Zero MaxSim is inside the DST pipeline."""
        self.next(self.evaluate)

    @step
    def evaluate(self):
        """Hard gate only for a named sub-target or the full corpus."""
        from atelier.flows.classify_phases import coverage_complete, unclassified_targets
        from atelier.sdg.sample import is_sample_source_id

        keys = list(self.target_keys or [])
        self.unclassified = unclassified_targets(self.classifications, keys)
        self.coverage_ok = coverage_complete(self.classifications, keys)
        hard = bool(getattr(self, "target_hard_gate", False))
        if hard:
            assert_coverage(self.classifications, keys)
        elif is_sample_source_id(str(self.sample_source_id)):
            # Sample without a collection/table slice cannot DST-converge.
            pass
        else:
            assert_coverage(self.classifications, keys)
        self.next(self.end)

    @step
    def end(self):
        """Publish embeddings/report (RustFS in a later PR). Metaflow terminal step."""
        pass


if __name__ == "__main__":
    raise SystemExit(
        "Importing this module constructs the FlowSpec before platform env "
        "is applied (Tilt ~/.metaflowconfig would win). "
        "Run: python -m atelier.flows.run_classify run --source-id=..."
    )

