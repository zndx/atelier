"""ClassificationFlow — one Metaflow sequence for optimize + classify.

The class body stays ``@step``. ``python -m atelier.flows.run_classify``
inserts ``--with kubernetes`` so ``airflow create`` emits a
KubernetesPodOperator per step. The decisions sweep is a foreach of
width 2, one shard per d3-flash replica, so the parallel System One
calls are two Metaflow tasks. The revisit is Engine/Complete
capability thinking. Fusion replays the joined labels once.
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
    reference_id = Parameter("reference-id", default="", type=str)
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
        rid = str(self.reference_id or "").strip()
        self.reference_source_id = rid
        self.train_source_id = rid or sid
        if rid:
            from atelier.sdg.pair import assert_pair

            report = assert_pair(sid, rid)
            self.pair_perfect_possible = report.perfect_possible
            self.pair_skos_missing = list(report.skos_missing)
        self.next(self.probe)

    @step
    def probe(self):
        """Cheap SKOS/vocab + encoder identity check. No GPU claim."""
        from atelier.config import load_config
        from atelier.flows.resident import probe_status

        train_id = str(getattr(self, "train_source_id", None) or self.sample_source_id)
        st = probe_status(load_config(), train_id)
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
        stopping = bool(self.only_precondition or spec.stop_after_precondition)
        # When we will classify the holdout, finalize every missing
        # reference artifact (MaxSim + NHSVM). Stage filters are for
        # artifact-only iteration (phase:maxsim / phase:precondition).
        stages = spec.precondition_stages if stopping else None
        self.precondition_ran = False
        if self.needs_precondition or spec.precondition_stages or not stopping:
            train_id = str(getattr(self, "train_source_id", None) or self.sample_source_id)
            self.precondition_ran = run_precondition_if_needed(
                load_config(), train_id,
                stages=stages,
            )
        if stopping:
            self.classifications = []
            self.next(self.end)
            return
        self.next(self.load)

    @step
    def load(self):
        """LOADING_VOCAB + DISCOVERING + SAMPLING — pin target entities."""
        from atelier.classify.decisions_backend import REPLICA_COUNT

        self.sweep_capability = llm_capability_for_sweep(revisit=False)
        self.sweep_width = REPLICA_COUNT
        self.next(self.sweep)

    @step
    def sweep(self):
        """Fan out one shard per d3-flash replica. Each shard calls System One."""
        width = int(getattr(self, "sweep_width", None) or 2)
        self.sweep_shards = list(range(width))
        self.next(self.sweep_shard, foreach="sweep_shards")

    @step
    def sweep_shard(self):
        """Walk this replica's columns. The other shard is the other replica."""
        from atelier.config import load_config
        from atelier.flows.resident import run_decisions_shard

        shard = int(self.input)
        self.shard_index = shard
        train_id = str(getattr(self, "train_source_id", None) or self.sample_source_id)
        self.shard_labels = run_decisions_shard(
            load_config(),
            str(self.sample_source_id),
            shard,
            width=int(getattr(self, "sweep_width", None) or 2),
            taxonomy_id=train_id,
        )
        self.next(self.sweep_join)

    @step
    def sweep_join(self, inputs):
        """Merge the two System One shards. Overlap fails closed."""
        from atelier.flows.classify_phases import merge_shard_labels

        self.decision_labels = merge_shard_labels(
            list(getattr(item, "shard_labels", []) or []) for item in inputs
        )
        self.classifications = [
            {
                "qualified_name": row["qualified_name"],
                "predicted_code": row.get("predicted_code") or "",
            }
            for row in self.decision_labels
        ]
        declared = list(getattr(self, "target_keys", []) or [])
        if not declared:
            self.target_keys = [
                c["qualified_name"] for c in self.classifications if c["qualified_name"]
            ]
        self.next(self.revisit)

    @step
    def revisit(self):
        """Thinking revisit via Engine/Complete over the decisions labels."""
        from atelier.config import load_config
        from atelier.flows.resident import run_dst_revisit

        rows = []
        for row in self.classifications:
            qualified = str(row.get("qualified_name") or "")
            if not qualified:
                continue
            rows.append({
                "qualified_name": qualified,
                "column_name": qualified.split(".")[-1],
                "predicted_code": row.get("predicted_code") or "",
            })
        revisited = run_dst_revisit(load_config(), rows=rows)
        by_qualified = {
            row["qualified_name"]: row["predicted_code"]
            for row in revisited["classifications"]
            if row.get("qualified_name") and row.get("predicted_code")
        }
        updated = []
        for row in self.classifications:
            code = by_qualified.get(str(row.get("qualified_name") or ""))
            if code:
                row = {**row, "predicted_code": code}
            updated.append(row)
        self.classifications = updated
        self.next(self.fuse)

    @step
    def fuse(self):
        """One DST pass. MaxSim and NHSVM see the decision labels via replay."""
        from atelier.classify.decisions_backend import ReplayDecisionsBackend
        from atelier.config import load_config
        from atelier.flows.resident import run_dst_pipeline

        skip = (not self.needs_precondition) or bool(self.precondition_ran)
        train_id = str(getattr(self, "train_source_id", None) or self.sample_source_id)
        result = run_dst_pipeline(
            load_config(), str(self.sample_source_id),
            skip_precondition=skip,
            taxonomy_id=train_id,
            include_revisit=False,
            llm_backend=ReplayDecisionsBackend(list(self.decision_labels)),
        )
        if result.get("state") == "ERROR":
            raise RuntimeError(result.get("error") or "classification pipeline ERROR")
        self.result_path = result.get("result_path") or ""
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

