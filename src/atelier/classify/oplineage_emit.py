"""OpenLineage event projection for Atelier classify + extend runs.

Builds OpenLineage-shaped event dicts from the rows we already persist
in PG (``fsm_runs``, ``datasets``, ``ml_artifact_sets``).  Day one,
nothing wires these to a transport — the function is a pure projection
so an operator who configures Marquez (or any other OpenLineage
backend) later only needs to add HTTP POST plumbing, not re-model the
data.

The shape follows the OpenLineage Object Model:

  Run    = atelier ``fsm_runs.id``
  Job    = "atelier.classify" or "atelier.extend_classify"
  Inputs = source's Hive tables (one Dataset per discovered table) —
           synthesized from ``classification_runs`` rows when available
           and falling back to a single source-level dataset otherwise.
  Outputs= the parquet at ``datasets.parquet_path`` AND, for classify
           runs, the ML artifact bundle (CatBoost / SVM / UMAP) keyed
           by their on-disk paths.

Custom facets:

- ``zndx_ml_artifact``: framework, has_svm, has_umap, vocab_signature,
  embedding_model, classes_count.  Attached to the parquet output
  (matches the Metaflow-flavored idea that "the run produced this
  bundle of artifacts and they are addressable as a unit").
- ``zndx_extend_lineage``: artifact_set_id consumed +
  parent_dataset_id linked to.  Attached only to extend runs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

# Producer URL — used by OpenLineage to identify the system that emitted
# the event.  No external dependency required (we don't need
# openlineage-python's HTTP transport for this projection step).
PRODUCER = "https://github.com/zndx/atelier"


def build_run_event(
    *,
    fsm_run: dict,
    dataset: dict | None = None,
    artifact_set: dict | None = None,
    parent_artifact_set: dict | None = None,
    event_type: str = "COMPLETE",
) -> dict:
    """Project an FSM run into an OpenLineage event dict.

    Args:
        fsm_run: Row from ``fsm_runs`` (``state``, ``id``, ``started_at``,
            ``updated_at``, ``source_id``, ``config``).
        dataset: Output dataset row, when one was registered for this
            run.  Carries ``parquet_path``, ``run_kind``, ``artifact_set_id``,
            ``parent_dataset_id``.  None for failed runs.
        artifact_set: For classify runs, the ArtifactSet this run
            produced.  For extend runs, the ArtifactSet this run
            consumed.  None when the run failed before producing one.
        parent_artifact_set: For extend runs, the upstream ArtifactSet
            (same as ``artifact_set`` when the extend reused exactly
            one bundle).  Provided separately to keep the relationship
            explicit in the emitted ParentRunFacet.
        event_type: One of ``START`` / ``RUNNING`` / ``COMPLETE`` /
            ``ABORT`` / ``FAIL``.  Defaults to COMPLETE; callers
            emitting at start time pass ``START``.

    Returns:
        OpenLineage event dict, ready to JSON-serialize and POST to a
        Marquez (or compatible) backend.  No transport is performed
        here.
    """
    run_kind = (dataset or {}).get("run_kind") or "classify"
    job_name = (
        "atelier.extend_classify" if run_kind == "extend"
        else "atelier.classify"
    )

    inputs = _input_datasets(fsm_run)
    outputs = _output_datasets(dataset, artifact_set)

    run_facets: dict[str, Any] = {
        "nominalTime": {
            "_producer": PRODUCER,
            "_schemaURL": (
                "https://openlineage.io/spec/facets/1-0-0/"
                "NominalTimeRunFacet.json"
            ),
            "nominalStartTime": fsm_run.get("started_at") or "",
            "nominalEndTime": fsm_run.get("updated_at") or "",
        },
        "zndx_execution_context": {
            "_producer": PRODUCER,
            "_schemaURL": (
                "https://github.com/zndx/atelier/spec/facets/"
                "ExecutionContextFacet.json"
            ),
            "run_kind": run_kind,
            "fsm_state": fsm_run.get("state", ""),
            "source_id": fsm_run.get("source_id", ""),
        },
    }

    # ParentRunFacet — extend runs link back to the run that produced
    # the consumed artifact set.  Mirrors the OpenLineage convention
    # that downstream runs carry their ancestry as a facet rather than
    # an inline reference.
    if (
        parent_artifact_set
        and parent_artifact_set.get("fsm_run_id")
        and run_kind == "extend"
    ):
        run_facets["parentRun"] = {
            "_producer": PRODUCER,
            "_schemaURL": (
                "https://openlineage.io/spec/facets/1-0-0/"
                "ParentRunFacet.json"
            ),
            "run": {"runId": parent_artifact_set["fsm_run_id"]},
            "job": {
                "namespace": "atelier",
                "name": "atelier.classify",
            },
        }

    return {
        "eventType": event_type,
        "eventTime": _now_iso(),
        "producer": PRODUCER,
        "schemaURL": (
            "https://openlineage.io/spec/2-0-0/OpenLineage.json"
            "#/$defs/RunEvent"
        ),
        "run": {
            "runId": fsm_run["id"],
            "facets": run_facets,
        },
        "job": {
            "namespace": "atelier",
            "name": job_name,
        },
        "inputs": inputs,
        "outputs": outputs,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _input_datasets(fsm_run: dict) -> list[dict]:
    """Synthesize Dataset entries for the run's input tables.

    For OOTB sample / synthetic / meta-tagging sources, emit a single
    dataset keyed by the source id.  For Hive sources, the table list
    isn't reachable from this function alone — we emit a single
    aggregate dataset and leave per-table expansion as a future
    enhancement (it would require querying ``classification_runs`` for
    distinct table_name values, which adds a DB hit per emit).
    """
    source_id = fsm_run.get("source_id") or "unknown"
    return [
        {
            "namespace": "atelier",
            "name": f"source/{source_id}",
            "facets": {
                "schema": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-0/"
                        "SchemaDatasetFacet.json"
                    ),
                    "fields": [
                        {"name": "table_name", "type": "string"},
                        {"name": "column_name", "type": "string"},
                        {"name": "column_type", "type": "string"},
                    ],
                },
            },
        }
    ]


def _output_datasets(
    dataset: dict | None,
    artifact_set: dict | None,
) -> list[dict]:
    """Build output Dataset entries for the run.

    Includes the parquet (always, when registered) and a separate
    Dataset per artifact file (CatBoost / SVM / UMAP) so OpenLineage
    consumers can show "this run produced N model artifacts".  Each
    artifact carries the custom ``zndx_ml_artifact`` facet with the
    framework + bundle metadata.
    """
    outputs: list[dict] = []

    if dataset and dataset.get("parquet_path"):
        outputs.append({
            "namespace": "atelier",
            "name": dataset["parquet_path"],
            "facets": {
                "schema": {
                    "_producer": PRODUCER,
                    "_schemaURL": (
                        "https://openlineage.io/spec/facets/1-0-0/"
                        "SchemaDatasetFacet.json"
                    ),
                    "fields": _classify_parquet_schema_fields(),
                },
                "zndx_extend_lineage": _extend_lineage_facet(dataset),
            },
        })

    if artifact_set:
        for path_field, role in (
            ("catboost_path", "catboost_classifier"),
            ("svm_path", "svm_frontier_classifier"),
            ("umap_path", "umap_projection"),
        ):
            path = artifact_set.get(path_field)
            if not path:
                continue
            outputs.append({
                "namespace": "atelier",
                "name": path,
                "facets": {
                    "zndx_ml_artifact": {
                        "_producer": PRODUCER,
                        "_schemaURL": (
                            "https://github.com/zndx/atelier/spec/facets/"
                            "MLArtifactFacet.json"
                        ),
                        "role": role,
                        "framework": (
                            "catboost" if "catboost" in role
                            else "sklearn" if "svm" in role
                            else "umap-learn"
                        ),
                        "vocab_signature": artifact_set.get("vocab_signature", ""),
                        "embedding_model": artifact_set.get("embedding_model", ""),
                        "embedding_dim": artifact_set.get("embedding_dim", 0),
                        "classes_count": _classes_count(artifact_set),
                    },
                },
            })

    return outputs


def _classify_parquet_schema_fields() -> list[dict]:
    """Atlas-compatible parquet schema fields (matches _write_parquet)."""
    return [
        {"name": "text", "type": "string"},
        {"name": "x", "type": "double"},
        {"name": "y", "type": "double"},
        {"name": "table_name", "type": "string"},
        {"name": "column_name", "type": "string"},
        {"name": "column_type", "type": "string"},
        {"name": "predicted_code", "type": "string"},
        {"name": "predicted_label", "type": "string"},
        {"name": "confidence", "type": "double"},
        {"name": "belief", "type": "double"},
        {"name": "plausibility", "type": "double"},
        {"name": "uncertainty", "type": "double"},
        {"name": "conflict", "type": "double"},
    ]


def _extend_lineage_facet(dataset: dict) -> dict:
    """Custom facet recording the artifact set + parent dataset for extend."""
    return {
        "_producer": PRODUCER,
        "_schemaURL": (
            "https://github.com/zndx/atelier/spec/facets/"
            "ExtendLineageFacet.json"
        ),
        "run_kind": dataset.get("run_kind", "classify"),
        "artifact_set_id": dataset.get("artifact_set_id"),
        "parent_dataset_id": dataset.get("parent_dataset_id"),
    }


def _classes_count(artifact_set: dict) -> int:
    """Decode the JSON-encoded classes column to an int count."""
    raw = artifact_set.get("classes")
    if not raw:
        return 0
    try:
        return len(json.loads(raw))
    except Exception:
        return 0
