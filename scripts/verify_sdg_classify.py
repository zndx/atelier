#!/usr/bin/env python3
"""Classify the pinned sdg-corpora sample and score vs SKOS/ontology.

Ground truth is the sample's annotations.csv (SDG SKOS), not a hand CSV.
A run with zero reference_code is unscored — fail closed.
"""
from __future__ import annotations

import dataclasses as _dc
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from atelier.classify.evaluation import evaluate_classifications
from atelier.classify.skos_attach import attach_skos_reference
from atelier.classify.fsm import AgentFSM
from atelier.classify.mock_llm import RealisticMockLLMBackend
from atelier.classify.pipeline import run_classification_pipeline
from atelier.classify.sampler import load_sample_source
from atelier.classify.taxonomy import load_annotations_from_filesystem
from atelier.config import load_config


def _sample_dir() -> Path:
    pointer = REPO / "build" / "sdg_sample" / "current.json"
    if pointer.is_file():
        loc = json.loads(pointer.read_text()).get("path") or ""
        p = Path(loc)
        if p.is_dir():
            return p
    matches = sorted((REPO / "build" / "sdg_sample").glob("*_macbook"))
    if matches:
        return matches[-1]
    raise SystemExit("no sdg_sample; run: just sdg-sample")


def main() -> int:
    sample_dir = _sample_dir()
    print(f"sdg_sample: {sample_dir}")
    tables = load_sample_source(sample_dir)
    if not tables:
        print("FAIL: no tables in sdg_sample", file=sys.stderr)
        return 2
    cats = load_annotations_from_filesystem(sample_dir, hierarchical=True, taxonomy="sdg")
    ref = attach_skos_reference(tables, cats)
    n_cols = sum(len(t.columns) for t in tables)
    n_ref = sum(1 for t in tables for c in t.columns if c.reference_code)
    vocab_n = len(getattr(cats, "all_by_code", {}) or {})
    if not isinstance(getattr(cats, "all_by_code", None), dict):
        vocab_n = len(list(getattr(cats, "categories", []) or []))
    print(f"tables={len(tables)} columns={n_cols} skos_reference={n_ref} vocab={vocab_n}")
    if n_ref == 0:
        print("FAIL: SDG ontology produced zero reference_code (unscored)", file=sys.stderr)
        return 2

    cfg = load_config()
    # Injected samples + mock LLM: skip in-situ OpenAI enrichment (that
    # stage needs a live key; this verify is classify + SKOS scoring).
    cfg = _dc.replace(
        cfg,
        classify_precondition_enabled=False,
        classify_svm_source="auto",
    )
    llm = RealisticMockLLMBackend(ref, base_accuracy=0.8, seed=15)
    fsm = AgentFSM()
    subset = tables[:4]
    result = run_classification_pipeline(
        cfg,
        fsm,
        source_id="sdg-corpora",
        samples=subset,
        category_set=cats,
        llm_backend=llm,
        tables_limit=4,
    )
    classifications = result.get("classifications") or result.get("columns") or []
    if result.get("state") == "ERROR" or not classifications:
        err = (result.get("error") or "no classifications")[:350]
        print(f"pipeline fail-closed: {err}")
        classifications = []
        for table in subset:
            resp = llm.classify_batch(table.columns, system_prompt="")
            by_name = {c.column_name: c for c in resp.classifications}
            for col in table.columns:
                pred = by_name.get(col.name)
                classifications.append(
                    {
                        "predicted_code": getattr(pred, "category_code", None) or "",
                        "reference_code": col.reference_code or "",
                        "confidence": getattr(pred, "confidence", 0) or 0,
                        "belief": 0.0,
                        "plausibility": 0.0,
                        "uncertainty": 0.0,
                        "conflict": 0.0,
                        "evidence_sources": ["llm_mock"],
                    }
                )
        result = {"run_id": "sdg_verify_llm_only", "state": "ERROR"}
    report = evaluate_classifications(classifications, category_set=cats, run_id=str(result.get("run_id") or ""))
    print(report.summary())
    out = REPO / "build" / "results" / "sdg_verify" / "evaluation_report.json"
    report.write_json(out)
    print(f"wrote {out}")
    if report.columns_with_reference < 1:
        print("FAIL: evaluation had no ontology-grounded columns", file=sys.stderr)
        return 2
    if result.get("state") == "ERROR":
        print(
            "FAIL: DST pipeline did not CONVERGE (Qdrant MaxSim / NHSVM head "
            "required). Ontology score above is LLM-only vs SKOS.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
