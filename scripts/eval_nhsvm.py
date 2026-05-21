#!/usr/bin/env python3
"""A/B evaluation of NHSVM tree-distance reweighting vs flat SVM.

Uses the existing trained SVM (build/models/svm.pkl) and the agent-mediated
reference (build/data/agent_mediated/agent_mediated.json) to compare flat
vs NHSVM-reweighted probability distributions.  No retraining — the NHSVM
reweighting is applied at inference time only.

The agent-mediated reference uses mnemonics (e.g. "EMAIL", "INOS"); the
SVM predicts user-taxonomy dot-codes (e.g. "1.1.1.9.3.1", "0.1").  The
working_set vocabulary bridges the two.

Usage:
    python scripts/eval_nhsvm.py
    python scripts/eval_nhsvm.py --temperatures 0.3,0.5,1.0,2.0
    python scripts/eval_nhsvm.py --output build/data/svm_training/nhsvm_evaluation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from atelier.classify.svm_classifier import (
    SVMClassifier,
    build_nhsvm_distance_matrix,
    build_svm_text,
    nhsvm_reweight,
)
from atelier.classify.taxonomy import _build_category_set_from_records

logger = logging.getLogger(__name__)


def _load_user_taxonomy(working_set_path: Path):
    """Build HierarchicalCategorySet from working_set.json vocabulary.

    The working_set uses the deployment's dot-code taxonomy (e.g.
    "0.1" for INOS), NOT the upstream ICE/BFO/CCO pretraining codes.
    """
    ws = json.loads(working_set_path.read_text())
    vocab = ws["vocabulary"]
    records = [
        {
            "id": info["code"],
            "ontology": info["label"],
            "annotation": mnemonic,
            "definition": info.get("definition", ""),
            "deprecated": info.get("deprecated", False),
        }
        for mnemonic, info in vocab.items()
    ]
    cs = _build_category_set_from_records(records, hierarchical=True)
    mnemonic_to_code = {k: v["code"] for k, v in vocab.items()}
    return cs, mnemonic_to_code


def _evaluate_temperature(
    svm: SVMClassifier,
    reference: dict[str, str],
    mnemonic_to_code: dict[str, str],
    category_set,
    alphas: dict[str, float],
    distance_matrix: dict[tuple[str, str], float],
    temperature: float,
    svm_codes: set[str],
) -> dict:
    """Run evaluation at a specific temperature."""
    flat_correct = 0
    nhsvm_correct = 0
    flat_subtree = 0
    nhsvm_subtree = 0
    total = 0
    flips_to_correct = []
    flips_to_wrong = []
    flat_predictions = Counter()
    nhsvm_predictions = Counter()

    for key, ref_tag in reference.items():
        if ref_tag is None or ref_tag not in mnemonic_to_code:
            continue
        ref_code = mnemonic_to_code[ref_tag]
        if ref_code not in category_set.leaf_codes:
            continue
        if ref_code not in svm_codes:
            continue

        total += 1
        parts = key.split(".", 1)
        col_name = parts[1] if len(parts) > 1 else parts[0]
        text = build_svm_text(col_name)
        proba = svm.predict_proba_single(text)

        flat_top1 = max(proba, key=proba.get)
        rw = nhsvm_reweight(
            proba, alphas, category_set, temperature,
            distance_matrix=distance_matrix,
        )
        nhsvm_top1 = max(rw, key=rw.get)

        flat_predictions[flat_top1] += 1
        nhsvm_predictions[nhsvm_top1] += 1

        if flat_top1 == ref_code:
            flat_correct += 1
        if nhsvm_top1 == ref_code:
            nhsvm_correct += 1

        ref_prefix = ref_code.split(".")[:2]
        if flat_top1.split(".")[:2] == ref_prefix:
            flat_subtree += 1
        if nhsvm_top1.split(".")[:2] == ref_prefix:
            nhsvm_subtree += 1

        if flat_top1 != nhsvm_top1:
            if nhsvm_top1 == ref_code and flat_top1 != ref_code:
                flips_to_correct.append({
                    "column": key,
                    "flat": flat_top1,
                    "nhsvm": nhsvm_top1,
                    "reference": ref_code,
                })
            elif flat_top1 == ref_code and nhsvm_top1 != ref_code:
                flips_to_wrong.append({
                    "column": key,
                    "flat": flat_top1,
                    "nhsvm": nhsvm_top1,
                    "reference": ref_code,
                })

    return {
        "temperature": temperature,
        "total_evaluated": total,
        "flat_exact_accuracy": flat_correct / total if total else 0,
        "nhsvm_exact_accuracy": nhsvm_correct / total if total else 0,
        "flat_subtree_accuracy": flat_subtree / total if total else 0,
        "nhsvm_subtree_accuracy": nhsvm_subtree / total if total else 0,
        "flat_exact_correct": flat_correct,
        "nhsvm_exact_correct": nhsvm_correct,
        "exact_delta": nhsvm_correct - flat_correct,
        "subtree_delta": nhsvm_subtree - flat_subtree,
        "flips_to_correct": len(flips_to_correct),
        "flips_to_wrong": len(flips_to_wrong),
        "net_gain": len(flips_to_correct) - len(flips_to_wrong),
        "top1_changes": len(flips_to_correct) + len(flips_to_wrong),
        "flip_details_correct": flips_to_correct[:10],
        "flip_details_wrong": flips_to_wrong[:10],
        "flat_inos_predictions": flat_predictions.get("0.1", 0),
        "nhsvm_inos_predictions": nhsvm_predictions.get("0.1", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="NHSVM A/B evaluation")
    parser.add_argument(
        "--svm", default="build/models/svm.pkl",
        help="Path to trained SVM model",
    )
    parser.add_argument(
        "--reference", default="build/data/agent_mediated/agent_mediated.json",
        help="Path to agent-mediated reference",
    )
    parser.add_argument(
        "--working-set", default="build/data/agent_mediated/working_set.json",
        help="Path to working_set.json (user taxonomy)",
    )
    parser.add_argument(
        "--temperatures", default="0.3,0.5,0.7,1.0,1.5,2.0",
        help="Comma-separated temperature values to evaluate",
    )
    parser.add_argument(
        "--output", default="build/data/svm_training/nhsvm_evaluation.json",
        help="Output path for evaluation results",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    svm = SVMClassifier.load(args.svm)
    svm_codes = set(svm._classes)
    logger.info("Loaded SVM with %d classes", len(svm_codes))

    reference = json.loads(Path(args.reference).read_text())
    logger.info("Loaded reference with %d entries", len(reference))

    cs, mnemonic_to_code = _load_user_taxonomy(Path(args.working_set))
    logger.info(
        "Loaded user taxonomy: %d leaves, %d total nodes",
        len(cs.leaf_codes), len(cs.all_categories),
    )

    alphas = cs.compute_nhsvm_alphas()
    logger.info("Computed NHSVM alpha weights for %d nodes", len(alphas))

    t0 = time.time()
    distance_matrix = build_nhsvm_distance_matrix(
        list(svm_codes), alphas, cs,
    )
    logger.info(
        "Precomputed distance matrix (%d pairs) in %.1fs",
        len(distance_matrix), time.time() - t0,
    )

    temperatures = [float(t) for t in args.temperatures.split(",")]
    results = []
    for temp in temperatures:
        t0 = time.time()
        result = _evaluate_temperature(
            svm, reference, mnemonic_to_code, cs, alphas,
            distance_matrix, temp, svm_codes,
        )
        elapsed = time.time() - t0
        results.append(result)

        delta = result["exact_delta"]
        delta_sign = "+" if delta >= 0 else ""
        print(
            f"T={temp:.1f}: exact {result['nhsvm_exact_accuracy']:.1%} "
            f"[{delta_sign}{delta}]  "
            f"subtree {result['nhsvm_subtree_accuracy']:.1%} "
            f"[{delta_sign}{result['subtree_delta']}]  "
            f"flips +{result['flips_to_correct']}/-{result['flips_to_wrong']}  "
            f"({elapsed:.1f}s)"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "evaluation_type": "nhsvm_ab_comparison",
        "svm_model": str(args.svm),
        "reference": str(args.reference),
        "taxonomy_source": str(args.working_set),
        "svm_classes": len(svm_codes),
        "taxonomy_leaves": len(cs.leaf_codes),
        "taxonomy_nodes": len(cs.all_categories),
        "flat_baseline_exact": results[0]["flat_exact_accuracy"],
        "flat_baseline_subtree": results[0]["flat_subtree_accuracy"],
        "results_by_temperature": results,
    }, indent=2))
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
