# Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
#
# This file contains material proprietary to Cloudera, Inc., and is provided
# to authorized licensees solely for use in connection with the Cloudera AI
# (CAI) Application from which it was obtained.  It may not be copied,
# modified, redistributed, or used in any other manner without the express
# written consent of Cloudera, Inc.

"""Structured evaluation framework for classification results.

Produces per-category precision/recall/F1, confusion matrix,
hierarchical accuracy (ancestor/descendant tolerance), and
aggregate DST metrics (mean belief, plausibility, conflict).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PerCategoryMetrics:
    code: str
    label: str
    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
        }


@dataclass(frozen=True)
class ConfusionEntry:
    true_code: str
    predicted_code: str
    count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "true_code": self.true_code,
            "predicted_code": self.predicted_code,
            "count": self.count,
        }


@dataclass(frozen=True)
class EvaluationReport:
    run_id: str
    timestamp: str
    total_columns: int
    columns_with_reference: int
    exact_accuracy: float
    micro_f1: float
    macro_f1: float
    hierarchical_accuracy: float
    per_category: list[PerCategoryMetrics]
    confusion_matrix: list[ConfusionEntry]
    mean_belief: float
    mean_plausibility: float
    mean_uncertainty_gap: float
    mean_conflict: float
    evidence_sources_used: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "total_columns": self.total_columns,
            "columns_with_reference": self.columns_with_reference,
            "exact_accuracy": round(self.exact_accuracy, 4),
            "micro_f1": round(self.micro_f1, 4),
            "macro_f1": round(self.macro_f1, 4),
            "hierarchical_accuracy": round(self.hierarchical_accuracy, 4),
            "per_category": [pc.to_dict() for pc in self.per_category],
            "confusion_matrix": [ce.to_dict() for ce in self.confusion_matrix],
            "mean_belief": round(self.mean_belief, 4),
            "mean_plausibility": round(self.mean_plausibility, 4),
            "mean_uncertainty_gap": round(self.mean_uncertainty_gap, 4),
            "mean_conflict": round(self.mean_conflict, 4),
            "evidence_sources_used": self.evidence_sources_used,
        }

    def write_json(self, path: Path) -> None:
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def summary(self) -> str:
        lines = [
            f"Evaluation Report ({self.run_id})",
            f"  {self.columns_with_reference}/{self.total_columns} columns with curated reference",
            f"  Exact accuracy:        {self.exact_accuracy:.4f}",
            f"  Micro F1:              {self.micro_f1:.4f}",
            f"  Macro F1:              {self.macro_f1:.4f}",
            f"  Hierarchical accuracy: {self.hierarchical_accuracy:.4f}",
            f"  Mean belief:           {self.mean_belief:.4f}",
            f"  Mean plausibility:     {self.mean_plausibility:.4f}",
            f"  Mean uncertainty gap:  {self.mean_uncertainty_gap:.4f}",
            f"  Mean conflict:         {self.mean_conflict:.4f}",
            f"  Evidence sources:      {', '.join(self.evidence_sources_used)}",
        ]
        if self.per_category:
            lines.append("  Per-category (top 10 by support):")
            top = sorted(self.per_category, key=lambda pc: -pc.support)[:10]
            for pc in top:
                lines.append(
                    f"    {pc.code:40s} P={pc.precision:.2f} R={pc.recall:.2f} "
                    f"F1={pc.f1:.2f} n={pc.support}"
                )
        return "\n".join(lines)


def evaluate_classifications(
    classifications: list[dict[str, Any]],
    category_set=None,
    run_id: str | None = None,
) -> EvaluationReport:
    """Build a structured EvaluationReport from classification dicts.

    Parameters
    ----------
    classifications:
        List of dicts as returned by ``_classify_column()`` in pipeline.py.
        Each dict must have at minimum: predicted_code, reference_code, confidence,
        belief, plausibility, uncertainty, conflict, evidence_sources.
    category_set:
        Optional HierarchicalCategorySet for hierarchical accuracy and labels.
    run_id:
        Explicit run identifier stamped into the report.  When None,
        falls back to a fresh uuid — but callers that know the outer
        pipeline run_id (``pipeline.run_classification_pipeline``)
        should pass it so the evaluation report's run_id matches the
        FSM/artifact run_id.  Prevents the "evaluation_report.run_id
        does not match outer run_id" integrity issue overwatch flagged.
    """
    total = len(classifications)
    if total == 0:
        return _empty_report()

    # Split into those with curated reference
    with_ref = [
        c for c in classifications
        if c.get("reference_code") and c.get("predicted_code")
    ]
    ref_count = len(with_ref)

    # Exact accuracy
    if ref_count > 0:
        y_true = [c["reference_code"] for c in with_ref]
        y_pred = [c["predicted_code"] for c in with_ref]
        correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
        exact_accuracy = correct / ref_count
    else:
        y_true, y_pred = [], []
        exact_accuracy = 0.0

    # F1 scores
    if ref_count >= 2:
        try:
            from sklearn.metrics import f1_score
            micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
            macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        except ImportError:
            micro_f1 = exact_accuracy
            macro_f1 = exact_accuracy
    else:
        micro_f1 = exact_accuracy
        macro_f1 = exact_accuracy

    # Per-category metrics
    per_category = _per_category_metrics(y_true, y_pred, category_set)

    # Confusion matrix
    confusion = _confusion_matrix(y_true, y_pred)

    # Hierarchical accuracy
    if category_set is not None and ref_count > 0:
        hier_acc = compute_hierarchical_accuracy(y_true, y_pred, category_set)
    else:
        hier_acc = exact_accuracy

    # DST aggregate metrics (over all columns, not just those with GT)
    mean_belief = _safe_mean(c.get("belief", 0.0) for c in classifications)
    mean_plaus = _safe_mean(c.get("plausibility", 1.0) for c in classifications)
    mean_unc = _safe_mean(c.get("uncertainty", 1.0) for c in classifications)
    mean_conflict = _safe_mean(c.get("conflict", 0.0) for c in classifications)

    # Collect all evidence source names used across columns
    sources: set[str] = set()
    for c in classifications:
        es = c.get("evidence_sources", {})
        if isinstance(es, dict):
            sources.update(es.keys())

    report_run_id = run_id if run_id else str(uuid.uuid4())[:8]

    return EvaluationReport(
        run_id=report_run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_columns=total,
        columns_with_reference=ref_count,
        exact_accuracy=exact_accuracy,
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        hierarchical_accuracy=hier_acc,
        per_category=per_category,
        confusion_matrix=confusion,
        mean_belief=mean_belief,
        mean_plausibility=mean_plaus,
        mean_uncertainty_gap=mean_unc,
        mean_conflict=mean_conflict,
        evidence_sources_used=sorted(sources),
    )


def compute_hierarchical_accuracy(
    y_true: list[str],
    y_pred: list[str],
    category_set,
) -> float:
    """Correct if predicted is ancestor or descendant of truth.

    Gives partial credit for predictions in the right branch of the
    taxonomy tree — e.g. predicting "ICE.SENSITIVE.PID" when truth is
    "ICE.SENSITIVE.PID.CONTACT.EMAIL" counts as correct.
    """
    if not y_true:
        return 0.0

    correct = 0
    for true_code, pred_code in zip(y_true, y_pred):
        if true_code == pred_code:
            correct += 1
        elif pred_code in category_set.ancestors(true_code):
            correct += 1
        elif pred_code in category_set.descendants(true_code):
            correct += 1
        elif true_code in category_set.ancestors(pred_code):
            correct += 1
        elif true_code in category_set.descendants(pred_code):
            correct += 1
    return correct / len(y_true)


def epistemic_evaluation(
    classifications: list[dict[str, Any]],
    category_set=None,
) -> dict[str, Any]:
    """Evaluate using belief paths — epistemic hierarchical metrics.

    Each classification dict should have a ``belief_path`` key (list of
    dicts with code/bel/pl/depth, leaf-first). Falls back gracefully when
    belief paths are absent.

    Returns:
      per_depth: {depth: {mean_bel, mean_pl, mean_gap, count}}
      cautious_accuracy: accuracy when only committing where Bel > τ
      mean_commitment_depth: average depth of cautious code
      belief_convergence: fraction of columns where leaf Bel > 0.7
    """
    TAU = 0.7
    with_ref = [
        c for c in classifications
        if c.get("reference_code") and c.get("predicted_code")
    ]
    if not with_ref:
        return {
            "per_depth": {},
            "cautious_accuracy": 0.0,
            "mean_commitment_depth": 0.0,
            "belief_convergence": 0.0,
        }

    # Per-depth aggregation
    depth_bels: dict[int, list[float]] = {}
    depth_pls: dict[int, list[float]] = {}

    # Cautious classification
    cautious_correct = 0
    cautious_total = 0
    commitment_depths: list[int] = []
    leaf_converged = 0

    for c in with_ref:
        bp = c.get("belief_path", [])
        ref = c["reference_code"]

        if bp:
            # Leaf is first entry
            leaf_bel = bp[0].get("bel", 0.0)
            if leaf_bel >= TAU:
                leaf_converged += 1

            # Per-depth stats
            for entry in bp:
                d = entry.get("depth", 0)
                depth_bels.setdefault(d, []).append(entry.get("bel", 0.0))
                depth_pls.setdefault(d, []).append(entry.get("pl", 1.0))

            # Cautious code: deepest with Bel >= TAU
            cautious = None
            for entry in bp:  # leaf-first
                if entry.get("bel", 0.0) >= TAU:
                    cautious = entry["code"]
                    commitment_depths.append(entry.get("depth", 0))
                    break

            if cautious is not None:
                cautious_total += 1
                # Correct if cautious code is the reference or an ancestor of it
                if cautious == ref:
                    cautious_correct += 1
                elif category_set is not None:
                    ancestors = category_set.ancestors(ref)
                    if cautious in ancestors:
                        cautious_correct += 1

    # Build per-depth summary
    per_depth: dict[int, dict[str, float]] = {}
    for d in sorted(set(depth_bels.keys()) | set(depth_pls.keys())):
        bels = depth_bels.get(d, [])
        pls = depth_pls.get(d, [])
        mean_b = sum(bels) / len(bels) if bels else 0.0
        mean_p = sum(pls) / len(pls) if pls else 0.0
        per_depth[d] = {
            "mean_bel": round(mean_b, 4),
            "mean_pl": round(mean_p, 4),
            "mean_gap": round(mean_p - mean_b, 4),
            "count": len(bels),
        }

    return {
        "per_depth": per_depth,
        "cautious_accuracy": (
            round(cautious_correct / cautious_total, 4) if cautious_total > 0 else 0.0
        ),
        "mean_commitment_depth": (
            round(sum(commitment_depths) / len(commitment_depths), 2)
            if commitment_depths else 0.0
        ),
        "belief_convergence": round(leaf_converged / len(with_ref), 4),
    }


# ── Internal helpers ──────────────────────────────────────────────


def _per_category_metrics(
    y_true: list[str],
    y_pred: list[str],
    category_set=None,
) -> list[PerCategoryMetrics]:
    if not y_true:
        return []

    labels = sorted(set(y_true) | set(y_pred))

    try:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0,
        )
    except ImportError:
        # Fallback: compute per-label metrics manually
        precision, recall, f1, support = _manual_prf(y_true, y_pred, labels)

    result = []
    for i, code in enumerate(labels):
        label = ""
        if category_set is not None:
            cat = getattr(category_set, "all_by_code", {}).get(code)
            if cat is None:
                cat = category_set.by_code.get(code)
            if cat is not None:
                label = cat.label
        result.append(PerCategoryMetrics(
            code=code,
            label=label,
            precision=float(precision[i]),
            recall=float(recall[i]),
            f1=float(f1[i]),
            support=int(support[i]),
        ))
    return result


def _confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
) -> list[ConfusionEntry]:
    counts: dict[tuple[str, str], int] = {}
    for t, p in zip(y_true, y_pred):
        key = (t, p)
        counts[key] = counts.get(key, 0) + 1
    return [
        ConfusionEntry(true_code=t, predicted_code=p, count=c)
        for (t, p), c in sorted(counts.items())
    ]


def _safe_mean(values) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _manual_prf(
    y_true: list[str], y_pred: list[str], labels: list[str],
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Fallback precision/recall/f1/support without sklearn."""
    prec, rec, f1s, sup = [], [], [], []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        s = tp + fn
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1s.append(f)
        sup.append(s)
    return prec, rec, f1s, sup


def _empty_report() -> EvaluationReport:
    return EvaluationReport(
        run_id="empty",
        timestamp=datetime.now(timezone.utc).isoformat(),
        total_columns=0,
        columns_with_reference=0,
        exact_accuracy=0.0,
        micro_f1=0.0,
        macro_f1=0.0,
        hierarchical_accuracy=0.0,
        per_category=[],
        confusion_matrix=[],
        mean_belief=0.0,
        mean_plausibility=0.0,
        mean_uncertainty_gap=0.0,
        mean_conflict=0.0,
        evidence_sources_used=[],
    )
