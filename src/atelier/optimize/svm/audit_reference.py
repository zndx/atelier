"""Reference-quality audit for reference-primary SVM training.

Reads the agent-mediated reference's audit.json (per-row confidence,
needs_operator_review flags, source agreement) and produces a
``training_weights.json`` file that downstream training paths
(``atelier.optimize.svm.train`` reference-primary mode, Gate B
reference-primary mode) consume for per-row weighting and per-code
coverage decisions.

Policy (defaults):
    - confidence high  → weight 1.0
    - confidence medium → weight 0.5
    - confidence low    → weight 0.0  (excluded from training)
    - confidence unsure → weight 0.0  (excluded from training)
    - needs_operator_review=true  → multiply weight by 0.5
    - per-code coverage_min = 3   → codes with < 3 kept rows flagged
                                     for synth augmentation

Idempotent.  No existing consumer breaks if this script never runs —
downstream code paths default to "all weight 1.0, no augmentation
flagging" when training_weights.json is absent.

CLI entry: ``scripts/audit_reference_quality.py`` is the thin shim
that calls ``run_audit()`` here.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


CONFIDENCE_DEFAULT_WEIGHTS: dict[str, float] = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.0,
    "unsure": 0.0,
}


def compute_weights(
    audit: dict[str, dict],
    high_weight: float,
    medium_weight: float,
    low_weight: float,
    needs_review_multiplier: float,
) -> dict[str, dict]:
    """Apply the policy to every row in the audit; return per-row records."""
    weight_table = {
        "high": high_weight,
        "medium": medium_weight,
        "low": low_weight,
        "unsure": low_weight,  # treat unsure as low
    }
    per_row: dict[str, dict] = {}
    for key, entry in audit.items():
        if not isinstance(entry, dict):
            continue
        confidence = entry.get("confidence", "unsure")
        base_weight = weight_table.get(confidence, 0.0)
        needs_review = bool(entry.get("needs_operator_review", False))
        weight = base_weight * (needs_review_multiplier if needs_review else 1.0)
        per_row[key] = {
            "weight": weight,
            "confidence": confidence,
            "needs_operator_review": needs_review,
            "code": entry.get("code"),
            "tag": entry.get("tag"),
            "agreement": entry.get("agreement"),
            "exclude": weight <= 0.0,
        }
    return per_row


def compute_coverage(
    per_row: dict[str, dict],
    coverage_min: int,
) -> dict[str, dict]:
    """Group by code and produce per-code coverage statistics."""
    by_code: dict[str, list[dict]] = defaultdict(list)
    for entry in per_row.values():
        code = entry.get("code")
        if code is None:
            continue
        by_code[code].append(entry)

    coverage: dict[str, dict] = {}
    for code, entries in by_code.items():
        n_total = len(entries)
        n_high = sum(1 for e in entries if e["confidence"] == "high")
        n_medium = sum(1 for e in entries if e["confidence"] == "medium")
        n_low = sum(1 for e in entries if e["confidence"] in ("low", "unsure"))
        n_excluded = sum(1 for e in entries if e["exclude"])
        n_kept = n_total - n_excluded
        effective_weight_sum = sum(e["weight"] for e in entries)
        coverage[code] = {
            "n_total": n_total,
            "n_high": n_high,
            "n_medium": n_medium,
            "n_low": n_low,
            "n_excluded": n_excluded,
            "n_kept": n_kept,
            "effective_weight_sum": round(effective_weight_sum, 4),
            "needs_synth_augmentation": n_kept < coverage_min,
        }
    return coverage


def build_summary(
    per_row: dict[str, dict],
    coverage: dict[str, dict],
    coverage_min: int,
) -> dict:
    n_high = sum(1 for e in per_row.values() if e["confidence"] == "high")
    n_medium = sum(1 for e in per_row.values() if e["confidence"] == "medium")
    n_low = sum(1 for e in per_row.values() if e["confidence"] in ("low", "unsure"))
    n_needs_review = sum(
        1 for e in per_row.values() if e["needs_operator_review"]
    )
    n_excluded = sum(1 for e in per_row.values() if e["exclude"])
    codes_under = sorted(
        code for code, c in coverage.items() if c["needs_synth_augmentation"]
    )
    return {
        "n_rows_total": len(per_row),
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "n_needs_review": n_needs_review,
        "n_excluded": n_excluded,
        "n_codes": len(coverage),
        "codes_under_coverage_min": codes_under,
        "n_codes_under_coverage_min": len(codes_under),
        "coverage_min": coverage_min,
    }


def run_audit(
    *,
    audit_path: Path,
    output_path: Path,
    high_weight: float = 1.0,
    medium_weight: float = 0.5,
    low_weight: float = 0.0,
    needs_review_multiplier: float = 0.5,
    coverage_min: int = 3,
) -> dict:
    """End-to-end audit: read audit.json, apply policy, write training_weights.json.

    Returns the assembled output dict (also written to disk).  Raises
    ``FileNotFoundError`` if ``audit_path`` is missing.
    """
    if not audit_path.exists():
        raise FileNotFoundError(f"audit file not found: {audit_path}")

    audit = json.loads(audit_path.read_text())
    if not isinstance(audit, dict):
        raise TypeError(
            f"audit.json must be a dict at top level, got {type(audit).__name__}"
        )

    per_row = compute_weights(
        audit,
        high_weight=high_weight,
        medium_weight=medium_weight,
        low_weight=low_weight,
        needs_review_multiplier=needs_review_multiplier,
    )
    coverage = compute_coverage(per_row, coverage_min=coverage_min)
    summary = build_summary(per_row, coverage, coverage_min=coverage_min)

    output_obj = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "audit_source": str(audit_path),
        "policy": {
            "high_weight": high_weight,
            "medium_weight": medium_weight,
            "low_weight": low_weight,
            "needs_review_multiplier": needs_review_multiplier,
            "coverage_min": coverage_min,
        },
        "summary": summary,
        "per_code_coverage": coverage,
        "per_row": per_row,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_obj, indent=2))
    return output_obj


def main() -> int:
    """CLI entry point.  Called from ``scripts/audit_reference_quality.py``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--audit",
        type=Path,
        default=Path("build/data/agent_mediated/audit.json"),
        help="Path to the agent-mediated audit.json input",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("build/data/agent_mediated/training_weights.json"),
        help="Path to write the resulting training_weights.json",
    )
    ap.add_argument("--high-weight", type=float, default=1.0)
    ap.add_argument("--medium-weight", type=float, default=0.5)
    ap.add_argument("--low-weight", type=float, default=0.0)
    ap.add_argument("--needs-review-multiplier", type=float, default=0.5)
    ap.add_argument("--coverage-min", type=int, default=3)
    args = ap.parse_args()

    try:
        output_obj = run_audit(
            audit_path=args.audit,
            output_path=args.output,
            high_weight=args.high_weight,
            medium_weight=args.medium_weight,
            low_weight=args.low_weight,
            needs_review_multiplier=args.needs_review_multiplier,
            coverage_min=args.coverage_min,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    except TypeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    summary = output_obj["summary"]
    print(
        f"=== reference quality audit ===\n"
        f"  source:         {args.audit}\n"
        f"  output:         {args.output}\n"
        f"  n_rows_total:   {summary['n_rows_total']}\n"
        f"  n_high:         {summary['n_high']}\n"
        f"  n_medium:       {summary['n_medium']}\n"
        f"  n_low/unsure:   {summary['n_low']}\n"
        f"  n_needs_review: {summary['n_needs_review']}\n"
        f"  n_excluded:     {summary['n_excluded']}\n"
        f"  n_codes:        {summary['n_codes']}\n"
        f"  codes under coverage_min={args.coverage_min}: "
        f"{summary['n_codes_under_coverage_min']}"
    )
    return 0
