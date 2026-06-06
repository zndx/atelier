#!/usr/bin/env python3
"""Validate subsumption alignment against the legacy LLM-mediated method.

Operator-invocable entry point for the hybrid A/B comparison.
Produces a validation report at ``build/cache/alignment_validation/``.

Usage:
    uv run python scripts/validate_alignment.py
    uv run python scripts/validate_alignment.py --taxonomy-id default --force
    uv run python scripts/validate_alignment.py --score-threshold 0.40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate subsumption alignment A/B against legacy LLM method",
    )
    parser.add_argument(
        "--taxonomy-id", default="default",
        help="Taxonomy ID to validate alignment for (default: 'default')",
    )
    parser.add_argument(
        "--score-threshold", type=float, default=0.35,
        help="Cosine similarity threshold for subsumption alignment",
    )
    parser.add_argument(
        "--embedding-model", default="all-MiniLM-L6-v2",
        help="Embedding model identifier",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-validation even if a report already exists",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("build/cache/alignment_validation"),
        help="Output directory for validation reports",
    )
    parser.add_argument(
        "--with-llm", action="store_true",
        help="Include legacy LLM comparison (requires API access)",
    )
    args = parser.parse_args()

    # Load vocabulary
    try:
        from atelier.classify.taxonomy import load_universal_vocabulary
        cats = load_universal_vocabulary(hierarchical=True)
    except Exception as exc:
        logger.error("Failed to load vocabulary: %s", exc)
        return 1

    # Optionally set up LLM backend for the legacy comparison
    llm_backend = None
    system_prompt = None
    model_name = None

    if args.with_llm:
        try:
            from atelier.config import load_config
            from atelier.config_overlay import apply_to_config
            from atelier.classify.llm_backend import LLMBackend
            from atelier.classify.prompts import build_system_prompt

            cfg = apply_to_config(load_config())
            llm_backend = LLMBackend(cfg)
            system_prompt = build_system_prompt(cats, cfg)
            model_name = getattr(cfg, "classify_subagent_model", None)
            logger.info("LLM backend configured for legacy comparison")
        except Exception as exc:
            logger.warning(
                "Could not set up LLM backend — legacy comparison will be "
                "empty: %s", exc,
            )

    # Run the validation
    from atelier.classify.alignment_validator import run_ab_validation

    report = run_ab_validation(
        cats,
        taxonomy_id=args.taxonomy_id,
        embedding_model=args.embedding_model,
        score_threshold=args.score_threshold,
        llm_backend=llm_backend,
        system_prompt=system_prompt,
        model_name=model_name,
        cache_dir=args.output_dir,
        force=args.force,
    )

    if report is None:
        logger.info(
            "Validation already exists for this combination. "
            "Use --force to re-run."
        )
        return 0

    # Print summary
    print("\n" + "=" * 60)
    print("ALIGNMENT VALIDATION REPORT")
    print("=" * 60)
    print(f"  Vocab hash:       {report.vocab_hash}")
    print(f"  New method:       {report.method_new}")
    print(f"  Legacy method:    {report.method_legacy}")
    print(f"  Timestamp:        {report.timestamp}")
    print(f"  Agreement rate:   {report.agreement_rate:.1%}")
    print(f"  New mapped:       {len(report.alignment_new)}")
    print(f"  Legacy mapped:    {len(report.alignment_legacy)}")
    print(f"  New-only:         {report.new_only_count}")
    print(f"  Legacy-only:      {report.legacy_only_count}")
    print(f"  Disagreements:    {len(report.disagreement_details)}")
    print("=" * 60)

    if report.disagreement_details:
        print("\nTop disagreements (first 10):")
        for d in report.disagreement_details[:10]:
            print(f"  {d['ice_code']}: new={d['new']} vs legacy={d['legacy']}")

    if report.accuracy_delta is not None:
        print(f"\n  Accuracy delta: {report.accuracy_delta:+.3f}")
        print(f"  New accuracy:   {report.accuracy_new:.3f}")
        print(f"  Legacy accuracy: {report.accuracy_legacy:.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
