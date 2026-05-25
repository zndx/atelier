#!/usr/bin/env bash
# scripts/run_corpus_expansion_pipeline.sh — long-run orchestrator.
#
# Runs the full SHAP-priority-guided synthetic corpus expansion plan
# (Phases A-E) end-to-end.  Designed for unattended execution on the
# CAI App pod where the Bedrock-routed Agent SDK runs.
#
# Each phase logs to its own report dir under build/.  Re-running this
# script after a partial completion resumes from where it stopped (each
# phase is independently resume-safe).
#
# Usage:
#   bash scripts/run_corpus_expansion_pipeline.sh
#   bash scripts/run_corpus_expansion_pipeline.sh --rows-per-node 100  # smaller smoke run
#   bash scripts/run_corpus_expansion_pipeline.sh --skip-refinement     # initial pass only
#
# Logs to /tmp/corpus_expansion.log when invoked under nohup.

set -euo pipefail

ROWS_PER_NODE=200
SKIP_REFINEMENT=0
MAX_REFINEMENT_PASSES=5
TARGET_ACCURACY=0.80

while [[ $# -gt 0 ]]; do
  case $1 in
    --rows-per-node) ROWS_PER_NODE="$2"; shift 2 ;;
    --skip-refinement) SKIP_REFINEMENT=1; shift ;;
    --max-passes) MAX_REFINEMENT_PASSES="$2"; shift 2 ;;
    --target-accuracy) TARGET_ACCURACY="$2"; shift 2 ;;
    -h|--help)
      grep "^#" "$0" | head -25
      exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

cd /home/cdsw

echo "=== $(date -u +%FT%TZ) Phase A: generator coverage audit ==="
python scripts/audit_generator_coverage.py

echo
echo "=== $(date -u +%FT%TZ) Phase B: Agent SDK generator authorship ==="
# Loop the agent until no more gaps remain (each session targets ~30 codes).
# Cap at 10 invocations to bound runaway.
for i in $(seq 1 10); do
  echo "--- Phase B agent session $i ---"
  python scripts/run_evolve_generators_sdk.py
  # Check if there are still untreated gap codes
  remaining=$(python -c "
import json
audit = json.load(open('build/svm_corpus_v2/coverage_audit.json'))
import os
proposals_dir = 'build/svm_corpus_v2/proposals'
treated = {f.removesuffix('.json').replace('_', '.') for f in os.listdir(proposals_dir)} if os.path.isdir(proposals_dir) else set()
remaining = [g['code'] for g in audit['gap_list'] if g['code'] not in treated]
print(len(remaining))
")
  echo "Remaining gap codes: $remaining"
  if [[ "$remaining" -eq 0 ]]; then
    echo "All gaps treated; exiting Phase B loop."
    break
  fi
done

echo
echo "=== $(date -u +%FT%TZ) Phase C: corpus generation at scale ==="
python scripts/generate_corpus_v2.py --rows-per-node "$ROWS_PER_NODE"

echo
echo "=== $(date -u +%FT%TZ) Phase D: long-run eval (synth+real → real held-out) ==="
python scripts/reflect_nhsvm_eval_shap_v2.py

if [[ "$SKIP_REFINEMENT" -eq 1 ]]; then
  echo "=== Skipping Phase E (--skip-refinement) ==="
else
  echo
  echo "=== $(date -u +%FT%TZ) Phase E: iterative refinement loop ==="
  python scripts/refine_generators_from_failures.py \
    --max-passes "$MAX_REFINEMENT_PASSES" \
    --target-accuracy "$TARGET_ACCURACY" \
    --rows-per-node "$ROWS_PER_NODE"
fi

echo
echo "=== $(date -u +%FT%TZ) Pipeline complete ==="
echo
echo "Reports:"
echo "  coverage audit:       build/svm_corpus_v2/coverage_audit.json"
echo "  acceptance log:       build/svm_corpus_v2/acceptance_log.json"
echo "  corpus manifest:      build/data/svm_training/corpus_v2/manifest.json"
echo "  eval report:          build/reflect_nhsvm_eval_shap_v2/report.md"
echo "  per-category JSON:    build/reflect_nhsvm_eval_shap_v2/per_category_accuracy.json"
echo "  refinement history:   build/svm_corpus_v2/refinement_history.json"
