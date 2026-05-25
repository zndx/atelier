#!/usr/bin/env bash
# scripts/run_corpus_expansion_pipeline.sh — long-run orchestrator.
#
# Runs the full SHAP-priority-guided synthetic corpus expansion plan
# (Phases A-E) end-to-end + the target-data-source health check (test
# stage).  Designed for unattended execution on the CAI App pod where
# the Bedrock-routed Agent SDK runs.
#
# Each phase logs to its own report dir under build/.  Re-running this
# script after a partial completion resumes from where it stopped (each
# phase is independently resume-safe).
#
# Best-pass tracking: refinement loop writes build/svm_corpus_v2/best_pass.json
# with the best validate-accuracy pass's idx + corpus hash.  Before the
# final test stage, this orchestrator restores the best-pass corpus
# snapshot if it differs from the current pass's corpus (so test runs
# against the BEST model the loop produced, not the LAST).
#
# Usage:
#   bash scripts/run_corpus_expansion_pipeline.sh
#   bash scripts/run_corpus_expansion_pipeline.sh --synth-examples-per-code 60
#   bash scripts/run_corpus_expansion_pipeline.sh --skip-refinement
#   bash scripts/run_corpus_expansion_pipeline.sh --skip-target-health
#   bash scripts/run_corpus_expansion_pipeline.sh --corpus-dir build/data/svm_training/corpus_v3
#
# Logs to /tmp/corpus_expansion.log when invoked under nohup.

set -euo pipefail

# Two distinct thresholds:
#  - TARGET_ACCURACY is the refinement-loop's stop signal: when full-
#    validate top-1 reaches this, the loop stops iterating (still subject
#    to plateau detection).  It is NOT the deployment gate.
#  - The deployment gate is the cosine-SVM mutual-affirmation check in
#    scripts/svm_cosine_uplift_gate.py, evaluating whether (cosine ⊕ SVM)
#    materially uplifts cosine-alone with bounded per-code regressions and
#    no overturns of confident-and-right cosine calls.  Gate parameters
#    live in that script's CLI.
SYNTH_PER_CODE=40
SKIP_REFINEMENT=0
SKIP_TARGET_HEALTH=0
MAX_REFINEMENT_PASSES=5
TARGET_ACCURACY=0.95
CORPUS_DIR="build/data/svm_training/corpus_v2"

while [[ $# -gt 0 ]]; do
  case $1 in
    --synth-examples-per-code) SYNTH_PER_CODE="$2"; shift 2 ;;
    --rows-per-node) SYNTH_PER_CODE="$2"; shift 2 ;;  # deprecated alias
    --skip-refinement) SKIP_REFINEMENT=1; shift ;;
    --skip-target-health) SKIP_TARGET_HEALTH=1; shift ;;
    --max-passes) MAX_REFINEMENT_PASSES="$2"; shift 2 ;;
    --target-accuracy) TARGET_ACCURACY="$2"; shift 2 ;;
    --corpus-dir) CORPUS_DIR="$2"; shift 2 ;;
    -h|--help)
      grep "^#" "$0" | head -30
      exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

cd /home/cdsw

SNAPSHOTS_DIR="build/svm_corpus_v2/corpus_snapshots"
mkdir -p "$SNAPSHOTS_DIR"

echo "=== $(date -u +%FT%TZ) Phase A: generator coverage audit ==="
python scripts/audit_generator_coverage.py

echo
echo "=== $(date -u +%FT%TZ) Phase B: Agent SDK generator authorship ==="
# Loop the agent until no more gaps remain (each session targets ~30 codes).
# Cap at 10 invocations to bound runaway.
for i in $(seq 1 10); do
  echo "--- Phase B agent session $i ---"
  python scripts/run_evolve_generators_sdk.py
  remaining=$(python -c "
import json, os
audit = json.load(open('build/svm_corpus_v2/coverage_audit.json'))
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
python scripts/generate_corpus_v2.py \
  --synth-examples-per-code "$SYNTH_PER_CODE" \
  --output "$CORPUS_DIR"

# Snapshot pass-0 corpus (initial pre-refinement state)
PASS0_SNAP="$SNAPSHOTS_DIR/pass_0"
mkdir -p "$PASS0_SNAP"
cp "$CORPUS_DIR/synth_rows.jsonl" "$PASS0_SNAP/"
cp "$CORPUS_DIR/manifest.json" "$PASS0_SNAP/"
echo "Snapshot taken: $PASS0_SNAP"

echo
echo "=== $(date -u +%FT%TZ) Phase D: initial eval (pass 0) ==="
python scripts/reflect_nhsvm_eval_shap_v2.py \
  --pass-idx 0 \
  --corpus-dir "$CORPUS_DIR"

if [[ "$SKIP_REFINEMENT" -eq 1 ]]; then
  echo "=== Skipping Phase E (--skip-refinement) ==="
else
  echo
  echo "=== $(date -u +%FT%TZ) Phase E: iterative refinement loop ==="
  # The refinement loop runs Phase D internally with --pass-idx for each
  # refinement pass and tracks best_pass.json + corpus snapshots.
  python scripts/refine_generators_from_failures.py \
    --max-passes "$MAX_REFINEMENT_PASSES" \
    --target-accuracy "$TARGET_ACCURACY" \
    --synth-examples-per-code "$SYNTH_PER_CODE" \
    --corpus-dir "$CORPUS_DIR"
fi

# Best-pass restoration before the test stage
BEST_PASS_JSON="build/svm_corpus_v2/best_pass.json"
if [[ -f "$BEST_PASS_JSON" ]]; then
  BEST_IDX=$(python -c "import json; print(json.load(open('$BEST_PASS_JSON'))['best_pass_idx'])")
  BEST_TOP1=$(python -c "import json; print(json.load(open('$BEST_PASS_JSON'))['best_top1'])")
  BEST_SNAP="$SNAPSHOTS_DIR/pass_${BEST_IDX}"
  if [[ -d "$BEST_SNAP" && -f "$BEST_SNAP/synth_rows.jsonl" ]]; then
    CURRENT_HASH=$(python -c "
import hashlib
print(hashlib.sha1(open('$CORPUS_DIR/synth_rows.jsonl','rb').read()).hexdigest()[:12])
")
    BEST_HASH=$(python -c "
import hashlib
print(hashlib.sha1(open('$BEST_SNAP/synth_rows.jsonl','rb').read()).hexdigest()[:12])
")
    if [[ "$CURRENT_HASH" != "$BEST_HASH" ]]; then
      echo
      echo "=== Restoring best-pass corpus (pass $BEST_IDX, top-1=$BEST_TOP1) ==="
      echo "    current corpus hash:  $CURRENT_HASH"
      echo "    best-pass corpus hash: $BEST_HASH"
      cp "$BEST_SNAP/synth_rows.jsonl" "$CORPUS_DIR/"
      cp "$BEST_SNAP/manifest.json" "$CORPUS_DIR/"
      # Re-run Phase D against restored corpus for the canonical final number
      echo
      echo "=== $(date -u +%FT%TZ) Phase D: final eval on restored best-pass corpus ==="
      python scripts/reflect_nhsvm_eval_shap_v2.py \
        --pass-idx "$BEST_IDX" \
        --corpus-dir "$CORPUS_DIR"
    else
      echo "Best-pass corpus already current ($CURRENT_HASH); no restore needed."
    fi
  else
    echo "WARN: best_pass.json points to pass $BEST_IDX but snapshot $BEST_SNAP missing"
  fi
else
  echo "No best_pass.json present — assuming initial Phase D pass 0 is canonical."
fi

echo
echo "=== $(date -u +%FT%TZ) DEPLOYMENT GATE: cosine-SVM mutual affirmation ==="
# Load-bearing gate.  Tests whether the trained SVM is a verified
# mutually-affirming addition to the cosine channel that
# `just optimize cosine` established.  Non-zero exit = NOT deployment-
# ready; the report at build/svm_uplift_gate/uplift_report.md tells
# the operator WHICH check failed.
if ! python scripts/svm_cosine_uplift_gate.py --corpus-dir "$CORPUS_DIR"; then
  echo
  echo "✗ SVM channel is NOT deployment-ready (uplift gate failed)."
  echo "  See build/svm_uplift_gate/uplift_report.md for which check"
  echo "  failed and which codes regressed.  Run another corpus"
  echo "  refinement cycle before attempting integration."
  GATE_RC=1
else
  echo "✓ DEPLOYMENT_READY — uplift gate passed."
  GATE_RC=0
fi

if [[ "$SKIP_TARGET_HEALTH" -eq 1 ]]; then
  echo "=== Skipping post-integration target-data-source diagnostic ==="
else
  echo
  echo "=== $(date -u +%FT%TZ) Post-integration diagnostic: hive-poc target health ==="
  # Informational only — does NOT gate.  Surfaces concerns the deployment
  # gate didn't catch (e.g., catch-all collapse on broader unlabeled data).
  python scripts/svm_target_health.py --corpus-dir "$CORPUS_DIR" || true
fi

# Exit with the deployment-gate status, not the diagnostic's status.
[ "$GATE_RC" -ne 0 ] && exit "$GATE_RC"

echo
echo "=== $(date -u +%FT%TZ) Pipeline complete ==="
echo
echo "Reports:"
echo "  coverage audit:       build/svm_corpus_v2/coverage_audit.json"
echo "  acceptance log:       build/svm_corpus_v2/acceptance_log.json"
echo "  metrology report:     build/svm_corpus_v2/metrology_report.json"
echo "  corpus manifest:      $CORPUS_DIR/manifest.json"
echo "  eval report:          build/reflect_nhsvm_eval_shap_v2/report.md"
echo "  per-pass results:     build/reflect_nhsvm_eval_shap_v2/results_pass*.json"
echo "  refinement history:   build/svm_corpus_v2/refinement_history.json"
echo "  best-pass record:     build/svm_corpus_v2/best_pass.json"
echo "  abandoned codes:      build/svm_corpus_v2/abandoned_codes.json"
echo "  uplift gate report:   build/svm_uplift_gate/uplift_report.md  ← the gate"
echo "  target health diag:   build/reflect_nhsvm_eval_shap_v2/target_health_report.json"
