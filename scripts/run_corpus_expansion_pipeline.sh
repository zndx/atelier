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
#   bash scripts/run_corpus_expansion_pipeline.sh \
#       --training-mode reference-primary --n-folds 5 --also-report-synth-only
#
# Training-mode flags (default: synth-only, fully back-compatible):
#   --training-mode {synth-only,reference-primary,reference+synth}
#   --n-folds N           folds for reference-primary (default 5)
#   --fold-seed N         (default 42)
#   --weights-path PATH   audit-derived weights (default
#                         build/data/agent_mediated/training_weights.json)
#   --augment-floor N     fold-train code coverage floor (default 5)
#   --also-report-synth-only  parallel synth-only diagnostic per pass
#   --skip-reference-audit    suppress Phase 0.5
#
# Logs to /tmp/corpus_expansion.log when invoked under nohup.

set -euo pipefail

# `just optimize svm` exits when BOTH gates pass:
#
#  Gate A — TARGET_ACCURACY (quantitative steering): the load-bearing
#    quality bar for the refinement loop and the agent's per-pass
#    generator authorship.  Same operator bar as the runtime ensemble's
#    production gate (validate accuracy is the upper-bound proxy for
#    test accuracy on hive-poc, since the reference is sampled from
#    hive-poc).  Refinement iterates until full-validate top-1 reaches
#    TARGET_ACCURACY OR honest plateau (2 consecutive passes < 1pp lift)
#    OR --max-passes.  Diminishing this bar would dissolve the metrology
#    machinery's purpose — DO NOT lower it below 0.95 for production
#    refinement.
#
#  Gate B — DEPLOYMENT_READY (architectural correctness): the
#    cosine-SVM mutual-affirmation check in scripts/svm_cosine_uplift_gate.py,
#    evaluating whether (cosine ⊕ SVM) materially uplifts cosine-alone
#    with bounded per-code regressions and no overturns of confident-
#    and-right cosine calls.  Gate parameters live in that script's CLI.
#
# Both required.  A high-accuracy SVM that fails Gate B is redundant or
# correlated with cosine; a Gate-B-passing SVM that hasn't reached Gate A
# is correct-but-weak.  Neither ships alone.
SYNTH_PER_CODE=40
SKIP_REFINEMENT=0
SKIP_TARGET_HEALTH=0
MAX_REFINEMENT_PASSES=5
TARGET_ACCURACY=0.95
CORPUS_DIR="build/data/svm_training/corpus_v2"

# Training-protocol args (default synth-only preserves back-compat).
# Under reference-primary, Phase 0.5 runs the reference-quality audit
# and Phase D + refinement loop train against the reference k-fold with
# synth augmentation.  See docs/notes/2026-05-25/phase_gate_brief.md
# (Training protocol section) for the protocol rationale.
TRAINING_MODE="synth-only"
N_FOLDS=5
FOLD_SEED=42
WEIGHTS_PATH="build/data/agent_mediated/training_weights.json"
AUGMENT_FLOOR=5
ALSO_REPORT_SYNTH_ONLY=0
SKIP_REFERENCE_AUDIT=0

while [[ $# -gt 0 ]]; do
  case $1 in
    --synth-examples-per-code) SYNTH_PER_CODE="$2"; shift 2 ;;
    --rows-per-node) SYNTH_PER_CODE="$2"; shift 2 ;;  # deprecated alias
    --skip-refinement) SKIP_REFINEMENT=1; shift ;;
    --skip-target-health) SKIP_TARGET_HEALTH=1; shift ;;
    --max-passes) MAX_REFINEMENT_PASSES="$2"; shift 2 ;;
    --target-accuracy) TARGET_ACCURACY="$2"; shift 2 ;;
    --corpus-dir) CORPUS_DIR="$2"; shift 2 ;;
    --training-mode) TRAINING_MODE="$2"; shift 2 ;;
    --n-folds) N_FOLDS="$2"; shift 2 ;;
    --fold-seed) FOLD_SEED="$2"; shift 2 ;;
    --weights-path) WEIGHTS_PATH="$2"; shift 2 ;;
    --augment-floor) AUGMENT_FLOOR="$2"; shift 2 ;;
    --also-report-synth-only) ALSO_REPORT_SYNTH_ONLY=1; shift ;;
    --skip-reference-audit) SKIP_REFERENCE_AUDIT=1; shift ;;
    -h|--help)
      grep "^#" "$0" | head -45
      exit 0 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

# Validate training-mode early.
case "$TRAINING_MODE" in
  synth-only|reference-primary|reference+synth) ;;
  *)
    echo "ERROR: unknown --training-mode: $TRAINING_MODE"
    echo "Valid: synth-only, reference-primary, reference+synth"
    exit 1 ;;
esac

# Build the Phase D arg array once so every Phase D invocation is identical.
# Empty under synth-only (no flags → Phase D defaults match historical behavior).
PHASE_D_ARGS=()
if [[ "$TRAINING_MODE" != "synth-only" ]]; then
  PHASE_D_ARGS+=("--training-mode" "$TRAINING_MODE")
  PHASE_D_ARGS+=("--n-folds" "$N_FOLDS")
  PHASE_D_ARGS+=("--fold-seed" "$FOLD_SEED")
  PHASE_D_ARGS+=("--weights-path" "$WEIGHTS_PATH")
  PHASE_D_ARGS+=("--augment-floor" "$AUGMENT_FLOOR")
  [[ "$ALSO_REPORT_SYNTH_ONLY" -eq 1 ]] && PHASE_D_ARGS+=("--also-report-synth-only")
fi

cd /home/cdsw

SNAPSHOTS_DIR="build/svm_corpus_v2/corpus_snapshots"
mkdir -p "$SNAPSHOTS_DIR"

echo "=== run_corpus_expansion_pipeline configuration ==="
echo "  training_mode:        $TRAINING_MODE"
if [[ "$TRAINING_MODE" != "synth-only" ]]; then
  echo "  n_folds:              $N_FOLDS"
  echo "  fold_seed:            $FOLD_SEED"
  echo "  weights_path:         $WEIGHTS_PATH"
  echo "  augment_floor:        $AUGMENT_FLOOR"
  echo "  also_report_synth:    $ALSO_REPORT_SYNTH_ONLY"
fi
echo "  corpus_dir:           $CORPUS_DIR"
echo "  target_accuracy:      $TARGET_ACCURACY"
echo "  max_refinement_passes: $MAX_REFINEMENT_PASSES"
echo "  skip_refinement:      $SKIP_REFINEMENT"
echo "  skip_target_health:   $SKIP_TARGET_HEALTH"
echo

echo "=== $(date -u +%FT%TZ) Phase A: generator coverage audit ==="
python scripts/audit_generator_coverage.py

if [[ "$TRAINING_MODE" != "synth-only" && "$SKIP_REFERENCE_AUDIT" -ne 1 ]]; then
  echo
  echo "=== $(date -u +%FT%TZ) Phase 0.5: reference-quality audit ==="
  # Produces $WEIGHTS_PATH (per-row weights + per-code coverage flags)
  # consumed by Phase D under reference-primary and reference+synth.
  python scripts/audit_reference_quality.py \
    --output "$WEIGHTS_PATH"
fi

echo
echo "=== $(date -u +%FT%TZ) Phase B: Agent SDK generator authorship ==="
# Loop the agent until no more gaps remain (each session targets ~30 codes).
# Cap at 10 invocations to bound runaway.
for i in $(seq 1 10); do
  # Check remaining gaps BEFORE invoking the agent.  Treats a code as
  # treated only if it appears in generators_v1.py's GENERATORS_BY_CODE
  # (acceptance-aware; not just proposal-file-presence-aware).  Skips
  # the agent invocation entirely when nothing is gap — saves 15-20 min
  # of wall-clock per spurious session that would otherwise be spent
  # in inspect-and-skip.
  remaining=$(python -c "
import json, sys
audit = json.load(open('build/svm_corpus_v2/coverage_audit.json'))
sys.path.insert(0, 'build/lib/generated')
try:
    import generators_v1
    accepted = set(generators_v1.GENERATORS_BY_CODE.keys())
except Exception:
    accepted = set()
remaining = [g['code'] for g in audit['gap_list'] if g['code'] not in accepted]
print(len(remaining))
")
  if [[ "$remaining" -eq 0 ]]; then
    echo "All gaps already treated; skipping Phase B loop."
    break
  fi
  echo "--- Phase B agent session $i (remaining=$remaining) ---"
  python scripts/run_evolve_generators_sdk.py
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
echo "=== $(date -u +%FT%TZ) Phase D: initial eval (pass 0, mode=$TRAINING_MODE) ==="
python scripts/reflect_nhsvm_eval_shap_v2.py \
  --pass-idx 0 \
  --corpus-dir "$CORPUS_DIR" \
  "${PHASE_D_ARGS[@]}"

if [[ "$SKIP_REFINEMENT" -eq 1 ]]; then
  echo "=== Skipping Phase E (--skip-refinement) ==="
else
  echo
  echo "=== $(date -u +%FT%TZ) Phase E: iterative refinement loop ==="
  # The refinement loop runs Phase D internally with --pass-idx for each
  # refinement pass and tracks best_pass.json + corpus snapshots.  Under
  # non-synth-only modes we forward the training-protocol args so every
  # nested Phase D invocation gets routed to the same dispatcher branch.
  REFINE_EXTRA_ARGS=()
  if [[ "$TRAINING_MODE" != "synth-only" ]]; then
    REFINE_EXTRA_ARGS+=("--training-mode" "$TRAINING_MODE")
    REFINE_EXTRA_ARGS+=("--n-folds" "$N_FOLDS")
    REFINE_EXTRA_ARGS+=("--fold-seed" "$FOLD_SEED")
    REFINE_EXTRA_ARGS+=("--weights-path" "$WEIGHTS_PATH")
    REFINE_EXTRA_ARGS+=("--augment-floor" "$AUGMENT_FLOOR")
    [[ "$ALSO_REPORT_SYNTH_ONLY" -eq 1 ]] && REFINE_EXTRA_ARGS+=("--also-report-synth-only")
  fi
  python scripts/refine_generators_from_failures.py \
    --max-passes "$MAX_REFINEMENT_PASSES" \
    --target-accuracy "$TARGET_ACCURACY" \
    --synth-examples-per-code "$SYNTH_PER_CODE" \
    --corpus-dir "$CORPUS_DIR" \
    "${REFINE_EXTRA_ARGS[@]}"
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
      echo "=== $(date -u +%FT%TZ) Phase D: final eval on restored best-pass corpus (mode=$TRAINING_MODE) ==="
      python scripts/reflect_nhsvm_eval_shap_v2.py \
        --pass-idx "$BEST_IDX" \
        --corpus-dir "$CORPUS_DIR" \
        "${PHASE_D_ARGS[@]}"
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
echo "=== $(date -u +%FT%TZ) Gate B: cosine-SVM mutual affirmation ==="
# Gate B (architectural correctness).  Tests whether the trained SVM is
# a verified mutually-affirming addition to the cosine channel that
# `just optimize cosine` established.  Necessary but not sufficient —
# Gate A (refinement loop reaching TARGET_ACCURACY OR plateau) is the
# separate quality bar.  Non-zero exit = Gate B failed; the report at
# build/svm_uplift_gate/uplift_report.md tells the operator WHICH check
# failed and WHICH codes regressed.
# Build Gate B args to mirror Phase D's training-mode (Gate B's head must
# match the deployment head Phase D measures, otherwise the gate tests a
# model that wouldn't ship).
GATE_B_ARGS=("--corpus-dir" "$CORPUS_DIR")
if [[ "$TRAINING_MODE" != "synth-only" ]]; then
  GATE_B_ARGS+=("--training-mode" "$TRAINING_MODE")
  GATE_B_ARGS+=("--weights-path" "$WEIGHTS_PATH")
  GATE_B_ARGS+=("--augment-floor" "$AUGMENT_FLOOR")
fi
if ! python scripts/svm_cosine_uplift_gate.py "${GATE_B_ARGS[@]}"; then
  echo
  echo "✗ SVM channel did NOT pass Gate B (architectural correctness)."
  echo "  See build/svm_uplift_gate/uplift_report.md for which check"
  echo "  failed and which codes regressed.  Run another corpus"
  echo "  refinement cycle (driving TARGET_ACCURACY higher OR resolving"
  echo "  per-code regressions) before attempting integration."
  GATE_RC=1
else
  echo "✓ Gate B passed (architectural correctness verified)."
  echo "  Gate A status: check refinement_history.json — full-validate"
  echo "  top-1 must also have reached TARGET_ACCURACY (or honest"
  echo "  plateau) for the SVM channel to be fully shippable."
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
