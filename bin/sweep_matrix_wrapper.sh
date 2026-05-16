#!/bin/bash
# Wrapper for the N-axis matrix sweep (scripts/sweep_matrix.py).
#
# Differs from build/sweep_wrapper.sh (single-axis bel_threshold
# sweep) in two ways:
#
#   1. Sets ATELIER_GROUND_TRUTH_PATH so the per-iteration scorer
#      writes scoring_trend.json / scoring_errors_iter{N}.tsv /
#      scoring_summary.md under each run's build/results/{run_id}/.
#      Production runs leave this unset and skip scoring with one
#      log line.
#
#   2. Invokes sweep_matrix.py with --axis specs instead of the
#      single --thresholds list, so it supports the 2D / 3D grids
#      we agreed on for the weekend's harm-first sweep.
#
# Usage (override the axes from the command line):
#
#   build/sweep_matrix_wrapper.sh \
#       --axis classify.cautious_review.bel_threshold=0.55,0.65,0.75,0.80 \
#       --axis classify.bootstrap.gap_threshold=0.12,0.18,0.24 \
#       --name bel_x_gap
#
# Default invocation (no extra args) runs the agreed 2D
# bel_threshold × gap_threshold grid.

cd /home/cdsw

# Materialized config -> shell-safe export (same as the single-axis
# wrapper; values with unquoted spaces would break direct `source`).
python3 << 'PY' > /tmp/atelier_env_export.sh
with open('build/config/atelier.env') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, _, v = line.partition('=')
        if not (k and (k[0].isalpha() or k[0] == '_') and all(c.isalnum() or c == '_' for c in k)):
            continue
        v_quoted = "'" + v.replace("'", "'\\''") + "'"
        print(f"export {k}={v_quoted}")
PY
# shellcheck disable=SC1091
source /tmp/atelier_env_export.sh

# ── Sweep-specific overrides ────────────────────────────────────────

# Engage both A10Gs at our corpus size (920 cols × 64 tables).
export ATELIER_GPU_SHARD_THRESHOLD=32000

# Force-set the LLM model.  The deployment env exports
# ATELIER_CLASSIFY_MODEL="" (empty), and HOCON later-wins semantics
# means empty wins over ANTHROPIC_SUBAGENT_MODEL.  Mirror the
# materialized value explicitly so both env vars hold the Bedrock
# Sonnet 4.5 ARN.
export ATELIER_CLASSIFY_MODEL="$ANTHROPIC_SUBAGENT_MODEL"
[ -z "$ATELIER_LLM_MODEL" ] && export ATELIER_LLM_MODEL="$ANTHROPIC_SUBAGENT_MODEL"

# ── Incremental scoring against the Opus-crafted reference ──────────
#
# Setting ATELIER_GROUND_TRUTH_PATH activates the per-iteration
# scorer wired into the bootstrap loop.  The reference POC GT at the path
# below is the canonical 920-column reference; the vocab-mismatch
# guard at pipeline startup will fail fast if the loaded vocabulary
# doesn't align (the bel_threshold-2026-05-15T22:42:57Z signature).
export ATELIER_GROUND_TRUTH_PATH="/home/cdsw/build/data/ground_truth/ground_truth.json"

# ── Sanity print to stderr ──────────────────────────────────────────
{
  echo "[wrapper] ATELIER_DB_URL=${ATELIER_DB_URL}"
  echo "[wrapper] ANTHROPIC_SUBAGENT_MODEL=${ANTHROPIC_SUBAGENT_MODEL:0:80}..."
  echo "[wrapper] ATELIER_GPU_SHARD_THRESHOLD=${ATELIER_GPU_SHARD_THRESHOLD}"
  echo "[wrapper] ATELIER_GROUND_TRUTH_PATH=${ATELIER_GROUND_TRUTH_PATH}"
  echo "[wrapper] AWS_REGION=${AWS_REGION}"
  echo "[wrapper] starting matrix sweep at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >&2

# ── Sweep invocation ────────────────────────────────────────────────
#
# If the caller passed --axis or other flags, forward them all.  If
# called with no args, run the agreed-on 2D bel_threshold × gap_threshold
# grid (12 runs, ~16 hours on this host).
if [ $# -eq 0 ]; then
    exec python3 scripts/sweep_matrix.py \
        --source-id hive-poc/reference_corpus \
        --axis classify.cautious_review.bel_threshold=0.55,0.65,0.75,0.80 \
        --axis classify.bootstrap.gap_threshold=0.12,0.18,0.24 \
        --name bel_x_gap \
        --output-dir build/sweeps \
        --continue-on-failure
else
    exec python3 scripts/sweep_matrix.py \
        --source-id hive-poc/reference_corpus \
        --output-dir build/sweeps \
        --continue-on-failure \
        "$@"
fi
