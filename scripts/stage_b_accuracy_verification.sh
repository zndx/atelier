#!/usr/bin/env bash
# Stage B — accuracy verification, executable in the long-lived CAI
# Application session after Stage A lands.
#
# Composes three existing scripts:
#   1. scripts/dst_sensitivity_study.py — synthetic-cell DST invariant
#      battery (Rec 7 part 1)
#   2. scripts/sweep_matrix.py          — bel × gap parameter sweep
#   3. scripts/score_sweep_matrix.py    — score sweep cells against
#      agent-mediated reference
#   4. inline jq aggregation            — K distribution from
#      per-column results (Rec 7 part 2 — consumes Stage A's
#      late_interaction_channel_conflict_k field)
#
# Run from project root (/home/cdsw on the CAI Application pod):
#   chmod +x scripts/stage_b_accuracy_verification.sh
#   SOURCE_ID=<your hive-poc source id> ./scripts/stage_b_accuracy_verification.sh
#
# Expected wall-clock: ~1-2 hours (12 sweep cells × ~5-10 min each).

set -euo pipefail

# ── Inputs (override via env) ────────────────────────────────────────
SOURCE_ID="${SOURCE_ID:-hive-poc/default}"
AGENT_MEDIATED="${AGENT_MEDIATED:-build/data/agent_mediated/agent_mediated.json}"
OUTPUT_DIR="${OUTPUT_DIR:-build/sweeps}"
NAME="${NAME:-bel_x_gap}"
BASELINE_CELL="${BASELINE_CELL:-bel=0.75,gap=0.18}"

# Bel × gap grid (4 × 3 = 12 cells)
BEL_VALUES="${BEL_VALUES:-0.55,0.65,0.75,0.80}"
GAP_VALUES="${GAP_VALUES:-0.12,0.18,0.24}"

# ── Preflight ────────────────────────────────────────────────────────
bel_count=$(tr ',' '\n' <<<"$BEL_VALUES" | wc -l)
gap_count=$(tr ',' '\n' <<<"$GAP_VALUES" | wc -l)
total_cells=$((bel_count * gap_count))

echo "== Stage B accuracy verification =="
echo "  source-id:     $SOURCE_ID"
echo "  agent-mediated reference:  $AGENT_MEDIATED"
echo "  grid:          bel={$BEL_VALUES} × gap={$GAP_VALUES} = $total_cells cells"
echo "  baseline cell: $BASELINE_CELL"
echo

# Materialize HOCON → atelier.env and source it
just resolve-config >/dev/null
# shellcheck disable=SC1091
source build/config/atelier.env

[ -f "$AGENT_MEDIATED" ] || {
    echo "ERROR: agent-mediated reference file not found: $AGENT_MEDIATED"
    exit 1
}

# ── 1. DST sensitivity synthetic-cell re-run ─────────────────────────
# Validates Stage A's instrumentation didn't break DST form-correctness
# invariants (mass conservation, monotonicity, ranking stability, etc.).
echo "[1/4] DST sensitivity study — synthetic-cell battery..."
uv run python scripts/dst_sensitivity_study.py

SENS_DIR=$(ls -td build/sensitivity/dst-* | head -1)
VIOLATIONS=$(jq 'length' "$SENS_DIR/violations.json" 2>/dev/null || echo "?")
if [ "$VIOLATIONS" != "0" ]; then
    echo "ERROR: $VIOLATIONS DST invariant violation(s). Inspect $SENS_DIR/violations.json"
    exit 1
fi
echo "  ✓ All DST invariants intact (0 violations)"
echo

# ── 2. Bel × gap parameter sweep ─────────────────────────────────────
echo "[2/4] Bel × gap parameter sweep — full classify pipeline per cell..."
uv run python scripts/sweep_matrix.py \
    --axis "classify.cautious_review.bel_threshold=$BEL_VALUES" \
    --axis "classify.bootstrap.gap_threshold=$GAP_VALUES" \
    --source-id "$SOURCE_ID" \
    --name "$NAME" \
    --output-dir "$OUTPUT_DIR" \
    --continue-on-failure 2>&1 | tee /tmp/sweep_run.log

# Locate the manifest the sweep emitted (most recent matching $NAME)
MANIFEST=$(ls -t "$OUTPUT_DIR/$NAME"-*.json 2>/dev/null | head -1)
if [ -z "$MANIFEST" ] || [ ! -f "$MANIFEST" ]; then
    echo "ERROR: Sweep did not produce a manifest. See /tmp/sweep_run.log"
    exit 1
fi
echo "  ✓ Sweep manifest: $MANIFEST"

# Confirm at least 80% of cells succeeded
OK_CELLS=$(jq '[.runs[] | select(.status == "ok")] | length' "$MANIFEST")
TOTAL_CELLS=$(jq '.grid_size' "$MANIFEST")
if [ "$OK_CELLS" -lt $((TOTAL_CELLS * 8 / 10)) ]; then
    echo "WARN: only $OK_CELLS/$TOTAL_CELLS cells succeeded. Inspect failed runs in $MANIFEST"
fi
echo "  $OK_CELLS/$TOTAL_CELLS cells completed successfully"
echo

# ── 3. Score against agent-mediated reference ────────────────────────
echo "[3/4] Score sweep outputs against agent-mediated reference..."
uv run python scripts/score_sweep_matrix.py \
    --manifest "$MANIFEST" \
    --reference "$AGENT_MEDIATED" \
    --results-dir build/results \
    --output-dir "$OUTPUT_DIR" \
    --baseline-cell "$BASELINE_CELL" \
    --binary-sensitivity

SCORING=$(ls -t "$OUTPUT_DIR"/scoring_matrix-*.json | head -1)
echo "  ✓ Scoring artifact: $SCORING"
echo

echo "Top 5 cells by strict accuracy:"
jq -r '.per_cell | to_entries | sort_by(-.value.strict_pct) | .[:5] |
       .[] | "  \(.value.strict_pct | tostring + "%" | (. + "       ")[0:8])\(.key)   (run \(.value.run_id))"' \
    "$SCORING"
echo

# ── 4. K distribution analysis (Rec 7 part 2) ────────────────────────
# Pulls late_interaction_channel_conflict_k off every column in the
# top-scoring run. Requires Stage A's instrumentation (per-column field).
echo "[4/4] K distribution analysis (Stage A's Rec 1 field)..."
TOP_RUN=$(jq -r '.per_cell | to_entries | sort_by(-.value.strict_pct) |
                 .[0].value.run_id' "$SCORING")

K_HIST=$(jq -r '
  [.[] | .late_interaction_channel_conflict_k // empty]
  | if length == 0 then
      {note: "No K values found — Stage A may not be deployed, or classifier did not fire late_interaction path"}
    else
      {n: length,
       mean: (add / length),
       max: max,
       p50: (sort | .[(length * 0.50) | floor]),
       p75: (sort | .[(length * 0.75) | floor]),
       p90: (sort | .[(length * 0.90) | floor]),
       p95: (sort | .[(length * 0.95) | floor]),
       fraction_above_0_5: ([.[] | select(. > 0.5)] | length / length)}
    end' \
    "build/results/$TOP_RUN/classifications.json")

echo "K distribution from top run ($TOP_RUN):"
echo "$K_HIST" | jq .
echo

# Also report top_kind + subtree_concentration distributions for
# Findings 4 (concentration cliff) + 6 (leaf/internal switch) diagnosis
echo "Top-kind + subtree-concentration distributions for top run:"
jq -r '
  {top_kind_counts: ([.[] | .top_kind] | group_by(.) | map({(.[0]): length}) | add),
   subtree_concentration_p50: ([.[] | .subtree_concentration // empty] | sort | .[(length * 0.5) | floor]),
   subtree_concentration_fraction_in_cliff_zone:
     ([.[] | .subtree_concentration // empty | select(. >= 0.45 and . <= 0.55)] | length /
      ([.[] | .subtree_concentration // empty] | length // 1))}
  | walk(if type == "number" then (. * 1000 | floor) / 1000 else . end)' \
  "build/results/$TOP_RUN/classifications.json"

# ── Final summary ────────────────────────────────────────────────────
echo
echo "=============================================="
echo "Stage B verification complete"
echo "=============================================="
echo "  Sweep manifest:    $MANIFEST"
echo "  Scoring matrix:    $SCORING"
echo "  Sensitivity dir:   $SENS_DIR"
echo
echo "Interpret the result:"
echo "  • Strict ≥ 95% on best cell  → architecture meets ~95% target."
echo "                                  Stage C (smoothing / hysteresis)"
echo "                                  becomes UX-only; BertSubs"
echo "                                  trajectory moves from 'needed' to"
echo "                                  'future refinement'."
echo "  • Strict 85–94% on best cell → Stage C recs 4+5 (sigmoid +"
echo "                                  hysteresis) likely load-bearing."
echo "                                  Use K_HIST.fraction_above_0_5 to"
echo "                                  pick β (Rec 2 — if the meaningful"
echo "                                  K tail sits below 0.5, β is too"
echo "                                  tight; loosen toward 0.40)."
echo "                                  Use subtree_concentration_fraction"
echo "                                  _in_cliff_zone to assess Rec 4"
echo "                                  (sigmoid smoothing) priority."
echo "  • Strict < 85%               → diagnose via per_column_diff CSV"
echo "                                  in $OUTPUT_DIR; likely needs"
echo "                                  taxonomy work + algorithmic"
echo "                                  smoothing in parallel."
