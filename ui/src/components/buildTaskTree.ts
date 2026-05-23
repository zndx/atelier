// Tree-builder for the nested progress UI.
//
// Pure function: consumes the current FSM status from /api/fsm/status
// and returns an ordered flat list of ProgressTask rows.  The rendered
// "tree" is purely visual (depth-based indent + guide characters in
// PhaseProgress.tsx); no <Tree> widget needed.
//
// See `.claude/plans/lovely-doodling-badger.md` for the data model
// and the server-side payload shape this consumes.

export type TaskStatus =
  | "pending"
  | "active"
  | "done"
  | "skipped"
  | "error";

export interface ProgressDeterminate {
  current: number;
  total: number;
  unit?: string;
}

export interface ProgressTask {
  id: string;
  depth: 0 | 1 | 2;
  parentId: string | null;
  name: string;
  status: TaskStatus;
  determinate?: ProgressDeterminate;
  indeterminate?: boolean;
  elapsedS?: number;
  etaS?: number | null;
  subPhaseLabel?: string;
}

interface FSMStatusLike {
  state: string;
  progress?: Record<string, unknown>;
  error?: string | null;
}

// Canonical pipeline phase ordering — drives the "earlier = done,
// later = pending" derivation.  GENERATING_SYNTH and TRAINING are
// skipped on real-data runs (rendered as faded "skipped" rows only
// if the FSM ever passed through them; today's pipeline doesn't).
const PIPELINE_ORDER: string[] = [
  "LOADING_VOCAB",
  "DISCOVERING",
  "SAMPLING",
  "LLM_SWEEP",
  "VALIDATING",
  "CLASSIFYING",
  "FUSING",
  "EVALUATING",
  "CONVERGED",
];

// Human-readable phase labels for display.
const PHASE_LABEL: Record<string, string> = {
  LOADING_VOCAB: "Loading vocabulary",
  DISCOVERING: "Discovering tables",
  SAMPLING: "Sampling",
  LLM_SWEEP: "LLM sweep",
  VALIDATING: "ML validation",
  CLASSIFYING: "Classifying",
  FUSING: "Fusing",
  EVALUATING: "Evaluating",
  CONVERGED: "Converged",
};

function asNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

function asString(v: unknown): string | undefined {
  if (typeof v === "string" && v.length > 0) return v;
  return undefined;
}

interface DeterminateFromProgress {
  current: number;
  total: number;
  unit?: string;
}

// Phase-specific reading of the FSM `progress` payload for the
// "current/total" pair that becomes the bar's numerator/denominator.
function readDeterminate(
  phase: string,
  progress: Record<string, unknown>,
): DeterminateFromProgress | undefined {
  switch (phase) {
    case "SAMPLING": {
      // phase_heartbeat sub_phase block carries `current` + `total`
      // (commit 6194f5b).  Legacy `tables_sampled` retained for old
      // payload shape.
      const sub = progress["sub_phase"] as Record<string, unknown> | undefined;
      const cur = asNumber(sub?.current) ?? asNumber(progress["phase_current"]);
      const tot = asNumber(sub?.total) ?? asNumber(progress["phase_total"]);
      if (cur !== undefined && tot !== undefined && tot > 0) {
        return { current: cur, total: tot, unit: "tables" };
      }
      const sampled = asNumber(progress["phase_tables_sampled"]);
      const discovered = asNumber(progress["tables_discovered"]);
      if (sampled !== undefined && discovered !== undefined && discovered > 0) {
        return { current: sampled, total: discovered, unit: "tables" };
      }
      return undefined;
    }
    case "LLM_SWEEP": {
      const labeled = asNumber(progress["llm_labeled"]);
      const total = asNumber(progress["columns_total"]);
      if (labeled !== undefined && total !== undefined && total > 0) {
        return { current: labeled, total, unit: "columns" };
      }
      return undefined;
    }
    case "VALIDATING":
    case "CLASSIFYING": {
      const sub = progress["sub_phase"] as Record<string, unknown> | undefined;
      const cur = asNumber(sub?.current);
      const tot = asNumber(sub?.total);
      if (cur !== undefined && tot !== undefined && tot > 0) {
        return { current: cur, total: tot, unit: "columns" };
      }
      return undefined;
    }
    default:
      return undefined;
  }
}

function readSubPhase(
  progress: Record<string, unknown>,
): { id: string; determinate?: ProgressDeterminate; elapsedS?: number } | undefined {
  const sub = progress["sub_phase"] as Record<string, unknown> | undefined;
  if (sub && typeof sub === "object") {
    const id = asString(sub.id);
    if (!id) return undefined;
    const current = asNumber(sub.current);
    const total = asNumber(sub.total);
    const determinate =
      current !== undefined && total !== undefined && total > 0
        ? { current, total }
        : undefined;
    const elapsedS = asNumber(sub.elapsed_s);
    return { id, determinate, elapsedS };
  }
  // Fallback: phase_label carries the sub-phase name without
  // structured progress (indeterminate sub-phase row).
  const phaseLabel = asString(progress["phase_label"]);
  if (phaseLabel) {
    const elapsedS = asNumber(progress["phase_seconds"]);
    return { id: phaseLabel, elapsedS };
  }
  return undefined;
}

function statusFor(
  phase: string,
  currentState: string,
  isError: boolean,
): TaskStatus {
  if (phase === currentState) {
    return isError ? "error" : "active";
  }
  const phaseIdx = PIPELINE_ORDER.indexOf(phase);
  const curIdx = PIPELINE_ORDER.indexOf(currentState);
  if (phaseIdx === -1 || curIdx === -1) return "pending";
  if (phaseIdx < curIdx) return "done";
  return "pending";
}

/**
 * Build the ordered flat list of ProgressTask rows for the current
 * FSM snapshot.  Caller renders each via <PhaseProgress depth={...}>.
 *
 * Tree shape:
 *   depth-0: aggregate pipeline-overall progress (one row, top)
 *   depth-1: per-FSM-phase rows
 *   depth-2: active sub-phase row under the currently active phase
 */
export function buildTaskTree(fsm: FSMStatusLike | null): ProgressTask[] {
  if (!fsm) return [];
  const currentState = fsm.state || "IDLE";
  const progress = (fsm.progress || {}) as Record<string, unknown>;
  const isError = currentState === "ERROR";

  const tasks: ProgressTask[] = [];

  // Depth-0 aggregate row: counts how many ordered phases have
  // completed.  CONVERGED is the terminal sentinel, not a phase,
  // so it's excluded from both denominator and "done" tally —
  // when the FSM is CONVERGED, all real phases register done.
  const realPhases = PIPELINE_ORDER.filter((p) => p !== "CONVERGED");
  const curIdx = PIPELINE_ORDER.indexOf(currentState);
  const donePhases =
    currentState === "CONVERGED"
      ? realPhases.length
      : Math.max(0, curIdx);
  const overallStatus: TaskStatus =
    isError
      ? "error"
      : currentState === "CONVERGED"
        ? "done"
        : currentState === "IDLE"
          ? "pending"
          : "active";
  tasks.push({
    id: "pipeline:root",
    depth: 0,
    parentId: null,
    name: "Pipeline",
    status: overallStatus,
    determinate: {
      current: donePhases,
      total: realPhases.length,
      unit: "phases",
    },
  });

  for (const phase of PIPELINE_ORDER) {
    const status = statusFor(phase, currentState, isError);
    if (status === "pending" && phase === "CONVERGED") {
      // Don't surface CONVERGED as a pending row — it's a terminal
      // state, not an activity.
      continue;
    }
    const isActive = status === "active" || status === "error";
    const determinate = isActive ? readDeterminate(phase, progress) : undefined;
    const elapsedS = isActive
      ? asNumber(progress["sweep_elapsed_s"]) ??
        asNumber(progress["phase_seconds"])
      : undefined;
    tasks.push({
      id: `fsm:${phase}`,
      depth: 1,
      parentId: null,
      name: PHASE_LABEL[phase] ?? phase,
      status,
      determinate,
      indeterminate: isActive && determinate === undefined,
      elapsedS,
    });

    // Sub-phase row appended only under the active phase.
    if (isActive) {
      const sub = readSubPhase(progress);
      if (sub) {
        tasks.push({
          id: `fsm:${phase}.sub:${sub.id}`,
          depth: 2,
          parentId: `fsm:${phase}`,
          name: sub.id,
          status: isError ? "error" : "active",
          determinate: sub.determinate,
          indeterminate: sub.determinate === undefined,
          elapsedS: sub.elapsedS,
          subPhaseLabel: sub.id,
        });
      }
    }
  }

  return tasks;
}
