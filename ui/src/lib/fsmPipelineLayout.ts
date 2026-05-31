// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { MarkerType } from "@xyflow/react";
import type { Edge } from "@xyflow/react";
import type {
  CanvasNode,
  FsmPhaseStatus,
  SkillStatus,
} from "../types/canvas";

// ── Phase definitions ──────────────────────────────────────────────
//
// One entry per FSM state in pipeline order.  Hover tooltips on
// FsmPhaseNode read from ``description``.

interface PhaseDef {
  state: string;
  label: string;
  description: string;
}

const PHASES: PhaseDef[] = [
  {
    state: "IDLE",
    label: "Idle",
    description: "No run in flight; ready to dispatch the next classification.",
  },
  {
    state: "LOADING_VOCAB",
    label: "Loading Vocab",
    description:
      "Load the user-supplied taxonomy and validate it for label collisions, duplicate codes, and orphaned aliases.",
  },
  {
    state: "DISCOVERING",
    label: "Discovering",
    description:
      "Probe the data source via cml.data_v1 to enumerate tables.",
  },
  {
    state: "SAMPLING",
    label: "Sampling",
    description:
      "Sample column metadata and a representative slice of values from each table.  Reference-key columns are filtered out before classification.",
  },
  {
    state: "LLM_SWEEP",
    label: "LLM Sweep",
    description:
      "Claude classifies each frontier-tier column into the user vocabulary; iteration 1 sweeps every column, later iterations revisit only disagreements.",
  },
  {
    state: "VALIDATING",
    label: "Validating",
    description:
      "ML models (CatBoost-fit-to-LLM, synth-trained SVM via the ICE→user alignment) re-validate the LLM's labels.  Disagreements feed the next iteration's revisit batch.",
  },
  {
    state: "CLASSIFYING",
    label: "Classifying",
    description:
      "Final per-column DST evidence fusion across name match, pattern, maxsim, LLM, CatBoost, SVM.",
  },
  {
    state: "FUSING",
    label: "Fusing",
    description:
      "Combine the per-column mass functions via Dempster's rule (or Yager) to produce the headline classification with belief, plausibility, and conflict.",
  },
  {
    state: "EVALUATING",
    label: "Evaluating",
    description:
      "Compute accuracy + per-category metrics; cautious-code review (when enabled) backs off over-specified predictions whose belief is below threshold.",
  },
  {
    state: "CONVERGED",
    label: "Converged",
    description:
      "Run completed — convergence criteria satisfied (mean_gap < gap_threshold, or coverage met, or max_iterations reached).",
  },
  {
    state: "ERROR",
    label: "Error",
    description:
      "Run failed — terminal error state.  Inspect the FSM error field and run logs for the root cause.",
  },
];

// ── Skill definitions ──────────────────────────────────────────────
//
// Each skill declares which capability flag gates it, which phase
// it attaches to, and which FSM state(s) trigger its ``active``
// highlight.  The list is the registry-MVP — for now hand-coded;
// when we hit ~6 surfaces this graduates to a backend ``/api/skills``
// endpoint reading from a real registry.

interface SkillDef {
  id: string;
  label: string;
  description: string;
  /** Which phase node this skill hangs off. */
  hostPhase: string;
  /** Vertical placement relative to the host phase. */
  position: "above" | "below";
  /** Optional model-family tag for the operator's awareness. */
  model?: string;
  /** When this FSM state matches, the skill renders ``active``. */
  activeStates: string[];
  /** Capability key to consult on ``capabilities`` to decide
   *  whether the skill is rendered at all. */
  capabilityKey: keyof RuntimeCapabilities;
}

const SKILLS: SkillDef[] = [
  {
    id: "agent_convergence",
    label: "Agent Convergence",
    description:
      "Claude drives the convergence loop, deciding which columns to revisit and when to declare convergence.  Replaces the programmatic loop when enabled.",
    hostPhase: "VALIDATING",
    position: "above",
    model: "Anthropic / Bedrock",
    activeStates: ["LLM_SWEEP", "VALIDATING"],
    capabilityKey: "classify_agent_enabled",
  },
  {
    id: "cautious_review",
    label: "Cautious Review",
    description:
      "Agent-mediated backoff for over-specified predictions: when belief at the predicted depth is weak, Claude considers backing off to the cautious code instead.",
    hostPhase: "EVALUATING",
    position: "below",
    model: "Subagent / Anthropic-direct",
    activeStates: ["EVALUATING"],
    capabilityKey: "cautious_review_enabled",
  },
  {
    id: "overwatch",
    label: "Overwatch",
    description:
      "Single-turn Opus analysis at run end; writes pipeline recommendations to overwatch.md.  Follows the Web Terminal Agent's selected model — direct Anthropic API or Bedrock, whichever the operator picked.",
    hostPhase: "CONVERGED",
    position: "above",
    model: "Follows Web Terminal Agent",
    activeStates: ["CONVERGED"],
    capabilityKey: "overwatch_enabled",
  },
];

// ── Capabilities + status derivation ───────────────────────────────

export interface RuntimeCapabilities {
  cautious_review_enabled: boolean;
  overwatch_enabled: boolean;
  classify_agent_enabled: boolean;
}

const PHASE_INDEX: Record<string, number> = Object.fromEntries(
  PHASES.map((p, i) => [p.state, i]),
);

const TERMINAL_STATES = new Set(["IDLE", "CONVERGED", "ERROR"]);

function phaseStatusFor(
  phase: PhaseDef,
  current: string | null,
): FsmPhaseStatus {
  if (!current || current === "IDLE") {
    return phase.state === "IDLE" ? "idle" : "upcoming";
  }
  if (current === "ERROR") {
    return phase.state === "ERROR" ? "error" : "upcoming";
  }
  if (current === "CONVERGED") {
    if (phase.state === "CONVERGED") return "converged";
    if (phase.state === "ERROR" || phase.state === "IDLE") return "upcoming";
    return "completed";
  }
  if (phase.state === current) return "current";
  // ERROR / IDLE / CONVERGED nodes are off-track for in-flight runs.
  if (phase.state === "ERROR" || phase.state === "IDLE" || phase.state === "CONVERGED") {
    return "upcoming";
  }
  // Linear pipeline ordering: states earlier in PHASES are completed.
  const cur = PHASE_INDEX[current] ?? -1;
  const me = PHASE_INDEX[phase.state] ?? -1;
  if (cur < 0 || me < 0) return "upcoming";
  return me < cur ? "completed" : "upcoming";
}

function skillStatusFor(skill: SkillDef, current: string | null): SkillStatus {
  if (!current || TERMINAL_STATES.has(current) === false) {
    if (current && skill.activeStates.includes(current)) return "active";
  } else if (skill.activeStates.includes(current)) {
    return "active";
  }
  return "available";
}

// ── Layout geometry ────────────────────────────────────────────────

const PHASE_X_SPACING = 170;
const PHASE_Y_MAIN = 120;
const PHASE_Y_TERMINAL_OFFSET = 110; // ERROR sits below the main row
const SKILL_Y_OFFSET = 75;
const SKILL_X_NUDGE = 10;

export interface FsmPipelineTopology {
  nodes: CanvasNode[];
  edges: Edge[];
}

export function buildFsmPipelineTopology(
  currentState: string | null,
  capabilities: RuntimeCapabilities,
): FsmPipelineTopology {
  const nodes: CanvasNode[] = [];
  const edges: Edge[] = [];

  // ── Phase nodes ──
  // Linear horizontal row (excluding ERROR which sits below).
  const linearPhases = PHASES.filter((p) => p.state !== "ERROR");
  const phasePositions: Record<string, { x: number; y: number }> = {};

  linearPhases.forEach((phase, i) => {
    const x = i * PHASE_X_SPACING;
    const y = PHASE_Y_MAIN;
    phasePositions[phase.state] = { x, y };
    nodes.push({
      id: phase.state,
      type: "fsmPhase",
      position: { x, y },
      draggable: false,
      data: {
        state: phase.state,
        label: phase.label,
        description: phase.description,
        status: phaseStatusFor(phase, currentState),
      },
    });
  });

  // ERROR off the main row, vertically below SAMPLING (rough mid-graph).
  const errorAnchorIdx = linearPhases.findIndex((p) => p.state === "SAMPLING");
  const errorPhase = PHASES.find((p) => p.state === "ERROR")!;
  const errorPos = {
    x: errorAnchorIdx >= 0 ? errorAnchorIdx * PHASE_X_SPACING : 0,
    y: PHASE_Y_MAIN + PHASE_Y_TERMINAL_OFFSET,
  };
  phasePositions["ERROR"] = errorPos;
  nodes.push({
    id: "ERROR",
    type: "fsmPhase",
    position: errorPos,
    draggable: false,
    data: {
      state: errorPhase.state,
      label: errorPhase.label,
      description: errorPhase.description,
      status: phaseStatusFor(errorPhase, currentState),
    },
  });

  // ── Phase edges ──
  // Linear flow.
  for (let i = 0; i < linearPhases.length - 1; i += 1) {
    const a = linearPhases[i].state;
    const b = linearPhases[i + 1].state;
    edges.push({
      id: `${a}->${b}`,
      source: a,
      target: b,
      type: "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14 },
      style: { stroke: "#bfbfbf" },
    });
  }

  // Iteration loop: VALIDATING → LLM_SWEEP back-edge.  Drawn as a
  // visually distinct curved line above the main row to signal that
  // the loop is the algorithm's central structure.
  edges.push({
    id: "VALIDATING->LLM_SWEEP",
    source: "VALIDATING",
    target: "LLM_SWEEP",
    type: "smoothstep",
    label: "revisit",
    labelStyle: { fontSize: 11, fill: "#722ed1", fontWeight: 600 },
    labelBgStyle: { fill: "#f9f0ff" },
    labelBgPadding: [4, 2],
    labelBgBorderRadius: 4,
    markerEnd: { type: MarkerType.ArrowClosed, width: 14, height: 14, color: "#722ed1" },
    style: { stroke: "#722ed1", strokeDasharray: "4 3" },
  });

  // ERROR edges from each non-terminal phase (dashed, faint).
  for (const p of linearPhases) {
    if (p.state === "IDLE" || p.state === "CONVERGED") continue;
    edges.push({
      id: `${p.state}->ERROR`,
      source: p.state,
      target: "ERROR",
      type: "smoothstep",
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: "#ffa39e" },
      style: { stroke: "#ffa39e", strokeDasharray: "2 4", opacity: 0.55 },
    });
  }

  // ── Skill nodes ──
  for (const skill of SKILLS) {
    if (!capabilities[skill.capabilityKey]) continue;
    const host = phasePositions[skill.hostPhase];
    if (!host) continue;
    const dy = skill.position === "above" ? -SKILL_Y_OFFSET : SKILL_Y_OFFSET;
    const skillId = `skill:${skill.id}`;
    nodes.push({
      id: skillId,
      type: "skill",
      position: { x: host.x + SKILL_X_NUDGE, y: host.y + dy },
      draggable: false,
      data: {
        skillId: skill.id,
        label: skill.label,
        description: skill.description,
        model: skill.model,
        status: skillStatusFor(skill, currentState),
      },
    });
    // Edge from host phase up/down to the skill (no arrow head — the
    // skill participates in the phase rather than being a successor).
    edges.push({
      id: `${skill.hostPhase}->${skillId}`,
      source: skill.hostPhase,
      target: skillId,
      type: "smoothstep",
      style: {
        stroke: "#fa8c16",
        strokeDasharray: "2 3",
        opacity: 0.55,
      },
    });
  }

  return { nodes, edges };
}
