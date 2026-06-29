import type { Node, Edge } from "@xyflow/react";

// FSM pipeline node data — the live canvas path. (Legacy orchestrator/
// keystone/dynamicAgent/artifact node types are not used by ui-next.)

export type FsmPhaseStatus =
  | "idle"
  | "upcoming"
  | "current"
  | "completed"
  | "error"
  | "converged";

export interface FsmPhaseData extends Record<string, unknown> {
  state: string;
  label: string;
  description: string;
  status: FsmPhaseStatus;
}

export type SkillStatus = "available" | "active" | "unconfigured";

export interface SkillData extends Record<string, unknown> {
  skillId: string;
  label: string;
  description: string;
  model?: string;
  status: SkillStatus;
}

export type FsmPhaseNode = Node<FsmPhaseData, "fsmPhase">;
export type SkillNode = Node<SkillData, "skill">;

export type CanvasNode = FsmPhaseNode | SkillNode;
export type CanvasEdge = Edge;
