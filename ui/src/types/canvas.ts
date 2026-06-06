import type { Node, Edge } from "@xyflow/react";

// ── Agent data from /api/agents ─────────────────────────────────

export interface AgentInfo {
  id: string;
  name: string;
  description: string;
  role: string;
  tool_ids: string[];
}

// ── Node data payloads ──────────────────────────────────────────

export interface OrchestratorNodeData extends Record<string, unknown> {
  label: string;
  model: string;
  status: "idle" | "orchestrating" | "completed";
}

export interface KeystoneNodeData extends Record<string, unknown> {
  agentId: string;
  name: string;
  description: string;
  role: string;
  status: "idle" | "active" | "completed" | "error";
  lastActivity?: string;
}

export interface DynamicAgentNodeData extends Record<string, unknown> {
  agentId: string;
  name: string;
  description: string;
  purpose: string;
  status: "spawning" | "active" | "completed" | "dismissed";
  ephemeral: true;
}

export interface ArtifactNodeData extends Record<string, unknown> {
  artifactId: string;
  name: string;
  type: "dataset" | "parquet" | "classification_result" | "evidence_vector" | "embedding";
  rowCount?: number;
  producedBy: string;
  status: "pending" | "ready" | "consumed";
}

// ── FSM pipeline node data ──────────────────────────────────────

/** Status of a phase node relative to the active run.
 *
 * - ``current``    — the run is in this phase right now (highlight)
 * - ``completed``  — the run advanced past this phase already
 * - ``upcoming``   — the run hasn't reached this phase yet (default)
 * - ``error``      — terminal ERROR state was reached
 * - ``converged``  — terminal CONVERGED state was reached
 * - ``idle``       — no run in flight
 */
export type FsmPhaseStatus =
  | "idle"
  | "upcoming"
  | "current"
  | "completed"
  | "error"
  | "converged";

export interface FsmPhaseData extends Record<string, unknown> {
  /** FSM state id, e.g. ``LLM_SWEEP``. */
  state: string;
  /** Display label, e.g. ``LLM Sweep``. */
  label: string;
  /** One-sentence tooltip describing what this phase does. */
  description: string;
  status: FsmPhaseStatus;
}

/** Skill nodes attach to phase nodes when the corresponding
 *  agentic surface is configured.  ``status`` mirrors the host
 *  phase's status when the skill is currently running, otherwise
 *  ``available`` (configured but not active) or ``unconfigured``
 *  (gated off via the capability flag — currently never rendered;
 *  the layout omits the node entirely). */
export type SkillStatus = "available" | "active" | "unconfigured";

export interface SkillData extends Record<string, unknown> {
  /** Stable skill id used for dedup + future registry lookup. */
  skillId: string;
  /** Display label, e.g. ``Cautious Review``. */
  label: string;
  /** One-sentence tooltip describing what the skill does. */
  description: string;
  /** Which model family services this skill — for the operator
   *  to know whether Anthropic-direct credentials are required. */
  model?: string;
  status: SkillStatus;
}

// ── Typed node aliases ──────────────────────────────────────────

export type OrchestratorNode = Node<OrchestratorNodeData, "orchestrator">;
export type KeystoneNode = Node<KeystoneNodeData, "keystone">;
export type DynamicAgentNode = Node<DynamicAgentNodeData, "dynamicAgent">;
export type ArtifactNode = Node<ArtifactNodeData, "artifact">;
export type FsmPhaseNode = Node<FsmPhaseData, "fsmPhase">;
export type SkillNode = Node<SkillData, "skill">;

export type CanvasNode =
  | OrchestratorNode
  | KeystoneNode
  | DynamicAgentNode
  | ArtifactNode
  | FsmPhaseNode
  | SkillNode;

// ── Edge data ───────────────────────────────────────────────────

export type EdgeType = "orchestration" | "dataflow" | "convergence";

export interface CanvasEdgeData extends Record<string, unknown> {
  edgeType: EdgeType;
  label?: string;
}

export type CanvasEdge = Edge<CanvasEdgeData>;

// ── WebSocket event protocol (future) ───────────────────────────

export type OrchestrationEvent =
  | { type: "agent_spawned"; agentId: string; name: string; purpose: string }
  | { type: "agent_active"; agentId: string }
  | { type: "agent_completed"; agentId: string; result?: string }
  | { type: "agent_dismissed"; agentId: string }
  | { type: "artifact_produced"; artifact: ArtifactNodeData; producedBy: string; consumedBy?: string[] }
  | { type: "orchestrator_status"; status: OrchestratorNodeData["status"] }
  | { type: "topology_reset" };
