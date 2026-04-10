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

// ── Typed node aliases ──────────────────────────────────────────

export type OrchestratorNode = Node<OrchestratorNodeData, "orchestrator">;
export type KeystoneNode = Node<KeystoneNodeData, "keystone">;
export type DynamicAgentNode = Node<DynamicAgentNodeData, "dynamicAgent">;
export type ArtifactNode = Node<ArtifactNodeData, "artifact">;

export type CanvasNode = OrchestratorNode | KeystoneNode | DynamicAgentNode | ArtifactNode;

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
