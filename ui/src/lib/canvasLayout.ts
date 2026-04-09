import { MarkerType } from "@xyflow/react";
import type { Edge } from "@xyflow/react";
import type { AgentInfo, CanvasNode, CanvasEdgeData } from "../types/canvas";

const KEYSTONE_ORDER: Record<string, number> = {
  classifier: 0,
  evidence_fuser: 1,
  visualization_director: 2,
};

const TIER_SPACING = 180;
const NODE_SPACING = 280;
const ORCHESTRATOR_Y = 50;

export interface CanvasTopology {
  nodes: CanvasNode[];
  edges: Edge<CanvasEdgeData>[];
}

export function buildKeystoneTopology(
  agents: AgentInfo[],
  modelName: string,
): CanvasTopology {
  const nodes: CanvasNode[] = [];
  const edges: Edge<CanvasEdgeData>[] = [];

  const sorted = [...agents].sort(
    (a, b) => (KEYSTONE_ORDER[a.role] ?? 99) - (KEYSTONE_ORDER[b.role] ?? 99),
  );

  // Tier 0: Orchestrator centered above keystones
  const keystoneWidth = Math.max(sorted.length, 1) * NODE_SPACING;
  const orchestratorX = (keystoneWidth - NODE_SPACING) / 2;

  nodes.push({
    id: "orchestrator",
    type: "orchestrator",
    position: { x: orchestratorX, y: ORCHESTRATOR_Y },
    draggable: false,
    data: {
      label: "Claude Orchestrator",
      model: modelName,
      status: "idle",
    },
  });

  // Tier 1: Keystone agents
  const keystoneY = ORCHESTRATOR_Y + TIER_SPACING;
  sorted.forEach((agent, index) => {
    const nodeId = `keystone-${agent.id}`;
    nodes.push({
      id: nodeId,
      type: "keystone",
      position: { x: index * NODE_SPACING, y: keystoneY },
      draggable: false,
      data: {
        agentId: agent.id,
        name: agent.name,
        description: agent.description,
        role: agent.role,
        status: "idle",
      },
    });

    // Orchestrator -> keystone edge
    edges.push({
      id: `e-orch-${agent.id}`,
      source: "orchestrator",
      target: nodeId,
      type: "smoothstep",
      animated: false,
      data: { edgeType: "orchestration" as const },
      markerEnd: { type: MarkerType.ArrowClosed, width: 16, height: 16 },
      style: { stroke: "#1890ff", strokeWidth: 2 },
    });
  });

  // Evidence convergence edges
  const classifierNode = sorted.find((a) => a.role === "classifier");
  const fuserNode = sorted.find((a) => a.role === "evidence_fuser");
  const vizNode = sorted.find((a) => a.role === "visualization_director");

  if (classifierNode && fuserNode) {
    edges.push({
      id: "e-conv-classifier-fuser",
      source: `keystone-${classifierNode.id}`,
      target: `keystone-${fuserNode.id}`,
      sourceHandle: "right",
      targetHandle: "left",
      type: "smoothstep",
      data: { edgeType: "convergence" as const, label: "evidence" },
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
      style: { stroke: "#52c41a", strokeWidth: 1.5, strokeDasharray: "6 3" },
      label: "evidence",
      labelStyle: { fontSize: 10, fill: "#52c41a" },
    });
  }

  if (vizNode && fuserNode) {
    edges.push({
      id: "e-conv-viz-fuser",
      source: `keystone-${vizNode.id}`,
      target: `keystone-${fuserNode.id}`,
      sourceHandle: "left",
      targetHandle: "right",
      type: "smoothstep",
      data: { edgeType: "convergence" as const, label: "embeddings" },
      markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12 },
      style: { stroke: "#52c41a", strokeWidth: 1.5, strokeDasharray: "6 3" },
      label: "embeddings",
      labelStyle: { fontSize: 10, fill: "#52c41a" },
    });
  }

  return { nodes, edges };
}
