import { useMemo } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type Node,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { nodeTypes } from "./canvas/nodeTypes";
import {
  buildFsmPipelineTopology,
  type RuntimeCapabilities,
} from "../lib/fsmPipelineLayout";

const MINIMAP_COLORS: Record<string, string> = {
  fsmPhase: "#6366f1",
  skill: "#a855f7",
};

function CanvasInner({
  state,
  capabilities,
}: {
  state: string | null;
  capabilities: RuntimeCapabilities;
}) {
  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const topo = buildFsmPipelineTopology(state, capabilities);
    return { nodes: topo.nodes as Node[], edges: topo.edges };
  }, [state, capabilities]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      fitViewOptions={{ padding: 0.25 }}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      proOptions={{ hideAttribution: true }}
      colorMode="dark"
    >
      <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#242430" />
      <Controls showInteractive={false} />
      <MiniMap
        nodeColor={(n) => MINIMAP_COLORS[n.type || ""] || "#2e2e3a"}
        maskColor="rgba(0,0,0,0.4)"
        pannable
        zoomable
      />
    </ReactFlow>
  );
}

export function PipelineCanvas({
  state,
  capabilities,
}: {
  state: string | null;
  capabilities: RuntimeCapabilities;
}) {
  return (
    <ReactFlowProvider>
      <CanvasInner state={state} capabilities={capabilities} />
    </ReactFlowProvider>
  );
}
