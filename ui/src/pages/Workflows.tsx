import { useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  BackgroundVariant,
  ReactFlowProvider,
  type Node,
  type Edge,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { Button, Spin, Typography, Space, Tag } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { nodeTypes } from "../components/canvas/nodeTypes";
import {
  buildFsmPipelineTopology,
  type RuntimeCapabilities,
} from "../lib/fsmPipelineLayout";

const { Title, Text } = Typography;

const MINIMAP_COLORS: Record<string, string> = {
  fsmPhase: "#96a2fc",
  skill: "#d99d54",
  // Legacy node types — preserved so the keystone-topology view (or
  // a future Agent Studio overlay) can light them up if it lands.
  orchestrator: "#101418",
  keystone: "#96a2fc",
  dynamicAgent: "#d99d54",
  artifact: "#30363c",
};

const FSM_POLL_INTERVAL_MS = 5000;

const STATE_COLOR: Record<string, string> = {
  IDLE: "default",
  LOADING_VOCAB: "blue",
  DISCOVERING: "blue",
  SAMPLING: "blue",
  LLM_SWEEP: "blue",
  VALIDATING: "blue",
  CLASSIFYING: "blue",
  FUSING: "blue",
  EVALUATING: "blue",
  CONVERGED: "green",
  ERROR: "red",
};

function WorkflowsInner() {
  const [loading, setLoading] = useState(true);
  const [currentState, setCurrentState] = useState<string | null>(null);
  const [capabilities, setCapabilities] = useState<RuntimeCapabilities>({
    cautious_review_enabled: false,
    overwatch_enabled: false,
    classify_agent_enabled: false,
    precondition_enabled: false,
  });

  const fetchCapabilities = () => {
    fetch("/api/status")
      .then((r) => r.json())
      .then((data) => {
        const cfg = data?.config || {};
        setCapabilities({
          cautious_review_enabled: Boolean(cfg.cautious_review_enabled),
          overwatch_enabled: Boolean(cfg.overwatch_enabled),
          classify_agent_enabled: Boolean(cfg.classify_agent_enabled),
          precondition_enabled: Boolean(cfg.precondition_enabled),
        });
      })
      .catch(() => {
        // Leave capabilities at defaults — every skill will be omitted
        // from the graph; phase nodes still render.
      });
  };

  const fetchFsmState = () => {
    fetch("/api/fsm/status")
      .then((r) => r.json())
      .then((data) => {
        setCurrentState(data?.state ?? null);
      })
      .catch(() => setCurrentState(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchCapabilities();
    fetchFsmState();
    const interval = setInterval(fetchFsmState, FSM_POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  const { nodes, edges } = useMemo<{ nodes: Node[]; edges: Edge[] }>(() => {
    const topo = buildFsmPipelineTopology(currentState, capabilities);
    return { nodes: topo.nodes as Node[], edges: topo.edges };
  }, [currentState, capabilities]);

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100%",
        }}
      >
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid var(--color-kumo-line)",
          background: "var(--color-kumo-elevated)",
          flexShrink: 0,
        }}
      >
        <Space>
          <Title level={5} style={{ margin: 0 }}>
            Pipeline Workflow
          </Title>
          <Text type="secondary" style={{ fontSize: 12 }}>
            FSM phases · skill attachments · live state
          </Text>
        </Space>
        <Space>
          <Tag color={STATE_COLOR[currentState ?? "IDLE"] ?? "default"}>
            {currentState ?? "IDLE"}
          </Tag>
          <Button
            icon={<ReloadOutlined />}
            size="small"
            onClick={() => {
              fetchCapabilities();
              fetchFsmState();
            }}
          >
            Refresh
          </Button>
        </Space>
      </div>

      <div style={{ flex: 1 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.25 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          proOptions={{ hideAttribution: false }}
        >
          {/* SVG pattern fill — attribute context, so a kumo literal (k4 slate+):
              subtle on the dark canvas, faint-but-visible on light. */}
          <Background
            variant={BackgroundVariant.Dots}
            gap={16}
            size={1}
            color="#30363c"
          />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={(node) => MINIMAP_COLORS[node.type || ""] || "#30363c"}
            maskColor="rgba(0, 0, 0, 0.08)"
            style={{ borderRadius: 6 }}
          />
        </ReactFlow>
      </div>
    </div>
  );
}

export default function Workflows() {
  return (
    <ReactFlowProvider>
      <WorkflowsInner />
    </ReactFlowProvider>
  );
}
