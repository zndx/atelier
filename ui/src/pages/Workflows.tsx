// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { useEffect, useState, useCallback } from "react";
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
import { Button, Spin, Typography, Alert, Space } from "antd";
import { ReloadOutlined } from "@ant-design/icons";
import { nodeTypes } from "../components/canvas/nodeTypes";
import { buildKeystoneTopology } from "../lib/canvasLayout";
import type { AgentInfo } from "../types/canvas";

const { Title, Text } = Typography;

const MINIMAP_COLORS: Record<string, string> = {
  orchestrator: "#001529",
  keystone: "#1890ff",
  dynamicAgent: "#faad14",
  artifact: "#d9d9d9",
};

function WorkflowsInner() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);

  const fetchAndLayout = useCallback(async () => {
    setLoading(true);
    setError(null);
    // Tolerant JSON parser: backend plain-text 500s ("Internal Server
    // Error") would otherwise bubble up as "Unexpected token 'I'" and
    // hide the real failure.
    const safeJson = async (r: Response): Promise<any | null> => {
      try {
        const text = await r.text();
        if (!text) return null;
        try {
          return JSON.parse(text);
        } catch {
          return { error: text };
        }
      } catch {
        return null;
      }
    };
    try {
      const [agentsResp, statusResp] = await Promise.all([
        fetch("/api/agents").then(safeJson),
        fetch("/api/status")
          .then(safeJson)
          .catch(() => null),
      ]);

      if (agentsResp?.error) {
        throw new Error(agentsResp.error);
      }
      const agentList: AgentInfo[] = agentsResp?.agents || [];
      const model =
        statusResp?.config?.agent_model || "Claude (Agent SDK)";

      const topology = buildKeystoneTopology(agentList, model);
      setNodes(topology.nodes);
      setEdges(topology.edges);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [setNodes, setEdges]);

  useEffect(() => {
    fetchAndLayout();
  }, [fetchAndLayout]);

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

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <Alert
          type="error"
          message="Failed to load workflow topology"
          description={error}
          showIcon
          action={
            <Button size="small" onClick={fetchAndLayout}>
              Retry
            </Button>
          }
        />
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%", display: "flex", flexDirection: "column" }}>
      <div
        style={{
          padding: "12px 16px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid #f0f0f0",
          background: "#fff",
          flexShrink: 0,
        }}
      >
        <Space>
          <Title level={5} style={{ margin: 0 }}>
            Orchestration Canvas
          </Title>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Keystone agent topology
          </Text>
        </Space>
        <Button
          icon={<ReloadOutlined />}
          size="small"
          onClick={fetchAndLayout}
        >
          Refresh
        </Button>
      </div>

      <div style={{ flex: 1 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.3 }}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          proOptions={{ hideAttribution: false }}
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} color="#e8e8e8" />
          <Controls showInteractive={false} />
          <MiniMap
            nodeColor={(node) => MINIMAP_COLORS[node.type || ""] || "#d9d9d9"}
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
