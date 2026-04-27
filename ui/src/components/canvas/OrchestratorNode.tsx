// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { Tag } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import type { OrchestratorNodeData } from "../../types/canvas";

type OrchestratorNodeType = Node<OrchestratorNodeData, "orchestrator">;

const statusColor: Record<string, string> = {
  idle: "#52c41a",
  orchestrating: "#1890ff",
  completed: "#8c8c8c",
};

export default function OrchestratorNode({ data }: NodeProps<OrchestratorNodeType>) {
  const isActive = data.status === "orchestrating";

  return (
    <>
      <div
        style={{
          background: "#001529",
          borderRadius: 8,
          padding: "14px 20px",
          minWidth: 220,
          color: "#fff",
          boxShadow: "0 2px 8px rgba(0,0,0,0.25)",
          position: "relative",
        }}
      >
        <div style={{ position: "absolute", top: -18, left: -18 }}>
          <div
            style={{
              width: 40,
              height: 40,
              borderRadius: "50%",
              background: "#1890ff",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
            }}
          >
            <RobotOutlined style={{ fontSize: 20, color: "#fff" }} />
          </div>
        </div>

        <div
          style={{
            position: "absolute",
            top: 8,
            right: 10,
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: statusColor[data.status] || "#52c41a",
            animation: isActive ? "pulse-status 1.5s ease-in-out infinite" : undefined,
          }}
        />

        <div style={{ marginLeft: 24, fontWeight: 600, fontSize: 14 }}>
          {data.label}
        </div>
        <div style={{ marginTop: 6 }}>
          <Tag color="geekblue" style={{ fontSize: 11 }}>
            {data.model}
          </Tag>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        style={{ background: "#1890ff" }}
      />

      <style>{`
        @keyframes pulse-status {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </>
  );
}
