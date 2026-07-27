import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { Tag } from "antd";
import { RobotOutlined } from "@ant-design/icons";
import type { OrchestratorNodeData } from "../../types/canvas";

type OrchestratorNodeType = Node<OrchestratorNodeData, "orchestrator">;

const statusColor: Record<string, string> = {
  idle: "var(--color-kumo-success)",
  orchestrating: "var(--color-kumo-brand)",
  completed: "var(--text-color-kumo-subtle)",
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
              background: "var(--color-kumo-brand)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 2px 4px rgba(0,0,0,0.2)",
            }}
          >
            <RobotOutlined style={{ fontSize: 20, color: "var(--text-color-kumo-inverse)" }} />
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
            background: statusColor[data.status] || "var(--color-kumo-success)",
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
        style={{ background: "var(--color-kumo-brand)" }}
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
