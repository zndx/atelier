import { Handle, Position, NodeToolbar, type NodeProps, type Node } from "@xyflow/react";
import { Tag, Typography } from "antd";
import {
  ExperimentOutlined,
  MergeCellsOutlined,
  DotChartOutlined,
} from "@ant-design/icons";
import type { KeystoneNodeData } from "../../types/canvas";
import type { ReactNode } from "react";

const { Text } = Typography;

type KeystoneNodeType = Node<KeystoneNodeData, "keystone">;

interface RoleTheme {
  icon: ReactNode;
  color: string;
  bg: string;
}

const ROLE_THEME: Record<string, RoleTheme> = {
  classifier: {
    icon: <ExperimentOutlined />,
    color: "var(--color-kumo-brand)",
    bg: "var(--color-kumo-info-tint)",
  },
  evidence_fuser: {
    icon: <MergeCellsOutlined />,
    color: "var(--color-kumo-success)",
    bg: "var(--color-kumo-success-tint)",
  },
  visualization_director: {
    icon: <DotChartOutlined />,
    color: "#722ed1",
    bg: "#f9f0ff",
  },
};

const STATUS_COLOR: Record<string, string> = {
  idle: "var(--color-kumo-line)",
  active: "var(--color-kumo-brand)",
  completed: "var(--color-kumo-success)",
  error: "var(--color-kumo-danger)",
};

export default function KeystoneNode({ data }: NodeProps<KeystoneNodeType>) {
  const theme = ROLE_THEME[data.role] || ROLE_THEME.classifier;

  return (
    <>
      <NodeToolbar position={Position.Top}>
        <div
          style={{
            background: "var(--color-kumo-elevated)",
            border: "1px solid var(--color-kumo-line)",
            borderRadius: 6,
            padding: "8px 12px",
            maxWidth: 240,
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          }}
        >
          <Text type="secondary" style={{ fontSize: 12 }}>
            {data.description}
          </Text>
        </div>
      </NodeToolbar>

      <div
        style={{
          background: theme.bg,
          borderRadius: 8,
          borderLeft: `4px solid ${theme.color}`,
          padding: "12px 16px",
          minWidth: 200,
          boxShadow: "0 1px 4px rgba(0,0,0,0.08)",
          position: "relative",
        }}
      >
        <div style={{ position: "absolute", top: -16, left: -16 }}>
          <div
            style={{
              width: 36,
              height: 36,
              borderRadius: "50%",
              background: "var(--color-kumo-elevated)",
              border: `2px solid ${theme.color}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 1px 3px rgba(0,0,0,0.12)",
              fontSize: 16,
              color: theme.color,
            }}
          >
            {theme.icon}
          </div>
        </div>

        <div
          style={{
            position: "absolute",
            top: 8,
            right: 8,
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: STATUS_COLOR[data.status] || "var(--color-kumo-line)",
            animation: data.status === "active" ? "pulse-keystone 1.5s ease-in-out infinite" : undefined,
          }}
        />

        <div style={{ marginLeft: 20, fontWeight: 600, fontSize: 13, color: "var(--text-color-kumo-strong)" }}>
          {data.name}
        </div>
        <div style={{ marginTop: 6 }}>
          <Tag
            style={{
              fontSize: 10,
              borderColor: theme.color,
              color: theme.color,
              background: "transparent",
            }}
          >
            {data.role.replace(/_/g, " ")}
          </Tag>
        </div>
      </div>

      <Handle type="target" position={Position.Top} id="top" style={{ background: "var(--color-kumo-line)" }} />
      <Handle type="source" position={Position.Bottom} id="bottom" style={{ background: "var(--color-kumo-line)" }} />
      <Handle type="source" position={Position.Right} id="right" style={{ background: theme.color, top: "50%" }} />
      <Handle type="target" position={Position.Left} id="left" style={{ background: theme.color, top: "50%" }} />
      <Handle type="target" position={Position.Right} id="right-target" style={{ background: theme.color, top: "50%" }} />
      <Handle type="source" position={Position.Left} id="left-source" style={{ background: theme.color, top: "50%" }} />

      <style>{`
        @keyframes pulse-keystone {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </>
  );
}
