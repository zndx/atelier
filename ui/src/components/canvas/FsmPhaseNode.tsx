import {
  Handle,
  Position,
  NodeToolbar,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import { Typography } from "antd";
import {
  CheckCircleFilled,
  CloseCircleFilled,
  LoadingOutlined,
} from "@ant-design/icons";
import type { FsmPhaseData, FsmPhaseStatus } from "../../types/canvas";

const { Text } = Typography;

type FsmPhaseNodeType = Node<FsmPhaseData, "fsmPhase">;

interface StatusTheme {
  border: string;
  bg: string;
  textColor: string;
  glow?: string;
  iconNode?: React.ReactNode;
}

const STATUS_THEME: Record<FsmPhaseStatus, StatusTheme> = {
  idle: {
    border: "var(--color-kumo-line)",
    bg: "var(--color-kumo-recessed)",
    textColor: "var(--text-color-kumo-subtle)",
  },
  upcoming: {
    border: "var(--color-kumo-line)",
    bg: "var(--color-kumo-elevated)",
    textColor: "var(--text-color-kumo-default)",
  },
  current: {
    border: "var(--color-kumo-brand)",
    bg: "var(--color-kumo-info-tint)",
    textColor: "var(--text-color-kumo-link)",
    glow: "0 0 0 4px rgba(150, 162, 252, 0.15)",
    iconNode: <LoadingOutlined style={{ color: "var(--text-color-kumo-link)" }} />,
  },
  completed: {
    border: "#b7eb8f",
    bg: "var(--color-kumo-success-tint)",
    textColor: "var(--color-kumo-success)",
    iconNode: <CheckCircleFilled style={{ color: "var(--color-kumo-success)" }} />,
  },
  converged: {
    border: "var(--color-kumo-success)",
    bg: "var(--color-kumo-success-tint)",
    textColor: "var(--text-color-kumo-success)",
    glow: "0 0 0 4px rgba(78, 196, 145, 0.18)",
    iconNode: <CheckCircleFilled style={{ color: "var(--color-kumo-success)" }} />,
  },
  error: {
    border: "var(--color-kumo-danger)",
    bg: "var(--color-kumo-danger-tint)",
    textColor: "var(--text-color-kumo-danger)",
    glow: "0 0 0 4px rgba(242, 136, 129, 0.18)",
    iconNode: <CloseCircleFilled style={{ color: "var(--color-kumo-danger)" }} />,
  },
};

export default function FsmPhaseNode({ data }: NodeProps<FsmPhaseNodeType>) {
  const theme = STATUS_THEME[data.status] || STATUS_THEME.upcoming;

  return (
    <>
      <NodeToolbar position={Position.Top}>
        <div
          style={{
            background: "var(--color-kumo-elevated)",
            border: "1px solid var(--color-kumo-line)",
            borderRadius: 6,
            padding: "8px 12px",
            maxWidth: 260,
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          }}
        >
          <Text strong style={{ fontSize: 12 }}>
            {data.state}
          </Text>
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {data.description}
            </Text>
          </div>
        </div>
      </NodeToolbar>

      <Handle type="target" position={Position.Left} style={{ opacity: 0 }} />
      <div
        style={{
          minWidth: 130,
          padding: "10px 14px",
          border: `1.5px solid ${theme.border}`,
          background: theme.bg,
          color: theme.textColor,
          borderRadius: 8,
          boxShadow: theme.glow,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 12,
          fontWeight: 600,
          textAlign: "center",
          letterSpacing: "0.02em",
          transition: "box-shadow 200ms ease, background 200ms ease",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 6,
          }}
        >
          {theme.iconNode}
          <span>{data.label}</span>
        </div>
      </div>
      <Handle
        type="source"
        position={Position.Right}
        style={{ opacity: 0 }}
      />
    </>
  );
}
