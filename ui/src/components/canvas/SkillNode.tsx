import {
  Handle,
  Position,
  NodeToolbar,
  type NodeProps,
  type Node,
} from "@xyflow/react";
import { Typography } from "antd";
import { ThunderboltOutlined, LoadingOutlined } from "@ant-design/icons";
import type { SkillData, SkillStatus } from "../../types/canvas";

const { Text } = Typography;

type SkillNodeType = Node<SkillData, "skill">;

interface StatusTheme {
  border: string;
  bg: string;
  textColor: string;
  glow?: string;
  iconNode: React.ReactNode;
}

const STATUS_THEME: Record<SkillStatus, StatusTheme> = {
  available: {
    border: "#ffd591",
    bg: "var(--color-kumo-warning-tint)",
    textColor: "var(--text-color-kumo-warning)",
    iconNode: <ThunderboltOutlined style={{ color: "var(--color-kumo-warning)" }} />,
  },
  active: {
    border: "var(--color-kumo-warning)",
    bg: "var(--color-kumo-warning-tint)",
    textColor: "var(--color-kumo-warning)",
    glow: "0 0 0 4px rgba(217, 157, 84, 0.20)",
    iconNode: <LoadingOutlined style={{ color: "var(--color-kumo-warning)" }} />,
  },
  unconfigured: {
    // Layout omits unconfigured skills entirely; theme retained as a
    // safety net so a misconfigured render path doesn't crash on
    // missing dictionary key.
    border: "var(--color-kumo-line)",
    bg: "var(--color-kumo-recessed)",
    textColor: "var(--text-color-kumo-inactive)",
    iconNode: <ThunderboltOutlined style={{ color: "var(--text-color-kumo-inactive)" }} />,
  },
};

export default function SkillNode({ data }: NodeProps<SkillNodeType>) {
  const theme = STATUS_THEME[data.status] || STATUS_THEME.available;

  return (
    <>
      <NodeToolbar position={Position.Top}>
        <div
          style={{
            background: "var(--color-kumo-elevated)",
            border: "1px solid var(--color-kumo-line)",
            borderRadius: 6,
            padding: "8px 12px",
            maxWidth: 280,
            boxShadow: "0 2px 8px rgba(0,0,0,0.12)",
          }}
        >
          <Text strong style={{ fontSize: 12 }}>
            {data.label}
          </Text>
          {data.model && (
            <Text type="secondary" style={{ fontSize: 11, marginLeft: 6 }}>
              · {data.model}
            </Text>
          )}
          <div>
            <Text type="secondary" style={{ fontSize: 12 }}>
              {data.description}
            </Text>
          </div>
        </div>
      </NodeToolbar>

      <Handle type="target" position={Position.Bottom} style={{ opacity: 0 }} />
      <div
        style={{
          minWidth: 110,
          padding: "6px 10px",
          border: `1.5px dashed ${theme.border}`,
          background: theme.bg,
          color: theme.textColor,
          borderRadius: 14,
          boxShadow: theme.glow,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 11,
          fontWeight: 600,
          textAlign: "center",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          transition: "box-shadow 200ms ease",
        }}
      >
        {theme.iconNode}
        <span>{data.label}</span>
      </div>
    </>
  );
}
