// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

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
    bg: "#fff7e6",
    textColor: "#ad6800",
    iconNode: <ThunderboltOutlined style={{ color: "#fa8c16" }} />,
  },
  active: {
    border: "#fa8c16",
    bg: "#fff7e6",
    textColor: "#ad4e00",
    glow: "0 0 0 4px rgba(250, 140, 22, 0.20)",
    iconNode: <LoadingOutlined style={{ color: "#fa8c16" }} />,
  },
  unconfigured: {
    // Layout omits unconfigured skills entirely; theme retained as a
    // safety net so a misconfigured render path doesn't crash on
    // missing dictionary key.
    border: "#f0f0f0",
    bg: "#fafafa",
    textColor: "#bfbfbf",
    iconNode: <ThunderboltOutlined style={{ color: "#bfbfbf" }} />,
  },
};

export default function SkillNode({ data }: NodeProps<SkillNodeType>) {
  const theme = STATUS_THEME[data.status] || STATUS_THEME.available;

  return (
    <>
      <NodeToolbar position={Position.Top}>
        <div
          style={{
            background: "#fff",
            border: "1px solid #d9d9d9",
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
