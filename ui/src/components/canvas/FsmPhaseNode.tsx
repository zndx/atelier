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
    border: "#d9d9d9",
    bg: "#fafafa",
    textColor: "#8c8c8c",
  },
  upcoming: {
    border: "#d9d9d9",
    bg: "#fff",
    textColor: "#595959",
  },
  current: {
    border: "#1677ff",
    bg: "#e6f4ff",
    textColor: "#0958d9",
    glow: "0 0 0 4px rgba(22, 119, 255, 0.15)",
    iconNode: <LoadingOutlined style={{ color: "#1677ff" }} />,
  },
  completed: {
    border: "#b7eb8f",
    bg: "#f6ffed",
    textColor: "#389e0d",
    iconNode: <CheckCircleFilled style={{ color: "#52c41a" }} />,
  },
  converged: {
    border: "#52c41a",
    bg: "#f6ffed",
    textColor: "#237804",
    glow: "0 0 0 4px rgba(82, 196, 26, 0.18)",
    iconNode: <CheckCircleFilled style={{ color: "#52c41a" }} />,
  },
  error: {
    border: "#ff4d4f",
    bg: "#fff1f0",
    textColor: "#cf1322",
    glow: "0 0 0 4px rgba(255, 77, 79, 0.18)",
    iconNode: <CloseCircleFilled style={{ color: "#ff4d4f" }} />,
  },
};

export default function FsmPhaseNode({ data }: NodeProps<FsmPhaseNodeType>) {
  const theme = STATUS_THEME[data.status] || STATUS_THEME.upcoming;

  return (
    <>
      <NodeToolbar position={Position.Top}>
        <div
          style={{
            background: "#fff",
            border: "1px solid #d9d9d9",
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
