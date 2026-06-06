import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { Tag, Typography } from "antd";
import { ThunderboltOutlined } from "@ant-design/icons";
import type { DynamicAgentNodeData } from "../../types/canvas";

const { Text } = Typography;

type DynamicAgentNodeType = Node<DynamicAgentNodeData, "dynamicAgent">;

const STATUS_COLOR: Record<string, string> = {
  spawning: "#faad14",
  active: "#1890ff",
  completed: "#52c41a",
  dismissed: "#8c8c8c",
};

export default function DynamicAgentNode({ data }: NodeProps<DynamicAgentNodeType>) {
  const isDismissed = data.status === "dismissed";

  return (
    <>
      <div
        style={{
          background: "#fafafa",
          borderRadius: 8,
          border: "2px dashed #bfbfbf",
          padding: "12px 16px",
          minWidth: 180,
          opacity: isDismissed ? 0.5 : 0.9,
          animation: data.status === "spawning" ? "fade-in-agent 0.4s ease-out" : undefined,
          position: "relative",
        }}
      >
        <div style={{ position: "absolute", top: -14, left: -14 }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              background: "#fff",
              border: `2px dashed ${STATUS_COLOR[data.status] || "#bfbfbf"}`,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 14,
              color: STATUS_COLOR[data.status] || "#bfbfbf",
            }}
          >
            <ThunderboltOutlined />
          </div>
        </div>

        <div style={{ marginLeft: 18, fontWeight: 600, fontSize: 12, color: "#434343" }}>
          {data.name}
        </div>
        <div style={{ marginTop: 4, marginLeft: 18 }}>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {data.purpose}
          </Text>
        </div>
        <div style={{ marginTop: 6 }}>
          <Tag
            color={STATUS_COLOR[data.status]}
            style={{ fontSize: 10, borderStyle: "dashed" }}
          >
            {data.status}
          </Tag>
        </div>
      </div>

      <Handle type="target" position={Position.Top} style={{ background: "#bfbfbf" }} />
      <Handle type="source" position={Position.Bottom} style={{ background: "#bfbfbf" }} />

      <style>{`
        @keyframes fade-in-agent {
          from { opacity: 0; transform: translateY(-8px); }
          to { opacity: 0.9; transform: translateY(0); }
        }
      `}</style>
    </>
  );
}
