import { Handle, Position, type NodeProps, type Node } from "@xyflow/react";
import { Tag } from "antd";
import {
  TableOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import type { ArtifactNodeData } from "../../types/canvas";
import type { ReactNode } from "react";

type ArtifactNodeType = Node<ArtifactNodeData, "artifact">;

const TYPE_CONFIG: Record<string, { icon: ReactNode; color: string }> = {
  dataset: { icon: <TableOutlined />, color: "var(--color-kumo-brand)" },
  parquet: { icon: <TableOutlined />, color: "#722ed1" },
  classification_result: { icon: <BarChartOutlined />, color: "var(--color-kumo-success)" },
  evidence_vector: { icon: <BarChartOutlined />, color: "var(--color-kumo-warning)" },
  embedding: { icon: <DotIcon />, color: "var(--color-kumo-info)" },
};

function DotIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14">
      <circle cx="7" cy="7" r="3" fill="currentColor" />
    </svg>
  );
}

const STATUS_BG: Record<string, string> = {
  pending: "var(--color-kumo-recessed)",
  ready: "var(--color-kumo-success-tint)",
  consumed: "var(--color-kumo-fill)",
};

export default function ArtifactNode({ data }: NodeProps<ArtifactNodeType>) {
  const config = TYPE_CONFIG[data.type] || TYPE_CONFIG.dataset;

  return (
    <>
      <div
        style={{
          background: STATUS_BG[data.status] || "var(--color-kumo-recessed)",
          borderRadius: 6,
          borderLeft: `3px solid ${config.color}`,
          padding: "8px 12px",
          minWidth: 140,
          maxWidth: 180,
          boxShadow: "0 1px 3px rgba(0,0,0,0.06)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ color: config.color, fontSize: 13 }}>{config.icon}</span>
          <span style={{ fontWeight: 500, fontSize: 11, color: "var(--text-color-kumo-default)" }}>
            {data.name}
          </span>
        </div>
        {data.rowCount != null && (
          <div style={{ marginTop: 4 }}>
            <Tag style={{ fontSize: 9, padding: "0 4px" }}>
              {data.rowCount.toLocaleString()} rows
            </Tag>
          </div>
        )}
      </div>

      <Handle type="target" position={Position.Left} style={{ background: config.color }} />
      <Handle type="source" position={Position.Right} style={{ background: config.color }} />
    </>
  );
}
