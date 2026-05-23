// Depth-aware progress row for the nested pipeline UI.
//
// Renders one row of the buildTaskTree output: status icon + indent
// + name + (Progress | Spin) + elapsed [+ ETA].  Indentation is
// CSS-driven; no <Tree> widget.  See
// `.claude/plans/lovely-doodling-badger.md`.

import { Progress, Spin, Tag, Tooltip, Typography } from "antd";
import {
  CheckCircleTwoTone,
  CloseCircleTwoTone,
  ClockCircleOutlined,
  MinusCircleOutlined,
} from "@ant-design/icons";
import type { ProgressTask, TaskStatus } from "./buildTaskTree";

const { Text } = Typography;

function formatElapsed(s: number | undefined): string {
  if (s === undefined || s < 0) return "";
  const total = Math.round(s);
  const m = Math.floor(total / 60);
  const sec = total % 60;
  if (m === 0) return `${sec}s`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
}

function statusIcon(status: TaskStatus): React.ReactNode {
  switch (status) {
    case "done":
      return <CheckCircleTwoTone twoToneColor="#52c41a" />;
    case "error":
      return <CloseCircleTwoTone twoToneColor="#ff4d4f" />;
    case "active":
      return <Spin size="small" />;
    case "skipped":
      return <MinusCircleOutlined style={{ color: "#bfbfbf" }} />;
    case "pending":
    default:
      return <ClockCircleOutlined style={{ color: "#bfbfbf" }} />;
  }
}

interface PhaseProgressProps {
  task: ProgressTask;
}

export function PhaseProgress({ task }: PhaseProgressProps) {
  // Indent guide for depth-2 sub-phases.  Depth-1 (phase rows) sit
  // flush; depth-2 (sub-phase rows) gain 24px indent + a leading
  // "└─ " glyph rendered via CSS so the visual hierarchy reads
  // immediately as a tree.
  const indent = task.depth * 24;
  const guide = task.depth >= 2 ? "└─ " : "";

  const muted = task.status === "pending" || task.status === "skipped";
  const nameStyle: React.CSSProperties = {
    fontWeight: task.status === "active" ? 600 : 500,
    color: muted ? "rgba(0,0,0,0.45)" : undefined,
  };

  const elapsed = formatElapsed(task.elapsedS);
  const etaText =
    task.etaS !== undefined && task.etaS !== null && task.etaS > 0
      ? `ETA ${formatElapsed(task.etaS)}`
      : null;

  // Bar / spinner area.  Determinate when current/total available;
  // otherwise indeterminate spinner (inline, small).  Pending and
  // skipped rows show neither — just the icon + name.
  let progressEl: React.ReactNode = null;
  if (task.status === "active" || task.status === "error") {
    if (task.determinate) {
      const { current, total, unit } = task.determinate;
      const percent = total > 0 ? Math.round((current / total) * 100) : 0;
      progressEl = (
        <Progress
          percent={percent}
          format={() => `${current} / ${total}${unit ? ` ${unit}` : ""}`}
          status={task.status === "error" ? "exception" : "active"}
          size="small"
          style={{ flex: 1, marginLeft: 12, marginRight: 12 }}
        />
      );
    } else if (task.indeterminate) {
      progressEl = (
        <Text type="secondary" style={{ marginLeft: 12, fontStyle: "italic" }}>
          in progress…
        </Text>
      );
    }
  } else if (task.status === "done") {
    progressEl = (
      <Tag color="success" style={{ marginLeft: 12 }}>
        complete
      </Tag>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        paddingLeft: indent,
        paddingTop: 4,
        paddingBottom: 4,
      }}
    >
      <span style={{ width: 16, display: "inline-flex", justifyContent: "center" }}>
        {statusIcon(task.status)}
      </span>
      {guide && <Text type="secondary">{guide}</Text>}
      <Text style={nameStyle}>{task.name}</Text>
      {progressEl}
      {elapsed && (
        <Tooltip title={etaText ?? undefined}>
          <Text type="secondary" style={{ minWidth: 64, textAlign: "right" }}>
            {elapsed}
          </Text>
        </Tooltip>
      )}
    </div>
  );
}
