import { Handle, Position, type NodeProps } from "@xyflow/react";
import { cn } from "../../ui/cn";
import type { FsmPhaseData, FsmPhaseStatus } from "../../lib/canvasTypes";

const STYLES: Record<FsmPhaseStatus, string> = {
  current: "border-accent bg-accent/15 text-white shadow-[0_0_18px_rgba(99,102,241,0.35)]",
  completed: "border-status-green/40 bg-status-green/10 text-gray-200",
  converged: "border-status-green bg-status-green/15 text-white",
  error: "border-status-red bg-status-red/15 text-white",
  upcoming: "border-surface-4 bg-surface-2 text-gray-500",
  idle: "border-surface-4 bg-surface-2 text-gray-400",
};

const HANDLE = "!h-1.5 !w-1.5 !border-0 !bg-surface-4";

export default function FsmPhaseNode({ data }: NodeProps) {
  const d = data as FsmPhaseData;
  const isCurrent = d.status === "current";
  return (
    <div
      title={d.description}
      className={cn(
        "min-w-[120px] rounded-lg border px-3 py-2 transition-colors",
        STYLES[d.status],
      )}
    >
      <Handle type="target" position={Position.Left} className={HANDLE} />
      <div className="flex items-center gap-1.5">
        {isCurrent && <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />}
        <span className="text-sm font-semibold">{d.label}</span>
      </div>
      <div className="mt-0.5 font-mono text-[9px] uppercase tracking-wide opacity-60">
        {d.state}
      </div>
      <Handle type="source" position={Position.Right} className={HANDLE} />
    </div>
  );
}
