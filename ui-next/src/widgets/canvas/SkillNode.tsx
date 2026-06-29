import { Handle, Position, type NodeProps } from "@xyflow/react";
import { Sparkles } from "lucide-react";
import { cn } from "../../ui/cn";
import type { SkillData } from "../../lib/canvasTypes";

const HANDLE = "!h-1.5 !w-1.5 !border-0 !bg-surface-4";

// Skills are AI-meta surfaces — the template reserves purple for that
// role, so skill nodes use purple rather than the indigo accent.
export default function SkillNode({ data }: NodeProps) {
  const d = data as SkillData;
  const active = d.status === "active";
  return (
    <div
      title={d.description}
      className={cn(
        "min-w-[110px] rounded-md border px-2.5 py-1.5 text-xs transition-colors",
        active
          ? "border-purple-400/60 bg-purple-500/15 text-purple-200 shadow-[0_0_14px_rgba(168,85,247,0.3)]"
          : "border-purple-500/25 bg-purple-500/5 text-purple-300/70",
      )}
    >
      <Handle type="target" position={Position.Top} className={HANDLE} />
      <div className="flex items-center gap-1.5 font-medium">
        <Sparkles className="h-3 w-3" />
        {d.label}
      </div>
      {d.model && <div className="mt-0.5 text-[9px] opacity-70">{d.model}</div>}
      <Handle type="source" position={Position.Bottom} className={HANDLE} />
    </div>
  );
}
