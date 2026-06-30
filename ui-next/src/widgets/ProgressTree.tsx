import { Check, CircleDashed, Loader2, X } from "lucide-react";
import { buildTaskTree, type ProgressTask } from "../lib/buildTaskTree";
import type { FSMStatus } from "../api/types";
import { ProgressBar } from "../ui/ProgressBar";
import { cn } from "../ui/cn";
import type { Tone } from "../ui/StatusDot";

function statusTone(s: ProgressTask["status"]): Tone {
  switch (s) {
    case "done":
      return "green";
    case "error":
      return "red";
    case "active":
      return "accent";
    default:
      return "neutral";
  }
}

function StatusGlyph({ status }: { status: ProgressTask["status"] }) {
  switch (status) {
    case "done":
      return <Check className="h-3.5 w-3.5 text-status-green" />;
    case "error":
      return <X className="h-3.5 w-3.5 text-status-red" />;
    case "active":
      return <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />;
    default:
      return <CircleDashed className="h-3.5 w-3.5 text-gray-600" />;
  }
}

function fmtElapsed(s?: number): string | null {
  if (s === undefined) return null;
  if (s < 60) return `${Math.round(s)}s`;
  const m = Math.floor(s / 60);
  return `${m}m ${Math.round(s % 60)}s`;
}

function Row({ task }: { task: ProgressTask }) {
  const tone = statusTone(task.status);
  const det = task.determinate;
  const elapsed = fmtElapsed(task.elapsedS);
  return (
    <div
      className={cn("py-1.5", task.depth === 1 && "pl-4", task.depth === 2 && "pl-8")}
      style={task.depth === 0 ? undefined : undefined}
    >
      <div className="flex items-center gap-2">
        <StatusGlyph status={task.status} />
        <span
          className={cn(
            "text-sm",
            task.status === "pending" ? "text-gray-500" : "text-gray-200",
            task.depth === 0 && "font-semibold text-white",
          )}
        >
          {task.name}
        </span>
        <span className="ml-auto flex items-center gap-2 font-mono text-[11px] text-gray-500">
          {det && (
            <span>
              {det.current.toLocaleString()}/{det.total.toLocaleString()}
              {det.unit ? ` ${det.unit}` : ""}
            </span>
          )}
          {elapsed && <span>{elapsed}</span>}
        </span>
      </div>
      {(det || task.indeterminate) && task.status !== "pending" && (
        <div className="mt-1.5">
          <ProgressBar
            value={det?.current}
            max={det?.total}
            tone={tone}
            indeterminate={!det && !!task.indeterminate}
          />
        </div>
      )}
    </div>
  );
}

export function ProgressTree({
  fsm,
  showLineage = false,
}: {
  fsm: FSMStatus | null;
  showLineage?: boolean;
}) {
  const tasks = buildTaskTree(fsm, { showLineage });
  if (tasks.length === 0) {
    return <div className="text-sm text-gray-500">No active run.</div>;
  }
  return (
    <div className="divide-y divide-surface-3/60">
      {tasks.map((t) => (
        <Row key={t.id} task={t} />
      ))}
    </div>
  );
}
