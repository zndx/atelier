import { cn } from "./cn";
import type { Tone } from "./StatusDot";

const FILL: Record<Tone, string> = {
  green: "bg-status-green",
  amber: "bg-status-amber",
  red: "bg-status-red",
  neutral: "bg-gray-500",
  accent: "bg-accent",
};

// DESIGN_TEMPLATE §5.23 — inline progress bar. `indeterminate` renders
// a moving sliver for unknown-duration phases.
export function ProgressBar({
  value,
  max = 100,
  tone = "accent",
  indeterminate = false,
  className,
}: {
  value?: number;
  max?: number;
  tone?: Tone;
  indeterminate?: boolean;
  className?: string;
}) {
  const pct =
    !indeterminate && value !== undefined && max > 0
      ? Math.max(0, Math.min(100, (value / max) * 100))
      : 0;
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-3", className)}>
      {indeterminate ? (
        <div className={cn("h-full w-1/3 animate-pulse rounded-full", FILL[tone])} />
      ) : (
        <div
          className={cn("h-full rounded-full transition-all duration-500", FILL[tone])}
          style={{ width: `${pct}%` }}
        />
      )}
    </div>
  );
}
