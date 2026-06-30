import { cn } from "./cn";

export type Tone = "green" | "amber" | "red" | "neutral" | "accent";

const DOT: Record<Tone, string> = {
  green: "bg-status-green",
  amber: "bg-status-amber",
  red: "bg-status-red",
  neutral: "bg-gray-500",
  accent: "bg-accent",
};

// DESIGN_TEMPLATE §5.8 — status indicator dot. `pulse` for live state.
export function StatusDot({
  tone,
  pulse = false,
  label,
  className,
}: {
  tone: Tone;
  pulse?: boolean;
  label?: string;
  className?: string;
}) {
  return (
    <span className={cn("inline-flex items-center gap-1.5", className)}>
      <span className="relative inline-flex h-2 w-2">
        {pulse && (
          <span
            className={cn("absolute inline-flex h-full w-full rounded-full opacity-60 animate-ping", DOT[tone])}
          />
        )}
        <span className={cn("relative inline-flex h-2 w-2 rounded-full", DOT[tone])} />
      </span>
      {label && <span className="text-xs text-gray-400">{label}</span>}
    </span>
  );
}
