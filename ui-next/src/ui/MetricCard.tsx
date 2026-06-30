import type { ReactNode } from "react";
import { cn } from "./cn";
import type { Tone } from "./StatusDot";

const VALUE_TONE: Record<Tone, string> = {
  green: "text-status-green",
  amber: "text-status-amber",
  red: "text-status-red",
  neutral: "text-white",
  accent: "text-accent",
};

// DESIGN_TEMPLATE §5.4 — small metric/KPI card. Numerals are mono.
export function MetricCard({
  label,
  value,
  unit,
  tone = "neutral",
  icon,
  className,
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: Tone;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("bg-surface-2 rounded-lg border border-surface-3 p-3", className)}>
      <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-gray-500">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      <div className={cn("mt-1 font-mono text-2xl font-bold leading-tight", VALUE_TONE[tone])}>
        {value}
        {unit && <span className="ml-1 text-sm font-medium text-gray-500">{unit}</span>}
      </div>
    </div>
  );
}
