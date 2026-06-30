import type { ReactNode } from "react";
import { cn } from "./cn";
import type { Tone } from "./StatusDot";

// DESIGN_TEMPLATE §5.9 — severity badge / status pill. Bright text on a
// dim tint of the same hue.
const PILL: Record<Tone, string> = {
  green: "bg-status-green/15 text-status-green border-status-green/30",
  amber: "bg-status-amber/15 text-status-amber border-status-amber/30",
  red: "bg-status-red/15 text-status-red border-status-red/30",
  neutral: "bg-surface-3 text-gray-300 border-surface-4",
  accent: "bg-accent/15 text-accent border-accent/30",
};

export function Pill({
  tone = "neutral",
  children,
  className,
  mono = false,
}: {
  tone?: Tone;
  children: ReactNode;
  className?: string;
  mono?: boolean;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs font-medium",
        mono && "font-mono",
        PILL[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}
