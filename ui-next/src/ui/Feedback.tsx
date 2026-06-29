import type { ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "./cn";

// DESIGN_TEMPLATE §5.15 — loading.
export function Spinner({ className, label }: { className?: string; label?: string }) {
  return (
    <div className={cn("flex flex-col items-center justify-center gap-3 text-gray-500", className)}>
      <Loader2 className="h-6 w-6 animate-spin text-accent" />
      {label && <span className="text-sm">{label}</span>}
    </div>
  );
}

// DESIGN_TEMPLATE §5.16 — empty state.
export function EmptyState({
  icon,
  title,
  description,
  action,
  className,
}: {
  icon?: ReactNode;
  title: string;
  description?: ReactNode;
  action?: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-surface-3 bg-surface-1/40 px-6 py-12 text-center",
        className,
      )}
    >
      {icon && <div className="text-gray-600 [&>svg]:h-8 [&>svg]:w-8">{icon}</div>}
      <div className="text-sm font-medium text-gray-300">{title}</div>
      {description && <div className="max-w-sm text-xs text-gray-500">{description}</div>}
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
}

// Inline error/info banner (DESIGN_TEMPLATE §5.10 alert).
export function Banner({
  tone = "info",
  children,
  className,
}: {
  tone?: "info" | "error" | "warning";
  children: ReactNode;
  className?: string;
}) {
  const tones = {
    info: "bg-accent/10 border-accent/30 text-gray-200",
    error: "bg-status-red/10 border-status-red/30 text-gray-200",
    warning: "bg-status-amber/10 border-status-amber/30 text-gray-200",
  };
  return (
    <div className={cn("rounded-lg border px-4 py-3 text-sm", tones[tone], className)}>
      {children}
    </div>
  );
}
