import type { ReactNode } from "react";
import { cn } from "./cn";

interface CardProps {
  children: ReactNode;
  className?: string;
  padding?: "none" | "sm" | "md";
}

// DESIGN_TEMPLATE §5.1 — bg-surface-2, rounded-lg, border-surface-3.
export function Card({ children, className, padding = "md" }: CardProps) {
  const pad = padding === "none" ? "" : padding === "sm" ? "p-3" : "p-4";
  return (
    <div className={cn("bg-surface-2 rounded-lg border border-surface-3", pad, className)}>
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: ReactNode;
  subtitle?: ReactNode;
  icon?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

// DESIGN_TEMPLATE §5.2 — card header strip.
export function CardHeader({ title, subtitle, icon, actions, className }: CardHeaderProps) {
  return (
    <div className={cn("flex items-center justify-between gap-3", className)}>
      <div className="flex items-center gap-2.5 min-w-0">
        {icon && <span className="text-accent shrink-0">{icon}</span>}
        <div className="min-w-0">
          <div className="text-sm font-semibold text-white truncate">{title}</div>
          {subtitle && <div className="text-xs text-gray-500 truncate">{subtitle}</div>}
        </div>
      </div>
      {actions && <div className="flex items-center gap-2 shrink-0">{actions}</div>}
    </div>
  );
}
