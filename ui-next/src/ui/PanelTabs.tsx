import type { ReactNode } from "react";
import { cn } from "./cn";

export interface PanelTab<T extends string = string> {
  id: T;
  label: string;
  icon?: ReactNode;
}

// DESIGN_TEMPLATE §3 — panel-tab pattern. The most reused Layout-B
// primitive: right-panel switcher, filter bar, section selector.
export function PanelTabs<T extends string>({
  tabs,
  active,
  onChange,
  className,
}: {
  tabs: PanelTab<T>[];
  active: T;
  onChange: (id: T) => void;
  className?: string;
}) {
  return (
    <div className={cn("flex flex-wrap items-center gap-1", className)}>
      {tabs.map((t) => {
        const isActive = t.id === active;
        return (
          <button
            key={t.id}
            onClick={() => onChange(t.id)}
            className={cn(
              "flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
              isActive
                ? "bg-accent/20 text-accent"
                : "text-gray-400 hover:bg-surface-3 hover:text-gray-200",
            )}
          >
            {t.icon && <span className="[&>svg]:h-3.5 [&>svg]:w-3.5">{t.icon}</span>}
            {t.label}
          </button>
        );
      })}
    </div>
  );
}
