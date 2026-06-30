import type { ButtonHTMLAttributes, ReactNode } from "react";
import { cn } from "./cn";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  icon?: ReactNode;
}

// DESIGN_TEMPLATE §5.5 — buttons.
const VARIANTS: Record<Variant, string> = {
  primary: "bg-accent hover:bg-accent-dim text-white",
  secondary: "bg-surface-3 hover:bg-surface-4 text-gray-200 border border-surface-4",
  ghost: "text-gray-400 hover:text-gray-200 hover:bg-surface-3",
  danger: "bg-status-red/90 hover:bg-status-red text-white",
};

const SIZES: Record<Size, string> = {
  sm: "px-2.5 py-1 text-xs",
  md: "px-3 py-1.5 text-sm",
};

export function Button({
  variant = "secondary",
  size = "md",
  icon,
  className,
  children,
  ...rest
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex items-center justify-center gap-1.5 rounded-md font-medium",
        "transition-colors focus:outline-none focus:ring-2 focus:ring-accent/40",
        "disabled:opacity-50 disabled:cursor-not-allowed",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...rest}
    >
      {icon}
      {children}
    </button>
  );
}
