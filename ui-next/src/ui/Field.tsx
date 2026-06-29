import type { InputHTMLAttributes, ReactNode, SelectHTMLAttributes } from "react";
import { cn } from "./cn";

// DESIGN_TEMPLATE §5.6 — form inputs (dark).
const BASE =
  "w-full rounded-md bg-surface-1 border border-surface-4 px-3 py-1.5 text-sm text-gray-200 " +
  "placeholder:text-gray-600 focus:outline-none focus:border-accent focus:ring-1 focus:ring-accent/40 " +
  "disabled:opacity-50";

export function FieldLabel({
  label,
  hint,
  children,
}: {
  label: ReactNode;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <label className="block space-y-1">
      <span className="block text-xs font-medium text-gray-400">{label}</span>
      {children}
      {hint && <span className="block text-[11px] text-gray-600">{hint}</span>}
    </label>
  );
}

export function TextInput({ className, ...rest }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(BASE, className)} {...rest} />;
}

export function Select({
  className,
  children,
  ...rest
}: SelectHTMLAttributes<HTMLSelectElement> & { children: ReactNode }) {
  return (
    <select className={cn(BASE, "appearance-none pr-8", className)} {...rest}>
      {children}
    </select>
  );
}

export function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors",
        "focus:outline-none focus:ring-2 focus:ring-accent/40 disabled:opacity-50",
        checked ? "bg-accent" : "bg-surface-4",
      )}
    >
      <span
        className={cn(
          "inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
          checked ? "translate-x-4" : "translate-x-0.5",
        )}
      />
    </button>
  );
}
