import clsx, { type ClassValue } from "clsx";

// Tiny class-name combiner. (No tailwind-merge — we don't author
// conflicting utilities, so clsx alone is enough.)
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs);
}
