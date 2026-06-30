import type { Tone } from "../ui/StatusDot";

export const FSM_LABELS: Record<string, string> = {
  IDLE: "Idle",
  LOADING_VOCAB: "Loading vocabulary",
  DISCOVERING: "Discovering tables",
  SAMPLING: "Sampling",
  LLM_SWEEP: "LLM sweep",
  VALIDATING: "ML validation",
  CLASSIFYING: "Classifying",
  FUSING: "Fusing",
  EVALUATING: "Evaluating",
  CONVERGED: "Converged",
  ERROR: "Error",
};

const TERMINAL = new Set(["IDLE", "CONVERGED", "ERROR"]);

export function isRunning(state: string | null | undefined): boolean {
  return !!state && !TERMINAL.has(state);
}

export function stateTone(state: string | null | undefined): Tone {
  if (!state || state === "IDLE") return "neutral";
  if (state === "ERROR") return "red";
  if (state === "CONVERGED") return "green";
  return "accent";
}

export function fsmLabel(state: string | null | undefined): string {
  if (!state) return "Idle";
  return FSM_LABELS[state] ?? state;
}
