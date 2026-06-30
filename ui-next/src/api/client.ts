// Central typed client over the gateway REST surface.
//
// The gateway returns success as a plain JSON dict and failures as
// { error: string } with a non-2xx status (gateway.py `_error_envelope`).
// `request` normalizes both: it throws ApiError on a non-2xx status or
// when the body carries an `error` key, otherwise returns the parsed body.

import type {
  AccelerationStatus,
  AgentInfo,
  DataPlatform,
  DataSourceInfo,
  DatasetInfo,
  FSMRun,
  FSMStartResult,
  FSMStatus,
  MLArtifactSet,
  OverwatchStatus,
  SettingsPayload,
  SkillInfo,
  SmokeTestResult,
  SystemStatus,
  TerminalModelsPayload,
} from "./types";

export class ApiError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

interface RequestOpts {
  method?: string;
  body?: unknown;
  // When true, return the raw Response instead of parsed JSON
  // (used by callers that stream binary, e.g. parquet).
  raw?: boolean;
}

export async function request<T>(path: string, opts: RequestOpts = {}): Promise<T> {
  const init: RequestInit = { method: opts.method ?? "GET" };
  if (opts.body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, init);
  if (opts.raw) return resp as unknown as T;

  let parsed: unknown = null;
  const text = await resp.text();
  if (text) {
    try {
      parsed = JSON.parse(text);
    } catch {
      parsed = text;
    }
  }
  const errKey =
    parsed && typeof parsed === "object" && "error" in parsed
      ? (parsed as { error?: unknown }).error
      : undefined;
  if (!resp.ok || errKey) {
    const msg =
      typeof errKey === "string"
        ? errKey
        : `Request to ${path} failed (HTTP ${resp.status})`;
    throw new ApiError(msg, resp.status, parsed);
  }
  return parsed as T;
}

// ── Health / status ─────────────────────────────────────────────
export const getStatus = () => request<SystemStatus>("/api/status");
export const getHealth = () => request<{ status: string; version?: string }>("/api/health");
export const getAcceleration = () => request<AccelerationStatus>("/api/acceleration");

// ── Agents / skills ─────────────────────────────────────────────
export const listAgents = () => request<{ agents: AgentInfo[] }>("/api/agents");
export const listSkills = () => request<{ skills: SkillInfo[] }>("/api/skills");
export const validateCredentials = () =>
  request<{ any_valid: boolean; providers?: unknown; configured?: unknown; error?: string }>(
    "/api/agents/validate-credentials",
    { method: "POST" },
  );
export const smokeTest = () => request<SmokeTestResult>("/api/agents/smoke-test", { method: "POST" });

// ── Data sources / datasets / artifacts ─────────────────────────
export const listDataSources = (includeArchived = false) =>
  request<{ sources: DataSourceInfo[] }>(
    `/api/data-sources?include_archived=${includeArchived}`,
  );
export const listDatasets = (sourceId?: string | null, includeArchived = false) => {
  const params = new URLSearchParams();
  if (sourceId) params.set("source_id", sourceId);
  params.set("include_archived", String(includeArchived));
  return request<{ datasets: DatasetInfo[] }>(`/api/datasets?${params.toString()}`);
};
export const activateDataset = (id: string) =>
  request<{ ok: boolean }>(`/api/datasets/${encodeURIComponent(id)}/activate`, { method: "POST" });
export const listArtifactSets = () =>
  request<{ artifact_sets: MLArtifactSet[] }>("/api/artifact-sets");
export const activateArtifactSet = (id: string) =>
  request<{ ok: boolean }>(`/api/artifact-sets/${encodeURIComponent(id)}/activate`, {
    method: "POST",
  });
export const listDataPlatforms = () =>
  request<{ platforms: DataPlatform[] }>("/api/data-platforms");
export const getVocabularyStats = (sourceId?: string | null) => {
  const q = sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : "";
  return request<{ terms: number; source: string }>(`/api/vocabulary/stats${q}`);
};

// Raw parquet bytes for the embeddings atlas.
export const fetchDatasetParquet = (id: string): Promise<Response> =>
  request<Response>(`/api/datasets/${encodeURIComponent(id)}/data`, { raw: true });

// ── FSM lifecycle ───────────────────────────────────────────────
export const getFsmStatus = () => request<FSMStatus>("/api/fsm/status");
export const startFsm = (sourceId?: string | null) => {
  const q = sourceId ? `?source_id=${encodeURIComponent(sourceId)}` : "";
  return request<FSMStartResult>(`/api/fsm/start${q}`, { method: "POST" });
};
export const extendFsm = (body: {
  source_id: string;
  artifact_set_id: string;
  parent_dataset_id?: string;
}) => request<FSMStartResult>("/api/fsm/extend", { method: "POST", body });
export const cancelFsm = (reason?: string) =>
  request<{ cancelled: boolean; state?: string; reason?: string }>("/api/fsm/cancel", {
    method: "POST",
    body: { reason },
  });
export const listFsmRuns = () => request<{ runs: FSMRun[] }>("/api/fsm/runs");

// ── Settings ────────────────────────────────────────────────────
export const getSettings = () => request<SettingsPayload>("/api/settings");
export const patchSettings = (values: Record<string, unknown>) =>
  request<{ ok: boolean; overlay: Record<string, unknown> }>("/api/settings", {
    method: "PATCH",
    body: values,
  });
export const resetSettings = () =>
  request<{ ok: boolean }>("/api/settings/reset", { method: "POST" });

// ── Terminal models ─────────────────────────────────────────────
export const listTerminalModels = () => request<TerminalModelsPayload>("/api/terminal/models");
export const setTerminalModel = (id: string) =>
  request<{ active: string; override_set: boolean }>("/api/terminal/models/active", {
    method: "POST",
    body: { id },
  });
export const clearTerminalModel = () =>
  request<{ active: string; override_set: boolean }>("/api/terminal/models/active", {
    method: "DELETE",
  });

// ── Overwatch ───────────────────────────────────────────────────
export const getOverwatchStatus = () => request<OverwatchStatus>("/api/overwatch/status");
export const getOverwatchReport = (runId: string) =>
  request<{ ok: boolean; run_id: string; report: string }>(
    `/api/overwatch/report/${encodeURIComponent(runId)}`,
  );
