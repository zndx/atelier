// Typed shapes for the Atelier gateway contract (src/atelier/gateway.py).
// These mirror the JSON the REST endpoints return. The gateway wraps
// failures as { error: string } with a non-2xx status; see client.ts.

export interface DataSourceInfo {
  id: string;
  source_type: string;
  source_uri: string;
  display_name: string;
  vocabulary_mode: string;
  created_at: string;
  metadata: string;
  is_archived?: boolean;
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  parquet_path: string;
  row_count: number;
  source_id: string;
  version_number: number;
  is_active: boolean;
  summary: string;
  fsm_run_id: string;
  created_at: string;
  artifact_set_id?: string | null;
  parent_dataset_id?: string | null;
  run_kind?: string;
  is_archived?: boolean;
}

export interface MLArtifactSet {
  id: string;
  source_id: string | null;
  fsm_run_id: string | null;
  parent_artifact_set_id: string | null;
  catboost_path: string;
  catboost_classes_path: string;
  svm_path: string | null;
  svm_classes_path: string | null;
  umap_path: string | null;
  classes: string;
  feature_groups: string | null;
  vocab_signature: string;
  embedding_model: string;
  embedding_dim: number;
  display_name: string | null;
  summary: string | null;
  is_active: boolean;
  is_archived: boolean;
  facets: string | null;
  created_at: string;
}

export interface SmokeTestResult {
  success: boolean;
  reply?: string;
  duration_ms?: number;
  session_id?: string;
  total_cost_usd?: number;
  retried?: boolean;
  error?: string;
}

// GET /api/status — aggregated health + safe config flags.
export interface SystemStatusConfig {
  has_anthropic?: boolean;
  has_bedrock?: boolean;
  has_classify_llm?: boolean;
  agent_model?: string;
  qdrant_host?: string;
  db_url_masked?: string;
  model_discovery?: boolean;
  cautious_review_enabled?: boolean;
  overwatch_enabled?: boolean;
  classify_agent_enabled?: boolean;
}

export interface ProbeResult {
  ok: boolean;
  version?: string;
  latency_ms?: number | null;
}

export interface SystemStatus {
  grpc: ProbeResult;
  postgres: ProbeResult;
  qdrant: ProbeResult;
  config: SystemStatusConfig;
  connected: boolean;
  degraded: boolean;
}

// GET /api/fsm/status
export interface FSMStatus {
  state: string;
  progress?: Record<string, unknown>;
  error?: string | null;
}

export interface FSMStartResult {
  started: boolean;
  source_id?: string;
  run_kind?: string;
  error?: string;
  queue_state?: unknown;
}

export interface FSMRun {
  run_id: string;
  state?: string;
  source_id?: string;
  run_kind?: string;
  started_at?: string;
  ended_at?: string | null;
  [k: string]: unknown;
}

// GET /api/settings
export interface SettingMetadata {
  label?: string;
  description?: string;
  type?: string;
  choices?: string[];
  min?: number;
  max?: number;
  step?: number;
  group?: string;
  caption?: string;
  [k: string]: unknown;
}

export interface SettingsPayload {
  metadata: Record<string, SettingMetadata>;
  values: Record<string, unknown>;
  overlay_keys: string[];
}

// GET /api/acceleration
export interface AccelerationStatus {
  available: boolean;
  device_count?: number;
  device_name?: string;
  methods?: Record<string, boolean>;
  config?: Record<string, unknown>;
  [k: string]: unknown;
}

// GET /api/agents
export interface AgentInfo {
  id: string;
  name: string;
  description: string;
  role: string;
  tool_ids: string[];
}

// GET /api/skills
export interface SkillInfo {
  id: string;
  title: string;
  description: string;
  content?: string;
}

// GET /api/data-platforms
export interface DataPlatform {
  id: string;
  kind: "hive" | "filesystem" | string;
  label: string;
  source_uri?: string;
  vocab_uri?: string;
  mount?: string;
  table_count?: number;
  column_count?: number;
}

// GET /api/terminal/models
export interface TerminalModel {
  id: string;
  label?: string;
  name?: string;
  available?: boolean;
  provider?: string;
  stats?: Record<string, unknown>;
  [k: string]: unknown;
}

export interface TerminalModelsPayload {
  models: TerminalModel[];
  active?: string | null;
  override_set?: boolean;
}

// GET /api/overwatch/status
export interface OverwatchStatus {
  enabled?: boolean;
  ready?: boolean;
  model?: string;
  [k: string]: unknown;
}
