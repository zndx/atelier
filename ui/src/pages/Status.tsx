// Copyright (c) 2026 Cloudera, Inc.  All rights reserved.
//
// This file contains material proprietary to Cloudera, Inc., and is provided
// to authorized licensees solely for use in connection with the Cloudera AI
// (CAI) Application from which it was obtained.  It may not be copied,
// modified, redistributed, or used in any other manner without the express
// written consent of Cloudera, Inc.

import { useCallback, useEffect, useRef, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  message,
  Popconfirm,
  Progress,
  Radio,
  Row,
  Select,
  Space,
  Spin,
  Switch,
  Table,
  Tag,
  Tooltip,
  Typography,
} from "antd";
import {
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  RocketOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Link } from "react-router-dom";
import { useDataset } from "../contexts/DatasetContext";

const { Title, Paragraph, Text } = Typography;

function formatAgo(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return `${Math.max(1, Math.floor(diff / 1000))}s ago`;
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

interface ServiceCheck {
  ok: boolean;
  version?: string;
  latency_ms?: number;
  error?: string;
}

interface ModelDiscovery {
  current_model: string;
  current_family: string | null;
  source: "bedrock_arn" | "direct";
  upgrade_available: boolean;
  latest_model?: string;
  latest_display_name?: string;
  latest_created_at?: string;
  reason?: string;
  cached: boolean;
}

interface ConfigInfo {
  has_anthropic: boolean;
  has_bedrock: boolean;
  has_classify_llm: boolean;
  agent_model: string;
  qdrant_host: string;
  qdrant_http_port: number;
  db_url_masked: string;
  model_discovery: ModelDiscovery | null;
}

interface StatusResponse {
  grpc: ServiceCheck;
  postgres: ServiceCheck;
  qdrant: ServiceCheck;
  config: ConfigInfo;
  connected: boolean;
}

// Mirrors bin/pglite-supervisor.sh state file + the gateway's
// /api/pglite/status response.  ``managed=false`` means pglite isn't
// the active backend (external Postgres or no supervisor) — UI hides
// the chip entirely in that case.
interface PGliteSupervisorState {
  managed: boolean;
  state?: "starting" | "running" | "backoff" | "circuit_broken" | "stopped";
  pid?: number | null;
  supervisor_pid?: number | null;
  started_at?: number | null;
  last_exit_code?: number | null;
  consecutive_failures?: number;
  restart_count?: number;
  backoff_until?: number | null;
  circuit_broken?: boolean;
  port?: number;
  updated_at?: number;
  probe?: { listening: boolean; error: string | null };
  reason?: string;
  error?: string;
}

interface ProviderResult {
  provider: string;
  valid: boolean;
  model?: string;
  error?: string;
  failure_kind?: "auth" | "permission" | "network" | "api" | "unknown";
}

interface CredentialResult {
  providers: Record<string, ProviderResult>;
  any_valid: boolean;
  configured: string[];
  error?: string;
}

interface SmokeResult {
  success: boolean;
  reply?: string;
  duration_ms?: number;
  session_id?: string;
  total_cost_usd?: number;
  retried?: boolean;
  error?: string;
}

interface DatabaseInfo {
  name: string;
  table_count: number;
  has_annotations: boolean;
  annotation_count: number;
  schema_format: string | null;
}

interface RefreshResult {
  ok: boolean;
  connection: string;
  latency_ms?: number;
  databases?: DatabaseInfo[];
  error?: string;
}

interface FSMStatus {
  id?: string;
  state: string;
  started_at?: string;
  updated_at?: string;
  progress?: Record<string, unknown>;
  config?: Record<string, unknown>;
  error?: string | null;
  result_path?: string | null;
  source_id?: string | null;
}

interface TerminalModelStats {
  n: number;
  ttft_ms_p50: number;
  tokps_p50: number;
}

interface TerminalModelEntry {
  id: string;
  label: string;
  provider: string;
  model_ref: string;
  context_window: number;
  max_output_tokens: number;
  thinking: string;
  available: boolean;
  unavailable_reason: string;
  notes: string;
  stats: TerminalModelStats;
}

interface TerminalModelsResponse {
  models: TerminalModelEntry[];
  active: TerminalModelEntry | null;
  override_set: boolean;
}

// Convergence-reason taxonomy.  Surfaces HOW a run ended alongside the
// CONVERGED state chip so an iter-1 early-exit cannot masquerade as
// legitimate belief-gap convergence.  Project directive — the pipeline
// is iterative by design; reasons marked suspect indicate that design
// wasn't exercised and the result deserves extra scrutiny.
const CONVERGENCE_REASON_DESCRIPTIONS: Record<string, string> = {
  no_revisit_candidates:
    "Revisit candidate set was empty at loop exit — legitimate only AFTER min_iterations.",
  gap_threshold_met:
    "Mean belief gap (Pl − Bel) fell below threshold — primary belief-gap convergence (honoring min_iterations floor).",
  plateau:
    "Belief gap stopped decreasing for 2+ iterations — converged by plateau detection.",
  budget_exhausted:
    "LLM call budget hit before convergence criteria were met.",
  max_iterations_reached:
    "Ran the full max_iterations without meeting belief-gap criteria.",
  coverage_and_gap_met:
    "Coverage target reached with mean belief gap below threshold (fallback path).",
  agent_convergence:
    "Agent-driven loop declared convergence.",
  unknown:
    "Reason not explicitly set — pipeline exited without a named path.",
};
const CONVERGENCE_REASON_IS_SUSPECT = new Set<string>([
  "unknown",
  "max_iterations_reached",
  "budget_exhausted",
]);

const FSM_STATE_COLORS: Record<string, string> = {
  IDLE: "default",
  LOADING_VOCAB: "processing",
  DISCOVERING: "processing",
  SAMPLING: "processing",
  LLM_SWEEP: "processing",
  VALIDATING: "processing",
  GENERATING_SYNTH: "processing",
  TRAINING: "processing",
  CLASSIFYING: "processing",
  FUSING: "processing",
  EVALUATING: "processing",
  CONVERGED: "success",
  ERROR: "error",
};

function WebTerminalAgentCard() {
  const [data, setData] = useState<TerminalModelsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [selecting, setSelecting] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const fetchModels = () => {
    setLoading(true);
    fetch("/api/terminal/models")
      .then((r) => r.json())
      .then((body: TerminalModelsResponse) => setData(body))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchModels();
    // Poll rolling stats every 15 s so the panel reflects live usage
    // without the operator having to refresh manually.
    const interval = setInterval(fetchModels, 15000);
    return () => clearInterval(interval);
  }, []);

  const handleSelect = (id: string) => {
    setSelecting(id);
    setErrorMsg(null);
    fetch("/api/terminal/models/active", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ id }),
    })
      .then((r) => r.json())
      .then((body) => {
        if (body.error) {
          setErrorMsg(String(body.error));
          return;
        }
        fetchModels();
      })
      .catch((e) => setErrorMsg(String(e)))
      .finally(() => setSelecting(null));
  };

  const handleClear = () => {
    setSelecting("__clear__");
    setErrorMsg(null);
    fetch("/api/terminal/models/active", { method: "DELETE" })
      .then(() => fetchModels())
      .finally(() => setSelecting(null));
  };

  const formatContext = (n: number) =>
    n >= 1_000_000 ? `${(n / 1_000_000).toFixed(0)}M` :
    n >= 1_000 ? `${(n / 1_000).toFixed(0)}K` : String(n);

  const activeId = data?.active?.id ?? null;
  const rows = data?.models ?? [];

  const columns = [
    {
      title: "",
      dataIndex: "id",
      key: "radio",
      width: 40,
      render: (id: string, row: TerminalModelEntry) => (
        <input
          type="radio"
          name="terminal-model"
          checked={activeId === id}
          disabled={!row.available || selecting !== null}
          onChange={() => handleSelect(id)}
          aria-label={`Select ${row.label}`}
        />
      ),
    },
    {
      title: "Model",
      dataIndex: "label",
      key: "label",
      render: (label: string, row: TerminalModelEntry) => (
        <Space direction="vertical" size={0}>
          <Text strong>{label}</Text>
          <Text type="secondary" style={{ fontSize: 11 }}>
            {row.thinking === "adaptive" ? "adaptive thinking" :
             row.thinking === "extended" ? "extended thinking" :
             "no thinking"}
          </Text>
        </Space>
      ),
    },
    {
      title: "Provider",
      dataIndex: "provider",
      key: "provider",
      render: (provider: string, row: TerminalModelEntry) => (
        <Tooltip title={row.unavailable_reason || row.model_ref}>
          <Tag color={provider === "bedrock" ? "blue" : "green"}>
            {provider === "bedrock" ? "Bedrock" : "Anthropic"}
          </Tag>
          {!row.available && <Tag color="default">unavailable</Tag>}
        </Tooltip>
      ),
    },
    {
      title: "Context",
      dataIndex: "context_window",
      key: "context",
      render: (n: number) => <Text code>{formatContext(n)}</Text>,
    },
    {
      title: "Max Output",
      dataIndex: "max_output_tokens",
      key: "max_out",
      render: (n: number) => <Text code>{formatContext(n)}</Text>,
    },
    {
      title: "TTFT (p50)",
      key: "ttft",
      render: (_: unknown, row: TerminalModelEntry) =>
        row.stats.n >= 3
          ? <Text>{row.stats.ttft_ms_p50.toFixed(0)} ms</Text>
          : <Text type="secondary">—</Text>,
    },
    {
      title: "Throughput (p50)",
      key: "tokps",
      render: (_: unknown, row: TerminalModelEntry) =>
        row.stats.n >= 3
          ? <Text>{row.stats.tokps_p50.toFixed(1)} tok/s</Text>
          : <Text type="secondary">—</Text>,
    },
    {
      title: "n",
      key: "n",
      width: 50,
      render: (_: unknown, row: TerminalModelEntry) =>
        <Tag color={row.stats.n > 0 ? "cyan" : "default"}>{row.stats.n}</Tag>,
    },
  ];

  return (
    <Card
      title="Web Terminal Agent"
      size="small"
      extra={
        <Space>
          {data?.override_set && (
            <Button size="small" onClick={handleClear}
                    disabled={selecting !== null}>
              Reset to default
            </Button>
          )}
          <Button size="small" icon={<ReloadOutlined />}
                  onClick={fetchModels} loading={loading}>
            Refresh
          </Button>
        </Space>
      }
    >
      <Paragraph type="secondary" style={{ marginTop: 0 }}>
        Pick the model the embedded Claude Agent SDK session routes through.
        Selection is gateway-lifetime — it sticks until the service restarts.
        Rolling stats accumulate over the last 20 queries per model.
        {data?.override_set && (
          <Text strong> · Override active: {data.active?.label}</Text>
        )}
      </Paragraph>
      {errorMsg && (
        <Paragraph type="danger" style={{ marginBottom: 12 }}>
          {errorMsg}
        </Paragraph>
      )}
      <Table
        rowKey="id"
        dataSource={rows}
        columns={columns}
        pagination={false}
        size="small"
        loading={loading && !data}
      />
    </Card>
  );
}


function ClassificationPipelineCard({ hasClassifyLlm }: { hasClassifyLlm?: boolean }) {
  const [fsm, setFsm] = useState<FSMStatus | null>(null);
  const [fsmLoading, setFsmLoading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const { activeSourceId, setActiveSourceId, sources, refreshDatasets, refreshArtifactSets } = useDataset();
  const activeSource = sources.find((s) => s.id === activeSourceId);

  const fetchFSM = () => {
    setFsmLoading(true);
    fetch("/api/fsm/status")
      .then((r) => r.json())
      .then(setFsm)
      .catch(() => setFsm(null))
      .finally(() => setFsmLoading(false));
  };

  useEffect(() => {
    fetchFSM();
    const interval = setInterval(fetchFSM, 5000);
    return () => clearInterval(interval);
  }, []);

  const startPipeline = () => {
    setStarting(true);
    const params = activeSourceId ? `?source_id=${encodeURIComponent(activeSourceId)}` : "";
    fetch(`/api/fsm/start${params}`, { method: "POST" })
      .then((r) => r.json())
      .then(() => {
        setTimeout(fetchFSM, 500);
      })
      .catch(() => {})
      .finally(() => setStarting(false));
  };

  const cancelPipeline = () => {
    setCancelling(true);
    fetch("/api/fsm/cancel", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ reason: "operator cancelled from Status panel" }),
    })
      .then((r) => r.json())
      .then((body) => {
        if (body?.error) {
          message.warning(body.error);
        } else if (body?.cancelled) {
          message.info(
            "Cancellation requested — pipeline will exit cleanly after the current batch.",
          );
        }
        setTimeout(fetchFSM, 500);
      })
      .catch(() => {})
      .finally(() => setCancelling(false));
  };

  const state = fsm?.state ?? "IDLE";
  const isRunning = !["IDLE", "CONVERGED", "ERROR"].includes(state);
  const progress = fsm?.progress ?? {};

  // When a run is active (or completed), show the source the run was
  // started against (from the FSM record) — not the sidebar's active
  // source, which the operator can change mid-run.  Fall back to the
  // sidebar source only when IDLE (no prior run) so the operator sees
  // what will be classified next.
  const runSourceId = fsm?.source_id;
  const runSource = runSourceId
    ? sources.find((s) => s.id === runSourceId)
    : null;
  const displaySource = state === "IDLE" ? activeSource : (runSource ?? activeSource);

  // Refresh datasets + artifact sets when pipeline converges so the
  // new dataset and its produced artifact set both appear without a
  // manual click.  Extend runs also converge — same code path.
  //
  // When the converged run's source differs from the active sidebar
  // source, switch to it so the new dataset shows in the Embeddings
  // list without a manual source change.
  //
  // Fire once per converged run, keyed by ``fsm.id``: the prior version
  // re-fired on every render while ``state === "CONVERGED"``, which
  // would forcibly switch the source back if the operator manually
  // changed it post-convergence.  ``handledRunRef`` records which run
  // we already handled so the source-switch is a one-shot event.
  const handledRunRef = useRef<string | null>(null);
  const fsmRunId = fsm?.id ?? null;
  useEffect(() => {
    if (state !== "CONVERGED") return;
    if (!fsmRunId || handledRunRef.current === fsmRunId) return;
    handledRunRef.current = fsmRunId;
    if (runSourceId && runSourceId !== activeSourceId) {
      // setActiveSourceId clears activeDatasetId and triggers
      // refreshDatasets via the DatasetContext source-change effect.
      setActiveSourceId(runSourceId);
    } else {
      refreshDatasets();
    }
    refreshArtifactSets();
  }, [
    state, fsmRunId, runSourceId, activeSourceId,
    setActiveSourceId, refreshDatasets, refreshArtifactSets,
  ]);

  return (
    <Card
      title="Classification Pipeline"
      extra={
        <Space>
          {displaySource ? (
            <Text type="secondary">
              Source: <Text code>{displaySource.display_name}</Text>
            </Text>
          ) : runSourceId ? (
            <Text type="secondary">
              Source: <Text code>{runSourceId}</Text>
            </Text>
          ) : (
            <Text type="secondary">No source selected</Text>
          )}
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchFSM}
            loading={fsmLoading}
            size="small"
          >
            Refresh
          </Button>
          {isRunning ? (
            <Button
              danger
              onClick={cancelPipeline}
              loading={cancelling}
              size="small"
            >
              Stop
            </Button>
          ) : (
            <Button
              type="primary"
              onClick={startPipeline}
              loading={starting}
              size="small"
            >
              Start Classification
            </Button>
          )}
        </Space>
      }
    >
      <Descriptions column={2} size="small" bordered>
        <Descriptions.Item label="State">
          <Tag color={FSM_STATE_COLORS[state] ?? "default"}>{state}</Tag>
          {state === "CONVERGED" && progress.convergence_reason != null && (
            <Tag
              color={
                CONVERGENCE_REASON_IS_SUSPECT.has(
                  String(progress.convergence_reason),
                )
                  ? "orange"
                  : "blue"
              }
              style={{ marginLeft: 8 }}
              title={(() => {
                // Detail (free-form prose from the agent loop's
                // declare_converged) takes precedence as the tooltip
                // body — it's the actual run-specific reasoning.
                // CONVERGENCE_REASON_DESCRIPTIONS[tag] is the generic
                // description fallback when no detail is present
                // (programmatic-loop runs).
                const detail = progress.convergence_reason_detail;
                if (detail) return String(detail);
                return String(
                  CONVERGENCE_REASON_DESCRIPTIONS[
                    String(progress.convergence_reason)
                  ] ?? progress.convergence_reason,
                );
              })()}
            >
              {String(progress.convergence_reason)}
            </Tag>
          )}
        </Descriptions.Item>
        <Descriptions.Item label="LLM Backend">
          <Tag color={hasClassifyLlm ? "green" : "orange"}>
            {hasClassifyLlm ? "Configured" : "Not configured"}
          </Tag>
        </Descriptions.Item>
        <Descriptions.Item label="Run ID">
          <Text code>{fsm?.id ?? "—"}</Text>
        </Descriptions.Item>
        {progress.categories_loaded != null && (
          <Descriptions.Item label="Terms">
            {String(progress.categories_loaded)}
          </Descriptions.Item>
        )}
        {progress.tables_classifiable != null ? (
          <Descriptions.Item label="Tables">
            {String(progress.tables_classifiable)}
            {progress.tables_discovered_raw != null &&
              progress.tables_discovered_raw !== progress.tables_classifiable && (
                <Text type="secondary" style={{ marginLeft: 8 }}>
                  ({String(progress.tables_filtered)} filtered,{" "}
                  {String(progress.tables_discovered_raw)} discovered)
                </Text>
              )}
          </Descriptions.Item>
        ) : progress.tables_discovered != null ? (
          <Descriptions.Item label="Tables">
            {String(progress.tables_discovered)}
          </Descriptions.Item>
        ) : null}
        {progress.columns_sampled != null && (
          <Descriptions.Item label="Columns Sampled">
            {String(progress.columns_sampled)}
          </Descriptions.Item>
        )}
        {progress.llm_labeled != null && (
          <Descriptions.Item label="LLM Labeled">
            {String(progress.llm_labeled)}
          </Descriptions.Item>
        )}
        {progress.iteration != null && (
          <Descriptions.Item label="Iteration">
            {String(progress.iteration)}
          </Descriptions.Item>
        )}
        {/* ── Tier 1: convergence (primary stopping criterion) ──
            Mean Gap is the actual gate; Trend Ratio is the
            iteration-to-iteration ρ over the gap (the honest
            single-criterion contraction).  Color thresholds reference
            the operator-configured gap_threshold so a tightened policy
            re-colors the live tag without a code change. */}
        {progress.mean_gap != null && (
          <Descriptions.Item label="Mean Gap (Pl − Bel)">
            <Tag
              color={
                progress.gap_threshold != null &&
                Number(progress.mean_gap) <= Number(progress.gap_threshold)
                  ? "green"
                  : progress.gap_threshold != null &&
                      Number(progress.mean_gap) <=
                        Number(progress.gap_threshold) * 1.5
                    ? "orange"
                    : "red"
              }
            >
              {Number(progress.mean_gap).toFixed(3)}
            </Tag>
            {progress.gap_threshold != null && (
              <Text type="secondary" style={{ marginLeft: 8 }}>
                target ≤ {Number(progress.gap_threshold).toFixed(2)}
              </Text>
            )}
          </Descriptions.Item>
        )}
        {progress.gap_contraction_rate != null &&
          Number(progress.gap_contraction_rate) > 0 && (
            <Descriptions.Item label="Trend Ratio">
              <Tag
                color={
                  Number(progress.gap_contraction_rate) < 0.7
                    ? "green"
                    : Number(progress.gap_contraction_rate) < 0.95
                      ? "orange"
                      : "red"
                }
              >
                {Number(progress.gap_contraction_rate).toFixed(2)}
              </Tag>
              <Text type="secondary" style={{ marginLeft: 8 }}>
                gapₙ / gapₙ₋₁ — &lt;1 tightening, →1 stalled
              </Text>
            </Descriptions.Item>
          )}
        {progress.columns_classified != null && (
          <Descriptions.Item label="Classified">
            {String(progress.columns_classified)}
          </Descriptions.Item>
        )}
        {progress.accuracy != null && (
          <Descriptions.Item label="Accuracy">
            <Tag color={Number(progress.accuracy) > 0.7 ? "green" : "orange"}>
              {(Number(progress.accuracy) * 100).toFixed(1)}%
            </Tag>
          </Descriptions.Item>
        )}
        {/* ── Tier 2: thesis core ──
            LLM-Fit Labels renders f directly (the LLM-labeled fraction
            in the operator's thesis); Revisit Queue is the
            next-iteration LLM workload, surfaced as both count (LLM
            budget) and fraction (scale-invariant thesis load). */}
        {progress.bootstrap_coverage != null && (
          <Descriptions.Item label="Coverage">
            <Tag color={Number(progress.bootstrap_coverage) >= 0.95 ? "green" : "orange"}>
              {(Number(progress.bootstrap_coverage) * 100).toFixed(1)}%
            </Tag>
          </Descriptions.Item>
        )}
        {progress.llm_coverage != null && (
          <Descriptions.Item label="LLM Coverage">
            <Tag
              color={
                Number(progress.llm_coverage) >= 0.95
                  ? "green"
                  : Number(progress.llm_coverage) >= 0.80
                    ? "orange"
                    : "red"
              }
            >
              {(Number(progress.llm_coverage) * 100).toFixed(1)}%
            </Tag>
          </Descriptions.Item>
        )}
        {progress.llm_fit_labels != null && (
          <Descriptions.Item label="LLM-Fit Labels (f)">
            {String(progress.llm_fit_labels)}
            {progress.llm_fit_fraction != null && (
              <Text type="secondary" style={{ marginLeft: 8 }}>
                f = {(Number(progress.llm_fit_fraction) * 100).toFixed(1)}%
              </Text>
            )}
          </Descriptions.Item>
        )}
        {progress.disagreements_count != null && (
          <Descriptions.Item label="Revisit Queue">
            {String(progress.disagreements_count)}
            {progress.disagreements_frac != null && (
              <Text type="secondary" style={{ marginLeft: 8 }}>
                ({(Number(progress.disagreements_frac) * 100).toFixed(1)}% of corpus)
              </Text>
            )}
          </Descriptions.Item>
        )}
        {/* ── Tier 3: prediction strength + evidence conflict ──
            Clarity = 1 − frac_unclear (fraction of cols clear of the
            gap/bel thresholds); K is evidence-conflict — the signal
            that distinguishes "LLM right, done" from "indep tier knows
            something the LLM is missing."  Both load-bearing under
            the operator's thesis even though K no longer gates
            convergence directly. */}
        {progress.mean_bel != null && (
          <Descriptions.Item label="Mean Belief">
            <Tag
              color={
                progress.bel_floor != null &&
                Number(progress.mean_bel) >= Number(progress.bel_floor)
                  ? "green"
                  : "orange"
              }
            >
              {Number(progress.mean_bel).toFixed(3)}
            </Tag>
            {progress.bel_floor != null && (
              <Text type="secondary" style={{ marginLeft: 8 }}>
                floor ≥ {Number(progress.bel_floor).toFixed(2)}
              </Text>
            )}
          </Descriptions.Item>
        )}
        {progress.clarity != null && (
          <Descriptions.Item label="Clarity">
            <Tag
              color={
                progress.clarity_target != null &&
                Number(progress.clarity) >= 1 - Number(progress.clarity_target)
                  ? "green"
                  : progress.clarity_target != null &&
                      Number(progress.clarity) >=
                        1 - Number(progress.clarity_target) * 1.2
                    ? "orange"
                    : "red"
              }
            >
              {(Number(progress.clarity) * 100).toFixed(1)}%
            </Tag>
            {progress.clarity_target != null && (
              <Text type="secondary" style={{ marginLeft: 8 }}>
                target ≥ {((1 - Number(progress.clarity_target)) * 100).toFixed(0)}%
              </Text>
            )}
          </Descriptions.Item>
        )}
        {progress.mean_k != null && (
          <Descriptions.Item label="Evidence Conflict (K)">
            {Number(progress.mean_k).toFixed(3)}
          </Descriptions.Item>
        )}
        {progress.indep_tier_disagreement_frac != null && (
          <Descriptions.Item label="Indep-Tier Disagreement">
            {(Number(progress.indep_tier_disagreement_frac) * 100).toFixed(1)}%
          </Descriptions.Item>
        )}
        {progress.llm_agreement != null && (
          <Descriptions.Item label="LLM Agreement">
            <Tag
              color={
                Number(progress.llm_agreement) >= 0.98
                  ? "green"
                  : Number(progress.llm_agreement) >= 0.90
                    ? "orange"
                    : "red"
              }
            >
              {(Number(progress.llm_agreement) * 100).toFixed(1)}%
            </Tag>
          </Descriptions.Item>
        )}
        {progress.avg_confidence != null && (
          <Descriptions.Item label="Avg Confidence">
            {Number(progress.avg_confidence).toFixed(3)}
          </Descriptions.Item>
        )}
        {progress.avg_conflict != null && (
          <Descriptions.Item label="Avg Conflict">
            {Number(progress.avg_conflict).toFixed(3)}
          </Descriptions.Item>
        )}
        {progress.llm_calls != null && (
          <Descriptions.Item label="LLM Calls">
            {String(progress.llm_calls)}
          </Descriptions.Item>
        )}
        {progress.sweep_batches != null && (
          <Descriptions.Item label="Batches">
            {String(progress.sweep_batches)}
          </Descriptions.Item>
        )}
        {progress.sweep_batch_size != null && (
          <Descriptions.Item label="Batch Size">
            {String(progress.sweep_batch_size)}
          </Descriptions.Item>
        )}
        {progress.sweep_elapsed_s != null && (
          <Descriptions.Item label="Sweep Elapsed">
            {(() => {
              const s = Number(progress.sweep_elapsed_s);
              const m = Math.floor(s / 60);
              const sec = Math.floor(s % 60);
              return `${m}m ${sec.toString().padStart(2, "0")}s`;
            })()}
          </Descriptions.Item>
        )}
        {progress.sweep_phase != null && (
          <Descriptions.Item label="Sub-phase">
            <Tag>{String(progress.sweep_phase)}</Tag>
          </Descriptions.Item>
        )}
        {progress.sweep_truncations != null &&
          Number(progress.sweep_truncations) > 0 && (
            <Descriptions.Item label="Truncations">
              <Tag color="orange">{String(progress.sweep_truncations)}</Tag>
            </Descriptions.Item>
          )}
        {progress.sweep_failed != null &&
          Number(progress.sweep_failed) > 0 && (
            <Descriptions.Item label="Failed Columns">
              <Tag color="red">{String(progress.sweep_failed)}</Tag>
            </Descriptions.Item>
          )}
        {progress.sweep_throttled != null &&
          Number(progress.sweep_throttled) > 0 && (
            <Descriptions.Item label="Throttled">
              <Tag color="orange">
                {String(progress.sweep_throttled)} retried at same size
              </Tag>
            </Descriptions.Item>
          )}
      </Descriptions>
      {state === "LLM_SWEEP" &&
        progress.columns_total != null &&
        progress.llm_labeled != null && (
          <div style={{ marginTop: 12 }}>
            <Progress
              percent={
                Number(progress.columns_total) > 0
                  ? Math.round(
                      (Number(progress.llm_labeled) /
                        Number(progress.columns_total)) *
                        100,
                    )
                  : 0
              }
              format={() =>
                `${progress.llm_labeled} / ${progress.columns_total} columns`
              }
              status="active"
            />
          </div>
        )}
      {fsm?.error && (
        <div style={{ marginTop: 12 }}>
          <Text type="danger">{fsm.error}</Text>
        </div>
      )}
      {fsm?.result_path && state === "CONVERGED" && (
        <div style={{ marginTop: 12 }}>
          <Text type="secondary">Results: </Text>
          <Text code>{fsm.result_path}</Text>
        </div>
      )}
    </Card>
  );
}

// Shape returned by /api/artifact-sets/{id}/extend-scope.  Each field
// the panel surfaces as a salient measure for the *selected* RunID.
interface ExtendScope {
  artifact_set_id: string;
  source_id: string;
  same_source: boolean;
  bundle: { catboost: boolean; svm: boolean; umap: boolean };
  classes_count: number;
  embedding_model: string | null;
  embedding_dim: number | null;
  vocab_signature: string | null;
  is_active: boolean;
  is_archived: boolean;
  created_at: string | null;
  summary: string | null;
  training_source_id: string | null;
  fsm_run_id: string | null;
  producing_dataset_id: string | null;
  source_table_count: number | null;
  source_column_count: number | null;
  classified_table_count: number | null;
  classified_column_count: number | null;
  new_table_count: number | null;
  new_column_count: number | null;
  vocab_compatibility: {
    status: "ok" | "superset" | "partial" | "disjoint";
    missing_codes: string[];
    extra_codes: string[];
    artifact_signature: string;
    candidate_signature: string;
  };
}

const COMPAT_COLOR: Record<string, string> = {
  ok: "green",
  superset: "blue",
  partial: "orange",
  disjoint: "red",
};

const COMPAT_DESCRIPTION: Record<string, string> = {
  ok: "Artifact vocab matches the source vocab exactly.",
  superset: "Source vocab includes all artifact codes plus extras the model can't predict.",
  partial: "Artifact has codes the source vocab doesn't define — Extend will still run, but predictions may emit unknown codes.",
  disjoint: "No vocab overlap — almost certainly the wrong artifact for this source.",
};

function MLArtifactsCard() {
  const {
    activeSourceId,
    sources,
    datasets,
    activeDatasetId,
    artifactSets,
    setActiveArtifactSetId,
    refreshArtifactSets,
  } = useDataset();

  const [scope, setScope] = useState<ExtendScope | null>(null);
  const [scopeLoading, setScopeLoading] = useState(false);
  const [extending, setExtending] = useState<boolean>(false);

  const activeSource = sources.find((s) => s.id === activeSourceId);
  const activeDataset = datasets.find((d) => d.id === activeDatasetId);

  // Resolve the artifact set to display.  Priority:
  //   1. The artifact set the selected dataset CONSUMED (Extend runs)
  //      OR PRODUCED (classify runs) — both stamp dataset.artifact_set_id.
  //   2. Lineage fallback: match by fsm_run_id.
  //   3. None — render an empty-state hint.
  const targetArtifactSet = (() => {
    if (!activeDataset) return null;
    if (activeDataset.artifact_set_id) {
      const m = artifactSets.find((a) => a.id === activeDataset.artifact_set_id);
      if (m) return m;
    }
    if (activeDataset.fsm_run_id) {
      const m = artifactSets.find((a) => a.fsm_run_id === activeDataset.fsm_run_id);
      if (m) return m;
    }
    return null;
  })();

  const fetchScope = useCallback(async () => {
    if (!targetArtifactSet || !activeSourceId) {
      setScope(null);
      return;
    }
    setScopeLoading(true);
    try {
      const r = await fetch(
        `/api/artifact-sets/${encodeURIComponent(targetArtifactSet.id)}/extend-scope?source_id=${encodeURIComponent(activeSourceId)}`,
      );
      const data = await r.json();
      if (data.error) {
        setScope(null);
      } else {
        setScope(data as ExtendScope);
      }
    } catch {
      setScope(null);
    } finally {
      setScopeLoading(false);
    }
  }, [targetArtifactSet, activeSourceId]);

  useEffect(() => {
    fetchScope();
  }, [fetchScope]);

  const promote = async () => {
    if (!targetArtifactSet) return;
    await setActiveArtifactSetId(targetArtifactSet.id);
    await fetchScope();
  };

  const startExtend = async () => {
    if (!activeSourceId || !targetArtifactSet) {
      message.error("Select a data source and a Run ID with an artifact set first");
      return;
    }
    setExtending(true);
    try {
      const r = await fetch("/api/fsm/extend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_id: activeSourceId,
          artifact_set_id: targetArtifactSet.id,
          parent_dataset_id: activeDatasetId ?? null,
        }),
      });
      const data = await r.json();
      if (data.started) {
        message.success(
          `Extend Classification started — using artifact ${targetArtifactSet.id.slice(0, 8)}`,
        );
      } else {
        message.error(data.error || "Failed to start Extend Classification");
      }
    } catch (e) {
      message.error(`Extend Classification failed: ${e}`);
    } finally {
      setExtending(false);
    }
  };

  const canExtend = Boolean(activeSourceId && targetArtifactSet && activeDataset);

  // Header label — mirrors the Classification Pipeline card's pattern of
  // surfacing the source the *selected run* was about, not the sidebar's
  // current source (which the operator may have changed).
  const headerSummary = (() => {
    if (!activeSource) return <Text type="secondary">No source selected</Text>;
    if (!activeDataset) return (
      <Text type="secondary">
        Source: <Text code>{activeSource.display_name}</Text>
      </Text>
    );
    return (
      <Text type="secondary">
        Source: <Text code>{activeSource.display_name}</Text>{" · "}
        Run: <Text code>{(activeDataset.fsm_run_id ?? activeDataset.id).slice(0, 8)}</Text>
        {" · "}v{activeDataset.version_number}
      </Text>
    );
  })();

  return (
    <Card
      title="ML Artifacts"
      extra={
        <Space>
          {headerSummary}
          <Button
            icon={<ReloadOutlined />}
            onClick={() => { refreshArtifactSets(); fetchScope(); }}
            loading={scopeLoading}
            size="small"
          >
            Refresh
          </Button>
          <Tooltip
            title={
              !canExtend
                ? "Select a Run ID in Data Source whose run produced (or consumed) an artifact set."
                : "Apply this run's CatBoost (and SVM/UMAP if bundled) to the selected source — skips the LLM sweep, DST iteration, and re-training."
            }
          >
            <Button
              type="primary"
              icon={<RocketOutlined />}
              onClick={startExtend}
              loading={extending}
              disabled={!canExtend}
              size="small"
            >
              Extend Classification
            </Button>
          </Tooltip>
        </Space>
      }
    >
      {!activeDataset ? (
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          No Run ID selected. Pick a row in the Data Source panel — its
          Run produces (or consumes) the artifact set displayed here.
        </Paragraph>
      ) : !targetArtifactSet ? (
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          The selected Run ({(activeDataset.fsm_run_id ?? activeDataset.id).slice(0, 8)})
          has no registered artifact set.{" "}
          {artifactSets.length === 0
            ? "Run the Classification Pipeline to produce one — each completed classify run registers its trained CatBoost / SVM / UMAP bundle for re-use."
            : "This is normal for legacy runs that pre-date artifact-set lineage."}
        </Paragraph>
      ) : (
        <>
          <Descriptions column={2} size="small" bordered>
            <Descriptions.Item label="Status">
              <Space>
                {targetArtifactSet.is_active ? (
                  <Tag color="green">Active</Tag>
                ) : targetArtifactSet.is_archived ? (
                  <Tag>Archived</Tag>
                ) : (
                  <Tag color="blue">Available</Tag>
                )}
                {!targetArtifactSet.is_active && !targetArtifactSet.is_archived && (
                  <Button size="small" type="link" onClick={promote}>
                    Make active
                  </Button>
                )}
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="Bundle">
              <Space size={4}>
                <Tooltip title="CatBoost classifier (always present in a registered set)">
                  <Tag
                    color={targetArtifactSet.catboost_path ? "blue" : "default"}
                    style={{ margin: 0 }}
                  >
                    CB
                  </Tag>
                </Tooltip>
                <Tooltip
                  title={
                    targetArtifactSet.svm_path
                      ? "Incremental SVM bundled — second-look classifier"
                      : "No SVM in this bundle"
                  }
                >
                  <Tag
                    color={targetArtifactSet.svm_path ? "blue" : "default"}
                    style={{ margin: 0 }}
                  >
                    SVM
                  </Tag>
                </Tooltip>
                <Tooltip
                  title={
                    targetArtifactSet.umap_path
                      ? "UMAP projection bundled — Extend lands in same coordinate space"
                      : "No UMAP — Extend re-fits a fresh projection"
                  }
                >
                  <Tag
                    color={targetArtifactSet.umap_path ? "blue" : "default"}
                    style={{ margin: 0 }}
                  >
                    UMAP
                  </Tag>
                </Tooltip>
              </Space>
            </Descriptions.Item>
            <Descriptions.Item label="Run ID">
              {targetArtifactSet.fsm_run_id ? (
                <Link to={`/overwatch/${targetArtifactSet.fsm_run_id}`}>
                  <Text code>{targetArtifactSet.fsm_run_id.slice(0, 8)}</Text>
                </Link>
              ) : (
                <Text code type="secondary">
                  {targetArtifactSet.id.slice(0, 8)}
                </Text>
              )}
            </Descriptions.Item>
            <Descriptions.Item label="Created">
              {targetArtifactSet.created_at ? (() => {
                try {
                  return new Date(targetArtifactSet.created_at).toLocaleString();
                } catch {
                  return targetArtifactSet.created_at;
                }
              })() : "—"}
            </Descriptions.Item>
            <Descriptions.Item label="Classes">
              {scope?.classes_count ?? "—"}
            </Descriptions.Item>
            <Descriptions.Item label="Embedding">
              {scope?.embedding_model ? (
                <>
                  <Text code>{scope.embedding_model}</Text>
                  {scope.embedding_dim != null && (
                    <Text type="secondary" style={{ marginLeft: 6 }}>
                      ({scope.embedding_dim}d)
                    </Text>
                  )}
                </>
              ) : (
                "—"
              )}
            </Descriptions.Item>

            {/* ── Training scope (what CatBoost was fit on) ── */}
            {scope?.classified_column_count != null && (
              <Descriptions.Item label="Trained On">
                {scope.classified_column_count.toLocaleString()} columns
                {scope.classified_table_count != null && (
                  <Text type="secondary" style={{ marginLeft: 6 }}>
                    across {scope.classified_table_count.toLocaleString()} tables
                  </Text>
                )}
              </Descriptions.Item>
            )}

            {/* ── Source scope + delta — the "extend headroom" the user
                wants to see for Extend Classification planning. ── */}
            {scope?.source_column_count != null && (
              <Descriptions.Item label="Source Scope">
                {scope.source_column_count.toLocaleString()} columns
                {scope.source_table_count != null && (
                  <Text type="secondary" style={{ marginLeft: 6 }}>
                    across {scope.source_table_count.toLocaleString()} tables
                  </Text>
                )}
              </Descriptions.Item>
            )}
            {scope &&
              (scope.new_column_count != null || scope.new_table_count != null) && (
                <Descriptions.Item
                  label={
                    <Tooltip
                      title={
                        scope.same_source
                          ? "Entities present in the source today that this artifact set hasn't classified yet — what an Extend run would target."
                          : "Artifact was trained on a different source — Extend would classify the candidate source from scratch using this CatBoost."
                      }
                    >
                      New Entities
                    </Tooltip>
                  }
                >
                  <Tag
                    color={
                      scope.new_column_count && scope.new_column_count > 0
                        ? "purple"
                        : "default"
                    }
                  >
                    {scope.new_column_count != null
                      ? `${scope.new_column_count.toLocaleString()} columns`
                      : "—"}
                  </Tag>
                  {scope.new_table_count != null && scope.new_table_count > 0 && (
                    <Text type="secondary" style={{ marginLeft: 6 }}>
                      across {scope.new_table_count.toLocaleString()} new tables
                    </Text>
                  )}
                  {!scope.same_source && (
                    <Text type="secondary" style={{ marginLeft: 6 }}>
                      (cross-source extend)
                    </Text>
                  )}
                </Descriptions.Item>
              )}

            {/* ── Vocab compatibility — surfaced inline so the operator
                sees the warning before clicking Extend. ── */}
            {scope?.vocab_compatibility && (
              <Descriptions.Item label="Vocab" span={2}>
                <Tooltip
                  title={
                    COMPAT_DESCRIPTION[scope.vocab_compatibility.status] ??
                    scope.vocab_compatibility.status
                  }
                >
                  <Tag
                    color={
                      COMPAT_COLOR[scope.vocab_compatibility.status] ?? "default"
                    }
                  >
                    {scope.vocab_compatibility.status}
                  </Tag>
                </Tooltip>
                {scope.vocab_compatibility.missing_codes.length > 0 && (
                  <Text type="secondary" style={{ marginLeft: 6 }}>
                    {scope.vocab_compatibility.missing_codes.length} artifact
                    code{scope.vocab_compatibility.missing_codes.length === 1
                      ? ""
                      : "s"}{" "}
                    not in source vocab
                  </Text>
                )}
                {scope.vocab_compatibility.extra_codes.length > 0 && (
                  <Text type="secondary" style={{ marginLeft: 6 }}>
                    · {scope.vocab_compatibility.extra_codes.length} source
                    code{scope.vocab_compatibility.extra_codes.length === 1
                      ? ""
                      : "s"}{" "}
                    artifact can't predict
                  </Text>
                )}
              </Descriptions.Item>
            )}

            {targetArtifactSet.summary && (
              <Descriptions.Item label="Summary" span={2}>
                {targetArtifactSet.summary}
              </Descriptions.Item>
            )}
          </Descriptions>
        </>
      )}
    </Card>
  );
}


function DataSourceCard() {
  const {
    sources,
    activeSourceId,
    setActiveSourceId,
    datasets,
    activeDatasetId,
    setActiveDatasetId,
    refreshDatasets,
    refreshArtifactSets,
  } = useDataset();

  const [archiving, setArchiving] = useState<string | null>(null);

  const activateVersion = (datasetId: string) => {
    fetch(`/api/datasets/${encodeURIComponent(datasetId)}/activate`, {
      method: "POST",
    })
      .then((r) => r.json())
      .then(() => {
        // userPicked=true: this is an explicit operator activation,
        // not an auto-promote.  Pins the choice so the auto-promote
        // effect won't bounce back to the previously-active row before
        // the next refreshDatasets settles is_active flags.
        setActiveDatasetId(datasetId, { userPicked: true });
        // Refresh so the local datasets list picks up the new
        // is_active flags from the server, keeping the UI in sync
        // without relying on a poll interval.
        return refreshDatasets();
      })
      .catch(() => {});
  };

  // Archive both the dataset and its associated artifact set in one
  // click.  Operators think of the row as "the run" — splitting these
  // across two panels means an archive on one side leaves the other
  // dangling and visible.  Files on disk are untouched (soft-delete).
  const archiveRun = async (
    datasetId: string,
    artifactSetId: string | null | undefined,
  ) => {
    setArchiving(datasetId);
    try {
      const calls: Promise<unknown>[] = [
        fetch(`/api/datasets/${encodeURIComponent(datasetId)}/archive`, {
          method: "POST",
        }),
      ];
      if (artifactSetId) {
        calls.push(
          fetch(
            `/api/artifact-sets/${encodeURIComponent(artifactSetId)}/archive`,
            { method: "POST" },
          ),
        );
      }
      await Promise.all(calls);
      await Promise.all([refreshDatasets(), refreshArtifactSets()]);
    } finally {
      setArchiving(null);
    }
  };

  return (
    <Card
      title="Data Source"
      extra={
        <Space>
          <Select
            value={activeSourceId ?? undefined}
            onChange={(v) => setActiveSourceId(v ?? null, { userPicked: true })}
            style={{ minWidth: 240 }}
            placeholder="No source selected"
            options={sources.map((s) => ({
              label: s.display_name,
              value: s.id,
            }))}
            disabled={sources.length === 0}
            size="small"
          />
          {activeDatasetId && (
            <Link to={`/embeddings/${activeDatasetId}`}>
              <Button icon={<EyeOutlined />} size="small">
                View Embeddings
              </Button>
            </Link>
          )}
        </Space>
      }
    >
      {datasets.length === 0 ? (
        <Paragraph type="secondary" style={{ marginBottom: 0 }}>
          No dataset versions yet. Run the classification pipeline to create one.
        </Paragraph>
      ) : (
        <Table
          size="small"
          pagination={false}
          rowKey="id"
          dataSource={datasets}
          onRow={(record) => ({
            style: {
              cursor: "pointer",
              background: record.id === activeDatasetId ? "#e6f4ff" : undefined,
            },
            onClick: () => activateVersion(record.id),
          })}
          columns={[
            {
              title: "Active",
              key: "active",
              width: 70,
              align: "center" as const,
              render: (_: unknown, record: { id: string; is_active: boolean }) => (
                <Radio
                  checked={record.is_active}
                  onClick={(e) => {
                    e.stopPropagation();
                    activateVersion(record.id);
                  }}
                />
              ),
            },
            {
              title: "Version",
              dataIndex: "version_number",
              key: "version",
              width: 80,
              render: (v: number) => <Text strong>v{v}</Text>,
            },
            {
              title: "Run ID",
              dataIndex: "fsm_run_id",
              key: "run_id",
              width: 110,
              render: (run_id: string | null, record: { id: string }) => {
                const id = run_id ?? record.id;
                return run_id ? (
                  <Link to={`/overwatch/${run_id}`}>
                    <Text code>{id.slice(0, 8)}</Text>
                  </Link>
                ) : (
                  <Text code type="secondary">{id.slice(0, 8)}</Text>
                );
              },
            },
            {
              title: "Columns",
              dataIndex: "row_count",
              key: "rows",
              width: 100,
              render: (v: number) => v?.toLocaleString() ?? "—",
            },
            {
              title: "Created",
              dataIndex: "created_at",
              key: "created",
              render: (v: string) => {
                if (!v) return "—";
                try {
                  return new Date(v).toLocaleString();
                } catch {
                  return v;
                }
              },
            },
            {
              title: "Summary",
              dataIndex: "summary",
              key: "summary",
              ellipsis: true,
              render: (v: string) => v || "—",
            },
            {
              title: "",
              key: "actions",
              width: 130,
              render: (_: unknown, record: any) => (
                <Space size={4} onClick={(e) => e.stopPropagation()}>
                  {record.fsm_run_id && (
                    <Link to={`/overwatch/${record.fsm_run_id}`}>
                      <Tag color="purple" style={{ cursor: "pointer", margin: 0 }}>
                        Overwatch
                      </Tag>
                    </Link>
                  )}
                  <Popconfirm
                    title="Archive this run?"
                    description={
                      record.is_active
                        ? "This is the active dataset — archiving hides both the dataset and its ML artifact bundle. Files on disk are kept."
                        : "Archives both the dataset and its ML artifact bundle. Files on disk are kept."
                    }
                    okText="Archive"
                    okButtonProps={{ danger: true }}
                    cancelText="Cancel"
                    onConfirm={() =>
                      archiveRun(record.id, record.artifact_set_id)
                    }
                  >
                    <Tooltip title="Archive run (dataset + ML artifacts)">
                      <Button
                        icon={<DeleteOutlined />}
                        size="small"
                        type="text"
                        danger
                        loading={archiving === record.id}
                      />
                    </Tooltip>
                  </Popconfirm>
                </Space>
              ),
            },
          ]}
        />
      )}
    </Card>
  );
}

// ReferenceColumnHandlingCard removed: reference-column exclusion is
// disabled at the pipeline level (all columns required for the current
// configuration).  The backend flag is retained as inert schema only.

function StatusBadge({ ok }: { ok: boolean }) {
  return ok ? (
    <Badge status="success" text="Healthy" />
  ) : (
    <Badge status="error" text="Unreachable" />
  );
}

// Live countdown to a future epoch-seconds timestamp.  Re-renders
// every second while ``until`` is in the future; collapses to "now"
// once the deadline has passed.  Used by the supervisor chip + the
// CAI Data Platform restart button to surface the backoff window.
function CountdownText({ until }: { until: number | null | undefined }) {
  const [now, setNow] = useState(() => Date.now() / 1000);
  useEffect(() => {
    if (!until) return;
    const id = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(id);
  }, [until]);
  if (!until) return null;
  const remaining = Math.max(0, Math.ceil(until - now));
  if (remaining === 0) return <Text type="secondary">restarting…</Text>;
  return <Text type="secondary">retry in {remaining}s</Text>;
}

function PGliteStatusChip({
  state,
  onRestart,
  restarting,
}: {
  state: PGliteSupervisorState | null;
  onRestart: () => void;
  restarting: boolean;
}) {
  if (!state || state.managed === false) return null;

  const probeOk = state.probe?.listening === true;
  const phase = state.state ?? "running";

  // Color logic: the chip reflects what the operator should *do*.
  // Green = healthy.  Orange = transient/recovering (no action).
  // Red = stuck (action: click Restart).
  let color: "green" | "orange" | "red" | "blue" = "blue";
  let label: string = phase;
  if (phase === "running" && probeOk) {
    color = "green";
    label = "running";
  } else if (phase === "running" && !probeOk) {
    // Process alive but socket not reachable — usually a brief
    // hand-off during a restart that the probe caught mid-flight.
    color = "orange";
    label = "probe failing";
  } else if (phase === "starting") {
    color = "blue";
    label = "starting";
  } else if (phase === "backoff") {
    color = "orange";
    label = "restarting";
  } else if (phase === "circuit_broken") {
    color = "red";
    label = "circuit broken";
  } else if (phase === "stopped") {
    color = "red";
    label = "stopped";
  }

  const tooltip = (
    <Space direction="vertical" size={2} style={{ fontSize: 12 }}>
      <Text style={{ color: "white" }}>State: {phase}</Text>
      {state.pid != null && <Text style={{ color: "white" }}>Child PID: {state.pid}</Text>}
      {state.restart_count != null && (
        <Text style={{ color: "white" }}>Restarts: {state.restart_count}</Text>
      )}
      {state.consecutive_failures != null && state.consecutive_failures > 0 && (
        <Text style={{ color: "white" }}>
          Consecutive failures: {state.consecutive_failures}
        </Text>
      )}
      {state.last_exit_code != null && (
        <Text style={{ color: "white" }}>Last exit: {state.last_exit_code}</Text>
      )}
      {state.started_at != null && (
        <Text style={{ color: "white" }}>
          Started: {formatAgo(state.started_at * 1000)}
        </Text>
      )}
      {state.probe?.error && (
        <Text style={{ color: "white" }}>Probe error: {state.probe.error}</Text>
      )}
    </Space>
  );

  return (
    <Space size={6}>
      <Tooltip title={tooltip}>
        <Tag color={color} style={{ cursor: "help", margin: 0 }}>
          {label}
        </Tag>
      </Tooltip>
      {phase === "backoff" && state.backoff_until != null && (
        <CountdownText until={state.backoff_until} />
      )}
      {phase === "circuit_broken" && (
        <Button
          size="small"
          danger
          icon={<ReloadOutlined />}
          loading={restarting}
          onClick={onRestart}
          title="Touch the supervisor restart sentinel — clears the circuit and respawns pglite immediately."
        >
          Restart
        </Button>
      )}
    </Space>
  );
}

export default function Status() {
  // Status-page selection state (platform, smoke result) lives in
  // DatasetContext so it survives in-app navigation.  activeSourceId
  // is here so the env-seeded source banner can compare against the
  // currently-selected source.
  const {
    sources,
    activeSourceId,
    statusPlatformId,
    setStatusPlatformId,
    smokeTest,
    setSmokeTest,
  } = useDataset();

  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // PGlite supervisor health.  Polled every 5s — the chip + restart
  // button on the CAI Data Platform panel both read from this single
  // source so they never disagree about phase/backoff/circuit-broken.
  const [pglite, setPglite] = useState<PGliteSupervisorState | null>(null);
  const [pgliteRestarting, setPgliteRestarting] = useState(false);

  const [credentials, setCredentials] = useState<CredentialResult | null>(null);
  const [credLoading, setCredLoading] = useState(false);

  // Smoke test result is cached in DatasetContext (localStorage-backed)
  // so the last run survives navigation; the "Last run Nm ago" chip +
  // staleness hint read from there.
  const smoke = smokeTest?.result ?? null;
  const [smokeLoading, setSmokeLoading] = useState(false);

  // Unified CAI Data Platform list — Hive connections + filesystem mounts.
  type Platform = {
    id: string;
    kind: "hive" | "filesystem";
    label: string;
    source_uri: string;
    vocab_uri: string;
    mount: string | null;
    table_count: number | null;
    column_count: number | null;
  };
  type FsStats = {
    ok: boolean;
    source_id: string;
    display_name: string;
    mount: string | null;
    vocab_uri: string;
    table_count: number | null;
    column_count: number | null;
    annotation_count: number | null;
    error?: string;
  };
  const [platforms, setPlatforms] = useState<Platform[]>([]);
  // Diagnostic state — surfaces whatever /api/data-platforms returned
  // so an empty panel distinguishes "endpoint errored" from "endpoint
  // returned no rows" from "request never completed".  Without this
  // every failure mode collapsed to the generic "No data platforms
  // registered" empty state, which hid real bugs.
  const [platformsError, setPlatformsError] = useState<string | null>(null);
  const [platformsLoading, setPlatformsLoading] = useState(false);
  // Selected platform id lives in the dataset context so it survives
  // in-app navigation.  Context exposes `string | null`; local code
  // reads it as `string | undefined` to keep antd's Select happy.
  const selectedPlatformId = statusPlatformId ?? undefined;
  const setSelectedPlatformId = (id: string | undefined) =>
    setStatusPlatformId(id ?? null);
  const selectedPlatform = platforms.find((p) => p.id === selectedPlatformId);
  // Hive code paths still key on the connection name; filesystem paths
  // don't use it.
  const selectedConn = selectedPlatform?.kind === "hive"
    ? selectedPlatform.id
    : undefined;
  const [fsStats, setFsStats] = useState<FsStats | null>(null);
  const [connLoading, setConnLoading] = useState(false);
  const [refreshResult, setRefreshResult] = useState<RefreshResult | null>(null);
  // Per-database state: which are enabled, and their vocab_uri selection
  const [dbEnabled, setDbEnabled] = useState<Record<string, boolean>>({});
  const [dbVocabUri, setDbVocabUri] = useState<Record<string, string>>({});

  const fetchStatus = () => {
    setLoading(true);
    fetch("/api/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null))
      .finally(() => setLoading(false));
  };

  const fetchPglite = () => {
    fetch("/api/pglite/status")
      .then((r) => r.json())
      .then((data: PGliteSupervisorState) => setPglite(data))
      .catch(() => setPglite(null));
  };

  const restartPglite = () => {
    setPgliteRestarting(true);
    fetch("/api/pglite/restart", { method: "POST" })
      .then((r) => r.json())
      .then((data) => {
        if (data?.ok) {
          message.success("PGlite restart requested — supervisor will respawn the process.");
          // Optimistic refresh; the watch loop polls every 2s on the
          // supervisor side, so by the next /api/pglite/status tick we
          // should see state=starting → state=running.
          setTimeout(fetchPglite, 1500);
        } else {
          message.error(data?.error ?? "Restart failed");
        }
      })
      .catch((e) => message.error(`Restart failed: ${e}`))
      .finally(() => setPgliteRestarting(false));
  };

  const fetchPlatforms = () => {
    setPlatformsLoading(true);
    setPlatformsError(null);
    fetch("/api/data-platforms")
      .then(async (r) => {
        const data = await r.json().catch(() => null);
        if (!r.ok || (data && typeof data === "object" && "error" in data)) {
          const msg = (data && data.error) || `HTTP ${r.status}`;
          setPlatformsError(String(msg));
          setPlatforms([]);
          return;
        }
        const list: Platform[] = Array.isArray(data?.platforms)
          ? data.platforms
          : [];
        setPlatforms(list);
        // Preserve the operator's persisted choice when it still exists
        // in the platform list; otherwise default to the first entry.
        const persistedStillValid =
          statusPlatformId && list.some((p) => p.id === statusPlatformId);
        if (!persistedStillValid && list.length > 0) {
          setSelectedPlatformId(list[0].id);
        }
      })
      .catch((e) => {
        setPlatformsError(String(e));
        setPlatforms([]);
      })
      .finally(() => setPlatformsLoading(false));
  };

  useEffect(() => {
    fetchStatus();
    // Auto-fire credential validation on every /status visit — cheap
    // probe ($0 vs ~$0.007 for the smoke test) so it's worth keeping
    // current state on screen without a manual button press.
    runCredentialCheck();
    fetchPlatforms();
    fetchPglite();
    // Poll supervisor health while the page is mounted.  5s is a
    // good compromise: faster than the 2s supervisor watch loop so
    // backoff transitions aren't visibly stale, slow enough to not
    // burn CPU on a panel the operator may leave open.
    const id = setInterval(fetchPglite, 5000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When the selected platform becomes a filesystem entry, fetch its
  // stats to populate the body.  Hive entries use the existing Refresh
  // path (auto-fired below) to populate their database list.
  useEffect(() => {
    if (!selectedPlatform || selectedPlatform.kind !== "filesystem") {
      setFsStats(null);
      return;
    }
    fetch(`/api/filesystem-sources/${encodeURIComponent(selectedPlatform.id)}/stats`)
      .then((r) => r.json())
      .then((data: FsStats) => setFsStats(data))
      .catch(() => setFsStats(null));
  }, [selectedPlatformId]);

  // Auto-refresh the Hive database list when a Hive platform becomes
  // selected (on mount via persisted statusPlatformId, or when the
  // operator picks a new platform).  Keeps the panel populated without
  // demanding a Refresh click every navigation.
  useEffect(() => {
    if (selectedPlatform?.kind === "hive") {
      runConnectionRefresh();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPlatform?.id]);

  const runCredentialCheck = () => {
    setCredLoading(true);
    fetch("/api/agents/validate-credentials", { method: "POST" })
      .then((r) => r.json())
      .then(setCredentials)
      .catch((e) =>
        setCredentials({
          providers: {},
          any_valid: false,
          configured: [],
          error: String(e),
        }),
      )
      .finally(() => setCredLoading(false));
  };

  const runSmokeTest = () => {
    setSmokeLoading(true);
    fetch("/api/agents/smoke-test", { method: "POST" })
      .then((r) => r.json())
      .then((result: SmokeResult) =>
        setSmokeTest({ result, lastRunAt: Date.now() }),
      )
      .catch((e) =>
        setSmokeTest({
          result: { success: false, error: String(e) },
          lastRunAt: Date.now(),
        }),
      )
      .finally(() => setSmokeLoading(false));
  };

  const runConnectionRefresh = () => {
    if (!selectedConn) return;
    setConnLoading(true);
    // Fetch the probe result and the persisted data-source rows in
    // parallel.  The persisted rows are the source of truth for
    // ``vocab_uri`` — without hydrating from them, every refresh would
    // overwrite the operator's prior vocabulary assignment with a
    // ``${dbName}.annotations`` default and the Select would silently
    // drift away from the value PATCHed to the server.
    const probe = fetch(
      `/api/data-connections/${encodeURIComponent(selectedConn)}/refresh`,
      { method: "POST" },
    ).then((r) => r.json());
    const sources = fetch("/api/data-sources")
      .then((r) => r.json())
      .catch(() => ({ sources: [] }));
    Promise.all([probe, sources])
      .then(([data, srcResp]: [RefreshResult, { sources?: Array<{ id: string; vocab_uri?: string }> }]) => {
        setRefreshResult(data);
        if (!data.ok || !data.databases) return;

        const persistedByDb: Record<string, string> = {};
        for (const s of srcResp.sources ?? []) {
          // source_id format: "<connection>/<database>"
          const [conn, dbName] = (s.id || "").split("/", 2);
          if (conn === selectedConn && dbName) {
            persistedByDb[dbName] = s.vocab_uri ?? "";
          }
        }

        const enabled: Record<string, boolean> = {};
        const vocabs: Record<string, string> = {};
        for (const db of data.databases) {
          // Enabled iff a non-archived row exists for this db.
          enabled[db.name] = persistedByDb[db.name] !== undefined;
          // Hydration order: persisted server value > auto-default.
          if (persistedByDb[db.name] !== undefined) {
            vocabs[db.name] = persistedByDb[db.name];
          } else if (db.has_annotations) {
            vocabs[db.name] = `${db.name}.annotations`;
          }
        }
        setDbEnabled(enabled);
        setDbVocabUri(vocabs);
      })
      .catch((e) =>
        setRefreshResult({
          ok: false,
          connection: selectedConn,
          error: String(e),
        }),
      )
      .finally(() => setConnLoading(false));
  };

  return (
    <>
      <Title level={2}>System Status</Title>
      <Paragraph type="secondary">
        Infrastructure health and SDK connectivity dashboard.
      </Paragraph>

      {/* ── Infrastructure ─────────────────────────── */}
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          <Card
            title="Infrastructure"
            extra={
              <Button
                icon={<ReloadOutlined />}
                onClick={fetchStatus}
                loading={loading}
                size="small"
              >
                Refresh
              </Button>
            }
          >
            {loading && !status ? (
              <Spin />
            ) : status ? (
              <Descriptions column={1} bordered size="small">
                <Descriptions.Item label="gRPC Server">
                  <StatusBadge ok={status.grpc.ok} />
                  {status.grpc.latency_ms != null && (
                    <Text type="secondary" style={{ marginLeft: 8 }}>
                      {status.grpc.latency_ms}ms
                    </Text>
                  )}
                  {status.grpc.version && (
                    <Tag color="blue" style={{ marginLeft: 8 }}>
                      v{status.grpc.version}
                    </Tag>
                  )}
                  {status.grpc.error && (
                    <Text type="danger" style={{ marginLeft: 8 }}>
                      {status.grpc.error}
                    </Text>
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="PostgreSQL">
                  <StatusBadge ok={status.postgres.ok} />
                  {status.postgres.latency_ms != null && (
                    <Text type="secondary" style={{ marginLeft: 8 }}>
                      {status.postgres.latency_ms}ms
                    </Text>
                  )}
                  {status.postgres.error && (
                    <Text type="danger" style={{ marginLeft: 8 }}>
                      {status.postgres.error}
                    </Text>
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="Qdrant">
                  <StatusBadge ok={status.qdrant.ok} />
                  {status.qdrant.latency_ms != null && (
                    <Text type="secondary" style={{ marginLeft: 8 }}>
                      {status.qdrant.latency_ms}ms
                    </Text>
                  )}
                  {status.qdrant.error && (
                    <Text type="danger" style={{ marginLeft: 8 }}>
                      {status.qdrant.error}
                    </Text>
                  )}
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <Text type="danger">Unable to reach gateway</Text>
            )}
          </Card>
        </Col>

        <Col xs={24} lg={10}>
          <Card title="Configuration">
            {status?.config ? (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="Agent Model">
                  {status.config.model_discovery?.source === "bedrock_arn" ? (
                    <Tooltip title={status.config.agent_model}>
                      <Tag>{status.config.model_discovery.current_model}</Tag>
                    </Tooltip>
                  ) : (
                    <Tag>{status.config.agent_model}</Tag>
                  )}
                  {status.config.model_discovery?.upgrade_available && (
                    <Tag color="gold" style={{ marginLeft: 8 }}>
                      Upgrade: {status.config.model_discovery.latest_display_name ?? status.config.model_discovery.latest_model}
                    </Tag>
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="Anthropic API">
                  <Tag color={status.config.has_anthropic ? "green" : "default"}>
                    {status.config.has_anthropic ? "Configured" : "Not set"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="AWS Bedrock">
                  <Tag color={status.config.has_bedrock ? "green" : "default"}>
                    {status.config.has_bedrock ? "Configured" : "Not set"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Classify LLM">
                  <Tag color={status.config.has_classify_llm ? "green" : "orange"}>
                    {status.config.has_classify_llm ? "Configured" : "Not set (mock mode)"}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="Database">
                  <Space size={8} wrap>
                    <Text code>{status.config.db_url_masked}</Text>
                    <PGliteStatusChip
                      state={pglite}
                      onRestart={restartPglite}
                      restarting={pgliteRestarting}
                    />
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="Qdrant">
                  <Text code>
                    {status.config.qdrant_host}:{status.config.qdrant_http_port}
                  </Text>
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <Spin />
            )}
          </Card>
        </Col>
      </Row>

      {/* ── Classification Pipeline ─────────────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <ClassificationPipelineCard hasClassifyLlm={status?.config?.has_classify_llm} />
        </Col>
      </Row>

      {/* ── ML Artifacts (Extend Classification) ─── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <MLArtifactsCard />
        </Col>
      </Row>

      {/* ── Data Source + Versions ───────────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <DataSourceCard />
        </Col>
      </Row>

      {/* Reference Column Handling card removed: capability disabled —
          all columns (natural + reference) flow through the pipeline
          unconditionally.  Component definition retained below for
          history; not rendered. */}

      {/* ── SDK Validation ────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24} md={12}>
          <Card
            title="Credential Validation"
            extra={
              <Button
                icon={<SafetyCertificateOutlined />}
                onClick={runCredentialCheck}
                loading={credLoading}
                size="small"
              >
                Validate
              </Button>
            }
          >
            <Paragraph type="secondary" style={{ marginBottom: 12 }}>
              Tests each configured LLM provider with a minimal API call.
            </Paragraph>
            {credentials ? (
              credentials.error && !credentials.configured.length ? (
                <Text type="danger">{credentials.error}</Text>
              ) : (
                <Descriptions column={1} size="small" bordered>
                  {Object.entries(credentials.providers).map(([name, p]) => {
                    const tagColor = p.valid
                      ? "green"
                      : p.failure_kind === "network"
                        ? "orange"  // network unreachable — actionable via config
                        : "red";    // auth/permission/unknown — actionable via creds
                    const tagText = p.valid
                      ? "Valid"
                      : p.failure_kind === "network"
                        ? "Unreachable"
                        : p.failure_kind === "auth"
                          ? "Auth failed"
                          : p.failure_kind === "permission"
                            ? "Permission denied"
                            : "Invalid";
                    return (
                      <Descriptions.Item key={name} label={name}>
                        <Tag color={tagColor}>{tagText}</Tag>
                        {p.model && (
                          <Text type="secondary" style={{ marginLeft: 8 }}>
                            {p.model}
                          </Text>
                        )}
                        {p.error && (
                          <Paragraph
                            type={p.failure_kind === "network" ? "warning" : "danger"}
                            style={{ marginTop: 8, marginBottom: 0, whiteSpace: "pre-wrap" }}
                          >
                            {p.error}
                          </Paragraph>
                        )}
                      </Descriptions.Item>
                    );
                  })}
                </Descriptions>
              )
            ) : (
              <Text type="secondary">
                Click "Validate" to test provider credentials.
              </Text>
            )}
          </Card>
        </Col>

        <Col xs={24} md={12}>
          <Card
            title="SDK Smoke Test"
            extra={
              <Space>
                {smokeTest && (
                  <Tag
                    color={
                      Date.now() - smokeTest.lastRunAt > 10 * 60 * 1000
                        ? "orange"
                        : "cyan"
                    }
                  >
                    Last run {formatAgo(smokeTest.lastRunAt)}
                  </Tag>
                )}
                <Button
                  icon={<ThunderboltOutlined />}
                  onClick={runSmokeTest}
                  loading={smokeLoading}
                  size="small"
                >
                  {smokeTest ? "Re-run" : "Run"}
                </Button>
              </Space>
            }
          >
            <Paragraph type="secondary" style={{ marginBottom: 12 }}>
              Full Claude Agent SDK round-trip. Costs ~$0.02.
              {smokeTest &&
                Date.now() - smokeTest.lastRunAt > 24 * 60 * 60 * 1000 && (
                  <>
                    {" · "}
                    <Text type="warning">
                      Result is over 24 hours old — consider re-running.
                    </Text>
                  </>
                )}
            </Paragraph>
            {smoke ? (
              smoke.success ? (
                <Descriptions column={1} size="small">
                  <Descriptions.Item label="Status">
                    <Tag color="green">Success</Tag>
                    {smoke.retried && (
                      <Tag color="orange" style={{ marginLeft: 4 }}>Cold start retry</Tag>
                    )}
                  </Descriptions.Item>
                  <Descriptions.Item label="Reply">
                    {smoke.reply}
                  </Descriptions.Item>
                  {smoke.duration_ms != null && (
                    <Descriptions.Item label="Duration">
                      {smoke.duration_ms}ms
                    </Descriptions.Item>
                  )}
                  {smoke.session_id && (
                    <Descriptions.Item label="Session">
                      <Text code>{smoke.session_id}</Text>
                    </Descriptions.Item>
                  )}
                  {smoke.total_cost_usd != null && (
                    <Descriptions.Item label="Cost">
                      ${smoke.total_cost_usd.toFixed(4)}
                    </Descriptions.Item>
                  )}
                </Descriptions>
              ) : (
                <Text type="danger">{smoke.error}</Text>
              )
            ) : (
              <Text type="secondary">
                Click "Run" to execute a smoke test.
              </Text>
            )}
          </Card>
        </Col>
      </Row>

      {/* ── Web Terminal Agent model selector ─────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <WebTerminalAgentCard />
        </Col>
      </Row>

      {/* ── CAI Data Platform ────────────────────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <Card
            title="CAI Data Platform"
            extra={
              <Space>
                <Select
                  value={selectedPlatformId}
                  onChange={(v) => {
                    setSelectedPlatformId(v);
                    setRefreshResult(null);
                  }}
                  style={{ minWidth: 280 }}
                  placeholder="Select platform"
                  options={platforms.map((p) => ({
                    label: p.label,
                    value: p.id,
                  }))}
                  disabled={!platforms.length}
                  size="small"
                />
                <Button
                  icon={<ReloadOutlined />}
                  onClick={fetchPlatforms}
                  loading={platformsLoading}
                  size="small"
                  title="Re-fetch /api/data-platforms"
                >
                  Reload
                </Button>
                {selectedPlatform?.kind === "hive" && (
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={runConnectionRefresh}
                    loading={connLoading}
                    disabled={!selectedConn}
                    size="small"
                  >
                    Refresh
                  </Button>
                )}
              </Space>
            }
          >
            {(() => {
              const active = sources.find((s) => s.id === activeSourceId);
              if (!active) return null;
              let meta: { seeded_from_env?: boolean; connection?: string; database?: string } = {};
              try { meta = JSON.parse(active.metadata || "{}"); } catch { /* ignore */ }
              if (!meta?.seeded_from_env) return null;
              const conn = meta.connection ?? "";
              const db = meta.database ?? "";
              return (
                <Alert
                  type="info"
                  showIcon
                  style={{ marginBottom: 12 }}
                  message={
                    <Text>
                      Env defaults active: <Text strong>{conn}</Text> ·{" "}
                      <Text strong>{db}</Text>. Edit vocab or switch platforms
                      below; your changes persist.
                    </Text>
                  }
                />
              );
            })()}
            {selectedPlatform?.kind === "hive" && (
              <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                Probe databases via <Text code>cml.data_v1</Text>. Toggle
                databases on for classification and assign a vocabulary.
              </Paragraph>
            )}
            {selectedPlatform?.kind === "filesystem" && (
              <Paragraph type="secondary" style={{ marginBottom: 12 }}>
                Local filesystem mount. Annotations (when present) come from
                an <Text code>annotations.csv</Text> inside the mount —
                mirroring the <Text code>{"{db}.annotations"}</Text> table
                that backs Hive sources.
              </Paragraph>
            )}
            {platformsError && (
              <Alert
                type="error"
                showIcon
                style={{ marginBottom: 12 }}
                message="Failed to load data platforms"
                description={
                  <Space direction="vertical" size={6}>
                    <Text code style={{ fontSize: 12 }}>
                      GET /api/data-platforms — {platformsError}
                    </Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      The Reload button above re-probes the endpoint.  If the
                      error persists, check the gateway logs for{" "}
                      <Text code>list_data_platforms failed</Text>.
                    </Text>
                    {/* When pglite is the active backend and the supervisor
                        reports anything other than a clean running state,
                        offer a one-click respawn — most "Failed to load data
                        platforms" errors trace back to a transient pglite
                        blip.  The button is disabled (with a countdown)
                        while the supervisor is already in backoff so the
                        operator doesn't click repeatedly during automatic
                        recovery. */}
                    {pglite?.managed && pglite.state !== "running" && (
                      <Space size={6} wrap>
                        <Button
                          size="small"
                          danger
                          icon={<ReloadOutlined />}
                          loading={pgliteRestarting}
                          disabled={
                            pgliteRestarting ||
                            pglite.state === "starting" ||
                            (pglite.state === "backoff" &&
                              !!pglite.backoff_until &&
                              pglite.backoff_until > Date.now() / 1000)
                          }
                          onClick={restartPglite}
                          title={
                            pglite.state === "circuit_broken"
                              ? "Supervisor is in circuit-broken state — click to clear and respawn."
                              : pglite.state === "backoff"
                              ? "Supervisor is already retrying; the countdown shows the next attempt."
                              : "Touch the supervisor restart sentinel to respawn pglite immediately."
                          }
                        >
                          Restart database
                        </Button>
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          supervisor: {pglite.state}
                        </Text>
                        {pglite.state === "backoff" && (
                          <CountdownText until={pglite.backoff_until} />
                        )}
                        {pglite.state === "starting" && (
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            supervisor restarting it now…
                          </Text>
                        )}
                      </Space>
                    )}
                  </Space>
                }
              />
            )}
            {!platformsError && platforms.length === 0 && (
              <Alert
                type="warning"
                showIcon
                message="No data platforms returned"
                description={
                  <Space direction="vertical" size={4}>
                    <Text>
                      The endpoint succeeded but returned an empty list.
                      Other panels (Classification Pipeline, ML Artifacts)
                      may be reading from{" "}
                      <Text code>/api/data-sources</Text> directly, so a
                      mismatch here points at{" "}
                      <Text code>list_data_platforms</Text>'s consolidator
                      logic rather than the underlying DB.
                    </Text>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Configure Hive via{" "}
                      <Text code>ATELIER_DATA_CONNECTIONS</Text> or mount a
                      local directory via{" "}
                      <Text code>ATELIER_META_TAGGING_DIR</Text> if neither
                      is set.  Otherwise paste this in the Application pod's
                      terminal to compare endpoints:
                    </Text>
                    <Text code style={{ fontSize: 12, whiteSpace: "pre" }}>
                      {"curl -s http://127.0.0.1:8090/api/data-platforms | python3 -m json.tool\n"}
                      {"curl -s http://127.0.0.1:8090/api/data-sources   | python3 -m json.tool\n"}
                      {"curl -s http://127.0.0.1:8090/api/data-connections | python3 -m json.tool"}
                    </Text>
                  </Space>
                }
              />
            )}
            {selectedPlatform?.kind === "hive" && refreshResult && !refreshResult.ok && (
              <Text type="danger">{refreshResult.error}</Text>
            )}
            {selectedPlatform?.kind === "filesystem" && fsStats && (
              <Table
                size="small"
                pagination={false}
                rowKey="source_id"
                dataSource={[fsStats]}
                columns={[
                  {
                    title: "Mount",
                    dataIndex: "mount",
                    key: "mount",
                    render: (v: string | null) => (
                      <Text code style={{ fontSize: 12 }}>
                        {v ?? "—"}
                      </Text>
                    ),
                  },
                  {
                    title: "Tables",
                    dataIndex: "table_count",
                    key: "table_count",
                    width: 80,
                    render: (v: number | null) =>
                      v?.toLocaleString() ?? <Text type="secondary">—</Text>,
                  },
                  {
                    title: "Columns",
                    dataIndex: "column_count",
                    key: "column_count",
                    width: 90,
                    render: (v: number | null) =>
                      v?.toLocaleString() ?? <Text type="secondary">—</Text>,
                  },
                  {
                    title: "Vocabulary",
                    key: "vocab",
                    render: (_: unknown, row: FsStats) => {
                      if (!row.vocab_uri) {
                        return (
                          <Text type="secondary" style={{ fontSize: 12 }}>
                            bundled / universal
                          </Text>
                        );
                      }
                      const label = row.vocab_uri.replace(/^file:\/\//, "");
                      const count = row.annotation_count;
                      return (
                        <Space size={6}>
                          <Text code style={{ fontSize: 12 }}>
                            {label}
                          </Text>
                          {count != null && (
                            <Text type="secondary" style={{ fontSize: 11 }}>
                              ({count} rows)
                            </Text>
                          )}
                        </Space>
                      );
                    },
                  },
                ]}
              />
            )}
            {selectedPlatform?.kind === "hive" && refreshResult?.ok && refreshResult.databases && (() => {
              const dbs = refreshResult.databases!;
              // Build vocab options from databases that have annotations
              const vocabOptions = dbs
                .filter((d) => d.has_annotations)
                .map((d) => ({
                  label: `${d.name}.annotations (${d.annotation_count})`,
                  value: `${d.name}.annotations`,
                }));
              vocabOptions.unshift({ label: "— none —", value: "" });

              const handleToggle = (dbName: string, checked: boolean) => {
                setDbEnabled((prev) => ({ ...prev, [dbName]: checked }));
                // Compute the vocab_uri to send NOW — reading from the
                // ``dbVocabUri`` state object alone races against the
                // setDbVocabUri below (React batches updates), so on the
                // first toggle-on the POST would otherwise carry an empty
                // ``vocab_uri`` and the operator's default would not
                // persist.
                const db = dbs.find((d) => d.name === dbName);
                let vocabForRequest = dbVocabUri[dbName] || "";
                if (checked && !vocabForRequest && db?.has_annotations) {
                  vocabForRequest = `${dbName}.annotations`;
                  setDbVocabUri((prev) => ({ ...prev, [dbName]: vocabForRequest }));
                }
                const sourceId = `${selectedConn}/${dbName}`;
                if (checked) {
                  fetch("/api/data-sources", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      source_id: sourceId,
                      source_type: "hive",
                      display_name: `Hive: ${sourceId}`,
                      vocab_uri: vocabForRequest,
                    }),
                  })
                    .then((r) => r.json())
                    .then((j) => {
                      if (j?.error) {
                        message.error(`Could not enable ${dbName}: ${j.error}`);
                      }
                    })
                    .catch((e) =>
                      message.error(`Could not enable ${dbName}: ${e}`),
                    );
                } else {
                  fetch(`/api/data-sources/${encodeURIComponent(sourceId)}/archive`, {
                    method: "POST",
                  })
                    .then((r) => r.json())
                    .then((j) => {
                      if (j?.error) {
                        message.error(`Could not disable ${dbName}: ${j.error}`);
                      }
                    })
                    .catch((e) =>
                      message.error(`Could not disable ${dbName}: ${e}`),
                    );
                }
              };

              const handleVocabChange = (dbName: string, uri: string) => {
                setDbVocabUri((prev) => ({ ...prev, [dbName]: uri }));
                const sourceId = `${selectedConn}/${dbName}`;
                fetch(`/api/data-sources/${encodeURIComponent(sourceId)}`, {
                  method: "PATCH",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ vocab_uri: uri }),
                })
                  .then((r) => r.json())
                  .then((j) => {
                    if (j?.error) {
                      message.error(`Could not save vocabulary for ${dbName}: ${j.error}`);
                    } else {
                      message.success(
                        `Vocabulary for ${dbName} → ${uri || "(none)"}`,
                      );
                    }
                  })
                  .catch((e) =>
                    message.error(`Could not save vocabulary for ${dbName}: ${e}`),
                  );
              };

              return (
                <>
                  {refreshResult.latency_ms != null && (
                    <div style={{ marginBottom: 8, fontSize: 12, color: "#8c8c8c" }}>
                      {dbs.length} databases discovered in {refreshResult.latency_ms}ms
                    </div>
                  )}
                  <Table
                    size="small"
                    pagination={false}
                    dataSource={dbs.map((d) => ({ key: d.name, ...d }))}
                    columns={[
                      {
                        title: "Database",
                        dataIndex: "name",
                        key: "name",
                        render: (v: string) => <Text strong>{v}</Text>,
                      },
                      {
                        title: "Tables",
                        dataIndex: "table_count",
                        key: "table_count",
                        width: 80,
                        render: (v: number) => v?.toLocaleString() ?? "—",
                      },
                      {
                        title: "Entities",
                        key: "entities",
                        width: 80,
                        render: () => <Text type="secondary">—</Text>,
                      },
                      {
                        title: "Enabled",
                        key: "enabled",
                        width: 80,
                        render: (_: unknown, row: DatabaseInfo) => (
                          <Switch
                            size="small"
                            checked={dbEnabled[row.name] ?? false}
                            onChange={(checked) => handleToggle(row.name, checked)}
                          />
                        ),
                      },
                      {
                        title: "Vocabulary",
                        key: "vocab_uri",
                        render: (_: unknown, row: DatabaseInfo) => (
                          <Select
                            size="small"
                            style={{ width: "100%" }}
                            value={dbVocabUri[row.name] ?? ""}
                            onChange={(v) => handleVocabChange(row.name, v)}
                            disabled={!dbEnabled[row.name]}
                            options={vocabOptions}
                          />
                        ),
                      },
                    ]}
                  />
                </>
              );
            })()}
          </Card>
        </Col>
      </Row>
    </>
  );
}
