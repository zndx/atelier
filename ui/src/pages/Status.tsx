import { useEffect, useState } from "react";
import {
  Alert,
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
  message,
  Progress,
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
  EyeOutlined,
  ReloadOutlined,
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
  error?: string | null;
  result_path?: string | null;
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
  k_threshold_met:
    "Mean DST conflict K fell below threshold (honoring min_iterations floor).",
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
  const { activeSourceId, refreshDatasets } = useDataset();

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

  // Refresh datasets when pipeline converges so the new dataset appears
  useEffect(() => {
    if (state === "CONVERGED") {
      refreshDatasets();
    }
  }, [state, refreshDatasets]);

  return (
    <Card
      title="Classification Pipeline"
      extra={
        <Space>
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
              title={String(
                CONVERGENCE_REASON_DESCRIPTIONS[
                  String(progress.convergence_reason)
                ] ?? progress.convergence_reason,
              )}
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
        {progress.mean_k != null && (
          <Descriptions.Item label="Mean K (conflict)">
            {Number(progress.mean_k).toFixed(3)}
          </Descriptions.Item>
        )}
        {progress.disagreements != null && (
          <Descriptions.Item label="Disagreements">
            {String(progress.disagreements)}
          </Descriptions.Item>
        )}
        {progress.iteration != null && (
          <Descriptions.Item label="Iteration">
            {String(progress.iteration)}
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

function DataSourceCard() {
  const {
    sources,
    activeSourceId,
    setActiveSourceId,
    datasets,
    activeDatasetId,
    setActiveDatasetId,
  } = useDataset();

  const activateVersion = (datasetId: string) => {
    fetch(`/api/datasets/${encodeURIComponent(datasetId)}/activate`, {
      method: "POST",
    })
      .then((r) => r.json())
      .then(() => setActiveDatasetId(datasetId))
      .catch(() => {});
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
              title: "Version",
              dataIndex: "version_number",
              key: "version",
              width: 80,
              render: (v: number, record: { id: string; is_active: boolean }) => (
                <Space>
                  <Text strong>v{v}</Text>
                  {record.is_active && <Tag color="blue">active</Tag>}
                </Space>
              ),
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
              key: "overwatch",
              width: 90,
              render: (_: unknown, record: any) =>
                record.fsm_run_id ? (
                  <Link to={`/overwatch/${record.fsm_run_id}`}>
                    <Tag color="purple" style={{ cursor: "pointer", margin: 0 }}>
                      Overwatch
                    </Tag>
                  </Link>
                ) : null,
            },
          ]}
        />
      )}
    </Card>
  );
}

function ReferenceColumnHandlingCard() {
  // UAT-compatibility toggle.  The synth corpus UAT uses pairs every
  // natural-named column with an answer-key twin named attr_*, code_*,
  // etc. — the numeric suffix literally IS the code.  Production
  // column naming doesn't hit this regex, so the toggle is a no-op on
  // real data.  Kept visible (not buried in Settings) because UAT
  // reviewers want to see accuracy in both configurations.  This
  // whole card — along with the backend flag — is slated for removal
  // once the synth-dataset lineage retires.
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  const load = () => {
    setLoading(true);
    fetch("/api/settings")
      .then((r) => r.json())
      .then((body) => {
        const values = body?.values ?? {};
        const v = values["classify_exclude_reference_columns"];
        setEnabled(typeof v === "boolean" ? v : true);
      })
      .catch(() => setEnabled(true))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    load();
  }, []);

  const toggle = async (next: boolean) => {
    setSaving(true);
    try {
      const r = await fetch("/api/settings", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ classify_exclude_reference_columns: next }),
      });
      const body = await r.json();
      if (!r.ok || body.error) {
        message.error(body.error || `PATCH failed: ${r.status}`);
      } else {
        setEnabled(next);
        message.success(
          next
            ? "Reference columns will be excluded on the next run."
            : "Reference columns will be included on the next run.",
        );
      }
    } catch (e) {
      message.error(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <Card
      title={
        <Space>
          <span>Reference Column Handling</span>
          <Tag color="gold" style={{ margin: 0 }}>
            UAT compatibility
          </Tag>
        </Space>
      }
      extra={
        <Space>
          {loading ? (
            <Spin size="small" />
          ) : (
            <>
              <Text type="secondary">
                {enabled ? "Excluded" : "Included"}
              </Text>
              <Switch
                checked={enabled ?? true}
                onChange={toggle}
                loading={saving}
                disabled={loading}
              />
            </>
          )}
        </Space>
      }
    >
      <Paragraph style={{ marginBottom: 8 }}>
        The UAT synth corpus pairs every natural-named column with an
        answer-key twin (pattern below) whose numeric suffix literally
        encodes the expected code. This toggle controls whether those
        twins enter the classification pipeline. Nowhere in the
        prediction path does the pipeline regex-decode the name; the
        toggle is strictly a pre-filter over the sample set.
      </Paragraph>
      <Paragraph style={{ marginBottom: 8 }}>
        <Text code>
          ^(attr|code|col|data|field|item|key|ref|val|var)_\d+(_\d+)*$
        </Text>
      </Paragraph>
      <Alert
        type={enabled ? "info" : "warning"}
        showIcon
        message={
          enabled
            ? "Exclude mode (production default)"
            : "Include mode"
        }
        description={
          enabled ? (
            <>
              Answer-key columns such as{" "}
              <Text code>attr_1_1_1_9_2_1</Text> are filtered out of
              the sample set before the LLM sweep. On production data
              the regex matches nothing, so this is a no-op there.
            </>
          ) : (
            <>
              Answer-key columns flow through the full classifier (LLM
              + cosine + CatBoost + SVM + DST fusion) as ordinary
              inputs and may influence classification results for
              adjacent columns.
            </>
          )
        }
        style={{ marginBottom: 0 }}
      />
    </Card>
  );
}

function StatusBadge({ ok }: { ok: boolean }) {
  return ok ? (
    <Badge status="success" text="Healthy" />
  ) : (
    <Badge status="error" text="Unreachable" />
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

  useEffect(() => {
    fetchStatus();
    // Auto-fire credential validation on every /status visit — cheap
    // probe ($0 vs ~$0.007 for the smoke test) so it's worth keeping
    // current state on screen without a manual button press.
    runCredentialCheck();
    fetch("/api/data-platforms")
      .then((r) => r.json())
      .then((data) => {
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
      .catch(() => setPlatforms([]));
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
    fetch(`/api/data-connections/${encodeURIComponent(selectedConn)}/refresh`, {
      method: "POST",
    })
      .then((r) => r.json())
      .then((data: RefreshResult) => {
        setRefreshResult(data);
        if (data.ok && data.databases) {
          // Auto-enable databases that have annotations; auto-select their vocab_uri
          const enabled: Record<string, boolean> = {};
          const vocabs: Record<string, string> = {};
          for (const db of data.databases) {
            enabled[db.name] = db.has_annotations;
            if (db.has_annotations) {
              vocabs[db.name] = `${db.name}.annotations`;
            }
          }
          setDbEnabled(enabled);
          setDbVocabUri(vocabs);
        }
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
                  <Text code>{status.config.db_url_masked}</Text>
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

      {/* ── Data Source + Versions ───────────────── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <DataSourceCard />
        </Col>
      </Row>

      {/* ── Reference Column Handling (UAT compatibility knob) ── */}
      <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
        <Col xs={24}>
          <ReferenceColumnHandlingCard />
        </Col>
      </Row>

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
            {platforms.length === 0 && (
              <Text type="secondary">
                No data platforms registered. Configure Hive via{" "}
                <Text code>ATELIER_DATA_CONNECTIONS</Text> or mount a local
                directory via <Text code>ATELIER_META_TAGGING_DIR</Text>.
              </Text>
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
                if (checked && !dbVocabUri[dbName]) {
                  // Auto-pick own annotations if available
                  const db = dbs.find((d) => d.name === dbName);
                  if (db?.has_annotations) {
                    setDbVocabUri((prev) => ({ ...prev, [dbName]: `${dbName}.annotations` }));
                  }
                }
                // Create/archive data source
                const sourceId = `${selectedConn}/${dbName}`;
                if (checked) {
                  fetch("/api/data-sources", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      source_id: sourceId,
                      source_type: "hive",
                      display_name: `Hive: ${sourceId}`,
                      vocab_uri: dbVocabUri[dbName] || "",
                    }),
                  }).catch(() => {});
                } else {
                  fetch(`/api/data-sources/${encodeURIComponent(sourceId)}/archive`, {
                    method: "POST",
                  }).catch(() => {});
                }
              };

              const handleVocabChange = (dbName: string, uri: string) => {
                setDbVocabUri((prev) => ({ ...prev, [dbName]: uri }));
                const sourceId = `${selectedConn}/${dbName}`;
                fetch(`/api/data-sources/${encodeURIComponent(sourceId)}`, {
                  method: "PATCH",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ vocab_uri: uri }),
                }).catch(() => {});
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
