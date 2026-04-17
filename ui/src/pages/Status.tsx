import { useEffect, useState } from "react";
import {
  Badge,
  Button,
  Card,
  Col,
  Descriptions,
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
  ArrowLeftOutlined,
  EyeOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Link } from "react-router-dom";
import { useDataset } from "../contexts/DatasetContext";

const { Title, Paragraph, Text } = Typography;

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

function ClassificationPipelineCard({ hasClassifyLlm }: { hasClassifyLlm?: boolean }) {
  const [fsm, setFsm] = useState<FSMStatus | null>(null);
  const [fsmLoading, setFsmLoading] = useState(false);
  const [starting, setStarting] = useState(false);
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
          <Button
            type="primary"
            onClick={startPipeline}
            loading={starting}
            disabled={isRunning}
            size="small"
          >
            {isRunning ? "Running..." : "Start Classification"}
          </Button>
        </Space>
      }
    >
      <Descriptions column={2} size="small" bordered>
        <Descriptions.Item label="State">
          <Tag color={FSM_STATE_COLORS[state] ?? "default"}>{state}</Tag>
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
        {progress.tables_discovered != null && (
          <Descriptions.Item label="Tables">
            {String(progress.tables_discovered)}
          </Descriptions.Item>
        )}
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
      </Descriptions>
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
            onChange={(v) => setActiveSourceId(v ?? null)}
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

function StatusBadge({ ok }: { ok: boolean }) {
  return ok ? (
    <Badge status="success" text="Healthy" />
  ) : (
    <Badge status="error" text="Unreachable" />
  );
}

export default function Status() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [loading, setLoading] = useState(true);

  const [credentials, setCredentials] = useState<CredentialResult | null>(null);
  const [credLoading, setCredLoading] = useState(false);

  const [smoke, setSmoke] = useState<SmokeResult | null>(null);
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
  const [selectedPlatformId, setSelectedPlatformId] = useState<string | undefined>();
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
    fetch("/api/data-platforms")
      .then((r) => r.json())
      .then((data) => {
        const list: Platform[] = Array.isArray(data?.platforms)
          ? data.platforms
          : [];
        setPlatforms(list);
        if (list.length > 0) setSelectedPlatformId(list[0].id);
      })
      .catch(() => setPlatforms([]));
  }, []);

  // When the selected platform becomes a filesystem entry, fetch its
  // stats to populate the body.  Hive entries use the existing Refresh
  // path and don't touch fsStats.
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
      .then(setSmoke)
      .catch((e) => setSmoke({ success: false, error: String(e) }))
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
      <div style={{ marginBottom: 16 }}>
        <Link to="/">
          <Button icon={<ArrowLeftOutlined />} size="small">
            Back
          </Button>
        </Link>
      </div>
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
                  {Object.entries(credentials.providers).map(([name, p]) => (
                    <Descriptions.Item key={name} label={name}>
                      <Tag color={p.valid ? "green" : "red"}>
                        {p.valid ? "Valid" : "Invalid"}
                      </Tag>
                      {p.model && (
                        <Text type="secondary" style={{ marginLeft: 8 }}>
                          {p.model}
                        </Text>
                      )}
                      {p.error && (
                        <Text type="danger" style={{ marginLeft: 8 }}>
                          {p.error}
                        </Text>
                      )}
                    </Descriptions.Item>
                  ))}
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
              <Button
                icon={<ThunderboltOutlined />}
                onClick={runSmokeTest}
                loading={smokeLoading}
                size="small"
              >
                Run
              </Button>
            }
          >
            <Paragraph type="secondary" style={{ marginBottom: 12 }}>
              Full Claude Agent SDK round-trip. Costs ~$0.02.
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
