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
  Table,
  Tag,
  Typography,
} from "antd";
import {
  ArrowLeftOutlined,
  DatabaseOutlined,
  ReloadOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";
import { Link } from "react-router-dom";

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
  error?: string;
}

type CellValue = string | number | boolean | null;

interface DataConnectionResult {
  ok: boolean;
  connection: string;
  query?: string;
  row_count?: number;
  columns?: string[];
  rows?: CellValue[][];
  latency_ms?: number;
  error?: string;
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

  const [connections, setConnections] = useState<string[]>([]);
  const [selectedConn, setSelectedConn] = useState<string | undefined>();
  const [connResult, setConnResult] = useState<DataConnectionResult | null>(
    null,
  );
  const [connLoading, setConnLoading] = useState(false);

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
    fetch("/api/data-connections")
      .then((r) => r.json())
      .then((data) => {
        const list: string[] = Array.isArray(data?.connections)
          ? data.connections
          : [];
        setConnections(list);
        if (list.length > 0) setSelectedConn(list[0]);
      })
      .catch(() => setConnections([]));
  }, []);

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

  const runConnectionTest = () => {
    if (!selectedConn) return;
    setConnLoading(true);
    fetch(`/api/data-connections/${encodeURIComponent(selectedConn)}/test`, {
      method: "POST",
    })
      .then((r) => r.json())
      .then(setConnResult)
      .catch((e) =>
        setConnResult({
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
                  <Tag>{status.config.agent_model}</Tag>
                  {status.config.model_discovery?.source === "bedrock_arn" && (
                    <Text type="secondary" style={{ marginLeft: 4, fontSize: 12 }}>
                      ({status.config.model_discovery.current_model})
                    </Text>
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
                  value={selectedConn}
                  onChange={setSelectedConn}
                  style={{ minWidth: 240 }}
                  placeholder="Select connection"
                  options={connections.map((c) => ({ label: c, value: c }))}
                  disabled={!connections.length}
                  size="small"
                />
                <Button
                  icon={<DatabaseOutlined />}
                  onClick={runConnectionTest}
                  loading={connLoading}
                  disabled={!selectedConn}
                  size="small"
                >
                  Test
                </Button>
              </Space>
            }
          >
            <Paragraph type="secondary" style={{ marginBottom: 12 }}>
              Runs <Text code>show databases</Text> against the selected CAI
              Data Connection via <Text code>cml.data_v1</Text>.
            </Paragraph>
            {connections.length === 0 && (
              <Text type="secondary">
                No connections configured. Set{" "}
                <Text code>ATELIER_DATA_CONNECTIONS</Text> (comma-separated) at
                deploy time.
              </Text>
            )}
            {connResult && connResult.ok ? (
              <>
                <Descriptions
                  column={2}
                  size="small"
                  style={{ marginBottom: 12 }}
                >
                  <Descriptions.Item label="Status">
                    <Tag color="green">Success</Tag>
                  </Descriptions.Item>
                  <Descriptions.Item label="Rows">
                    {connResult.row_count}
                  </Descriptions.Item>
                  <Descriptions.Item label="Latency">
                    {connResult.latency_ms}ms
                  </Descriptions.Item>
                  <Descriptions.Item label="Query">
                    <Text code>{connResult.query}</Text>
                  </Descriptions.Item>
                </Descriptions>
                <Table
                  size="small"
                  pagination={false}
                  dataSource={(connResult.rows ?? []).map((row, i) => ({
                    key: i,
                    ...Object.fromEntries(
                      (connResult.columns ?? []).map((c, j) => [c, row[j]]),
                    ),
                  }))}
                  columns={(connResult.columns ?? []).map((c) => ({
                    title: c,
                    dataIndex: c,
                    key: c,
                  }))}
                />
              </>
            ) : connResult ? (
              <Text type="danger">{connResult.error}</Text>
            ) : null}
          </Card>
        </Col>
      </Row>
    </>
  );
}
