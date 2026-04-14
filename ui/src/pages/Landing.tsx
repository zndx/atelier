import { Card, Col, Row, Spin, Statistic, Tag, Typography } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  BookOutlined,
  ClusterOutlined,
  CodeOutlined,
  DatabaseOutlined,
  DotChartOutlined,
  ExperimentOutlined,
  ToolOutlined,
} from "@ant-design/icons";
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import type { AgentInfo } from "../types/canvas";
import { useDataset } from "../contexts/DatasetContext";

const Terminal = lazy(() => import("../components/Terminal"));

const { Title, Paragraph } = Typography;

interface StatusSummary {
  grpc: { ok: boolean; version?: string };
  postgres: { ok: boolean };
  qdrant: { ok: boolean };
  connected: boolean;
  degraded?: boolean;
}

function Landing() {
  const [status, setStatus] = useState<StatusSummary | null>(null);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [termCount, setTermCount] = useState<number | null>(null);
  const { activeDatasetId, activeSourceId, datasets, sources } = useDataset();

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null));

    fetch("/api/agents")
      .then((r) => r.json())
      .then((data) => setAgents(data.agents || []))
      .catch(() => setAgents([]));

    const vocabParams = activeSourceId
      ? `?source_id=${encodeURIComponent(activeSourceId)}`
      : "";
    fetch(`/api/vocabulary/stats${vocabParams}`)
      .then((r) => r.json())
      .then((data) => setTermCount(data.terms ?? null))
      .catch(() => setTermCount(null));
  }, [activeSourceId]);

  const skillCount = useMemo(
    () => agents.reduce((n, a) => n + (a.tool_ids?.length || 0), 0),
    [agents],
  );

  // Entities = tables (datasets) + columns (row_count per dataset).
  // Aligns with Apache Atlas where both hive_table and hive_column are
  // first-class entity types that can be independently tagged with Terms.
  const entityCount = useMemo(
    () =>
      datasets.reduce((n, d) => n + (d.row_count || 0), 0) + datasets.length,
    [datasets],
  );

  return (
    <>
      <Title level={2}>Welcome to Atelier</Title>
      <Paragraph type="secondary">
        Agentic classification workbench powered by the Claude Agent SDK with
        interactive embedding visualization and adaptive keystone-agent
        orchestration.
      </Paragraph>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Link to="/status">
            <Card hoverable>
              <Statistic
                title="Service Status"
                value={
                  status?.connected
                    ? status.degraded
                      ? "Degraded"
                      : "Connected"
                    : "Disconnected"
                }
                prefix={
                  status?.connected ? (
                    status.degraded ? (
                      <WarningOutlined />
                    ) : (
                      <CheckCircleOutlined />
                    )
                  ) : (
                    <CloseCircleOutlined />
                  )
                }
                valueStyle={{
                  color: status?.connected
                    ? status.degraded
                      ? "#faad14"
                      : "#52c41a"
                    : "#ff4d4f",
                }}
              />
              {status?.grpc?.version && (
                <Tag color="blue" style={{ marginTop: 8 }}>
                  v{status.grpc.version}
                </Tag>
              )}
              {status?.degraded && (
                <div style={{ marginTop: 4, fontSize: 11, color: "#8c8c8c" }}>
                  {[
                    status.postgres?.ok ? null : "postgres",
                    status.qdrant?.ok ? null : "qdrant",
                  ]
                    .filter(Boolean)
                    .join(", ")}{" "}
                  unreachable
                </div>
              )}
            </Card>
          </Link>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Skills"
              value={skillCount}
              prefix={<ToolOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Entities"
              value={entityCount}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Terms"
              value={termCount ?? "—"}
              prefix={<BookOutlined />}
              suffix={
                sources.length > 1 ? (
                  <span style={{ fontSize: 12, color: "#8c8c8c" }}>
                    {sources.length} sources
                  </span>
                ) : null
              }
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={12} lg={8}>
          <Link to="/agents">
            <Card
              title="Agents"
              extra={<ExperimentOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Define and orchestrate keystone agents using the Claude Agent SDK.
                Agents adapt as classification workflows evolve.
              </Paragraph>
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Link to={activeDatasetId ? `/embeddings/${activeDatasetId}` : "/embeddings"}>
            <Card
              title="Embeddings"
              extra={<DotChartOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Interactive visualization of classification embeddings.
                Explore results from the signals pipeline.
              </Paragraph>
              {(() => {
                const active = datasets.find((d) => d.id === activeDatasetId);
                return active ? (
                  <div style={{ marginTop: 12 }}>
                    <Tag color="blue">
                      {active.name} ({active.row_count.toLocaleString()} rows)
                    </Tag>
                  </div>
                ) : null;
              })()}
            </Card>
          </Link>
        </Col>
        <Col xs={24} md={12} lg={8}>
          <Link to="/workflows">
            <Card
              title="Workflows"
              extra={<ClusterOutlined />}
              hoverable
              style={{ height: "100%" }}
            >
              <Paragraph type="secondary">
                Orchestration canvas — situational awareness for Claude's
                dynamic keystone agent coordination.
              </Paragraph>
              <Tag color="blue" style={{ marginTop: 8 }}>
                Preview
              </Tag>
            </Card>
          </Link>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col span={24}>
          <Card
            title={
              <span>
                <CodeOutlined style={{ marginRight: 8 }} />
                Terminal
              </span>
            }
            extra={
              <Tag
                color="geekblue"
                style={{ margin: 0, fontSize: 11 }}
              >
                Claude Agent SDK
              </Tag>
            }
            styles={{
              body: {
                padding: 0,
                background: "#0d1117",
                borderRadius: "0 0 6px 6px",
              },
            }}
          >
            <Suspense
              fallback={
                <div
                  style={{
                    height: 340,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "#0d1117",
                  }}
                >
                  <Spin />
                </div>
              }
            >
              <Terminal style={{ height: 340 }} />
            </Suspense>
          </Card>
        </Col>
      </Row>
    </>
  );
}

export default Landing;
