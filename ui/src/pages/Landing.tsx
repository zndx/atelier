import { Card, Col, Row, Spin, Statistic, Tag, Typography } from "antd";
import {
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClusterOutlined,
  CodeOutlined,
  DotChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { lazy, Suspense, useEffect, useState } from "react";
import { Link } from "react-router-dom";

const Terminal = lazy(() => import("../components/Terminal"));

const { Title, Paragraph } = Typography;

interface StatusSummary {
  grpc: { ok: boolean; version?: string };
  postgres: { ok: boolean };
  qdrant: { ok: boolean };
  connected: boolean;
}

interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  row_count: number;
}

function Landing() {
  const [status, setStatus] = useState<StatusSummary | null>(null);
  const [agents, setAgents] = useState<unknown[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then(setStatus)
      .catch(() => setStatus(null));

    fetch("/api/agents")
      .then((r) => r.json())
      .then((data) => setAgents(data.agents || []))
      .catch(() => setAgents([]));

    fetch("/api/datasets")
      .then((r) => r.json())
      .then((data) => setDatasets(data.datasets || []))
      .catch(() => setDatasets([]));
  }, []);

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
                value={status?.connected ? "Connected" : "Disconnected"}
                prefix={
                  status?.connected ? (
                    <CheckCircleOutlined />
                  ) : (
                    <CloseCircleOutlined />
                  )
                }
                valueStyle={{
                  color: status?.connected ? "#52c41a" : "#ff4d4f",
                }}
              />
              {status?.grpc?.version && (
                <Tag color="blue" style={{ marginTop: 8 }}>
                  v{status.grpc.version}
                </Tag>
              )}
            </Card>
          </Link>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Keystone Agents"
              value={agents.length}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Datasets"
              value={datasets.length}
              prefix={<DotChartOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Link to="/workflows">
            <Card hoverable>
              <Statistic
                title="Workflows"
                value="Canvas"
                prefix={<ClusterOutlined />}
                suffix={
                  <Tag color="blue" style={{ marginLeft: 8, fontSize: 11 }}>
                    preview
                  </Tag>
                }
              />
            </Card>
          </Link>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={12} lg={8}>
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
        </Col>
        <Col xs={24} md={12} lg={8}>
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
            {datasets.length > 0 && (
              <div style={{ marginTop: 12 }}>
                {datasets.map((ds) => (
                  <Link key={ds.id} to={`/embeddings/${ds.id}`}>
                    <Tag color="blue" style={{ cursor: "pointer", marginBottom: 4 }}>
                      {ds.name} ({ds.row_count} rows)
                    </Tag>
                  </Link>
                ))}
              </div>
            )}
          </Card>
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
