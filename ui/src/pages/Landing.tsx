import { Card, Col, Row, Statistic, Tag, Typography } from "antd";
import {
  ApiOutlined,
  ClusterOutlined,
  DotChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

const { Title, Paragraph } = Typography;

interface HealthStatus {
  status: string;
  version: string;
}

interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  row_count: number;
}

function Landing() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [agents, setAgents] = useState<unknown[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));

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
          <Card>
            <Statistic
              title="Service Status"
              value={health ? "Online" : "Offline"}
              prefix={<ApiOutlined />}
              valueStyle={{ color: health ? "#52c41a" : "#ff4d4f" }}
            />
            {health && (
              <Tag color="blue" style={{ marginTop: 8 }}>
                v{health.version}
              </Tag>
            )}
          </Card>
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
          <Card>
            <Statistic
              title="Workflows"
              value={0}
              prefix={<ClusterOutlined />}
              suffix={
                <Tag color="default" style={{ marginLeft: 8, fontSize: 11 }}>
                  planned
                </Tag>
              }
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} md={8}>
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
        <Col xs={24} md={8}>
          <Card
            title="Embeddings Viewer"
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
        <Col xs={24} md={8}>
          <Card
            title="Workflows"
            extra={<ClusterOutlined />}
            hoverable
            style={{ height: "100%" }}
          >
            <Paragraph type="secondary">
              XYFlow canvas illustrating relationships between keystone agents
              and their evolution over time.
            </Paragraph>
            <Tag color="default" style={{ marginTop: 8 }}>
              Planned
            </Tag>
          </Card>
        </Col>
      </Row>
    </>
  );
}

export default Landing;
