import { Card, Col, Row, Statistic, Tag, Typography } from "antd";
import {
  ApiOutlined,
  ClusterOutlined,
  DotChartOutlined,
  ExperimentOutlined,
} from "@ant-design/icons";
import { useEffect, useState } from "react";

const { Title, Paragraph } = Typography;

interface HealthStatus {
  status: string;
  version: string;
}

function Landing() {
  const [health, setHealth] = useState<HealthStatus | null>(null);

  useEffect(() => {
    fetch("/api/health")
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth(null));
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
              value={0}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="Datasets"
              value={0}
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
            title="Atlas Viewer"
            extra={<DotChartOutlined />}
            hoverable
            style={{ height: "100%" }}
          >
            <Paragraph type="secondary">
              Interactive embedding visualization powered by embedding-atlas.
              Explore classification results from the signals pipeline.
            </Paragraph>
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
          </Card>
        </Col>
      </Row>
    </>
  );
}

export default Landing;
