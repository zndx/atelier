import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Card, Col, Empty, Row, Spin, Tag, Typography } from "antd";
import { ArrowLeftOutlined, DotChartOutlined } from "@ant-design/icons";

const { Title, Paragraph } = Typography;

interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  row_count: number;
}

export default function EmbeddingsIndex() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("/api/datasets")
      .then((r) => r.json())
      .then((data) => setDatasets(data.datasets || []))
      .catch(() => setDatasets([]))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div
        style={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "calc(100vh - 128px)",
        }}
      >
        <Spin size="large" />
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 24 }}>
        <Link to="/">
          <ArrowLeftOutlined style={{ fontSize: 16, color: "#1890ff" }} />
        </Link>
        <Title level={3} style={{ margin: 0 }}>
          Embeddings
        </Title>
      </div>

      {datasets.length === 0 ? (
        <Empty
          image={<DotChartOutlined style={{ fontSize: 64, color: "#d9d9d9" }} />}
          description={
            <div>
              <Paragraph type="secondary">No datasets available yet.</Paragraph>
              <Paragraph type="secondary" style={{ fontSize: 12 }}>
                Run a classification pipeline to generate embedding visualizations.
              </Paragraph>
            </div>
          }
        />
      ) : (
        <Row gutter={[16, 16]}>
          {datasets.map((ds) => (
            <Col xs={24} sm={12} md={8} key={ds.id}>
              <Link to={`/embeddings/${ds.id}`}>
                <Card hoverable style={{ height: "100%" }}>
                  <Title level={5} style={{ margin: 0 }}>
                    {ds.name}
                  </Title>
                  {ds.description && (
                    <Paragraph
                      type="secondary"
                      style={{ marginTop: 8, marginBottom: 8 }}
                      ellipsis={{ rows: 2 }}
                    >
                      {ds.description}
                    </Paragraph>
                  )}
                  <Tag color="blue">
                    {ds.row_count.toLocaleString()} rows
                  </Tag>
                </Card>
              </Link>
            </Col>
          ))}
        </Row>
      )}
    </div>
  );
}
