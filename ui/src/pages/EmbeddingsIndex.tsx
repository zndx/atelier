import { Link } from "react-router-dom";
import { Card, Col, Empty, Row, Tag, Typography } from "antd";
import { DotChartOutlined } from "@ant-design/icons";
import { useDataset } from "../contexts/DatasetContext";

const { Title, Paragraph } = Typography;

export default function EmbeddingsIndex() {
  const { datasets } = useDataset();

  return (
    <div>
      <Title level={3} style={{ marginBottom: 24 }}>
        Embeddings
      </Title>

      {datasets.length === 0 ? (
        <Empty
          image={<DotChartOutlined style={{ fontSize: 64, color: "var(--text-color-kumo-inactive)" }} />}
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
                    {ds.row_count.toLocaleString()} entities
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
