import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingAtlas } from "embedding-atlas/react";
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Alert, Spin, Typography, Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";

const { Title, Paragraph } = Typography;

interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  parquet_path: string;
  row_count: number;
}

export default function EmbeddingsViewer() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [coordinator, setCoordinator] = useState<Coordinator | null>(null);
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!datasetId) return;

    async function initialize() {
      try {
        // Fetch dataset metadata
        const resp = await fetch("/api/datasets");
        const data = await resp.json();
        const ds = data.datasets.find(
          (d: DatasetInfo) => d.id === datasetId
        );

        if (!ds) {
          setError(`Dataset "${datasetId}" not found`);
          setLoading(false);
          return;
        }
        setDataset(ds);

        // Initialize DuckDB WASM via Mosaic
        const wasm = await wasmConnector();
        const coord = new Coordinator();
        coord.databaseConnector(wasm);

        // Fetch parquet and load into DuckDB
        const parquetResp = await fetch(`/api/datasets/${datasetId}/data`);
        if (!parquetResp.ok) {
          setError("Failed to fetch parquet data");
          setLoading(false);
          return;
        }
        const buffer = await parquetResp.arrayBuffer();

        // Register the parquet buffer as a table
        await coord.exec(
          `CREATE TABLE dataset AS SELECT * FROM read_parquet('buffer')`,
          { buffer: new Uint8Array(buffer) }
        );

        setCoordinator(coord);
        setLoading(false);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      }
    }

    initialize();
  }, [datasetId]);

  if (error) {
    return (
      <div style={{ padding: 24 }}>
        <Link to="/">
          <Button icon={<ArrowLeftOutlined />} style={{ marginBottom: 16 }}>
            Back
          </Button>
        </Link>
        <Alert type="error" message="Error" description={error} showIcon />
      </div>
    );
  }

  if (loading || !coordinator) {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          height: "calc(100vh - 128px)",
          gap: 16,
        }}
      >
        <Spin size="large" />
        <Paragraph type="secondary">
          Loading {dataset?.name || datasetId}...
        </Paragraph>
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 128px)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 12 }}>
        <Link to="/">
          <Button icon={<ArrowLeftOutlined />} size="small">
            Back
          </Button>
        </Link>
        <Title level={4} style={{ margin: 0 }}>
          {dataset?.name || datasetId}
        </Title>
        {dataset?.description && (
          <Paragraph type="secondary" style={{ margin: 0 }}>
            {dataset.description}
          </Paragraph>
        )}
      </div>
      <div style={{ flex: 1 }}>
        <EmbeddingAtlas
          coordinator={coordinator}
          data={{
            table: "dataset",
            id: "id",
            text: "text",
            projection: { x: "x", y: "y" },
          }}
          colorScheme="light"
        />
      </div>
    </div>
  );
}
