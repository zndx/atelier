import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingAtlas } from "embedding-atlas/react";
import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Alert, Spin, Typography, Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import type { DatasetInfo } from "../contexts/DatasetContext";

const { Title, Paragraph } = Typography;

export default function Embeddings() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [coordinator] = useState(() => new Coordinator());
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!datasetId) return;
    let cancelled = false;

    async function initialize() {
      try {
        // 1. Initialize DuckDB via Mosaic wasmConnector
        setStatus("Initializing DuckDB...");
        const connector = wasmConnector();
        coordinator.databaseConnector(connector);

        // 2. Fetch dataset metadata
        setStatus("Fetching dataset metadata...");
        const resp = await fetch("/api/datasets");
        const data = await resp.json();
        const ds = data.datasets.find(
          (d: DatasetInfo) => d.id === datasetId
        );
        if (!ds) {
          if (!cancelled) setError(`Dataset "${datasetId}" not found`);
          return;
        }
        if (!cancelled) setDataset(ds);

        // 3. Fetch parquet bytes on main thread (goes through Vite proxy)
        if (!cancelled) setStatus(ds.row_count ? `Downloading ${ds.row_count.toLocaleString()} rows...` : "Downloading data...");
        const parquetResp = await fetch(`/api/datasets/${datasetId}/data`);
        if (!parquetResp.ok) {
          if (!cancelled) {
            if (parquetResp.status === 404) {
              setError("no-data");
            } else {
              setError(`Failed to fetch parquet: HTTP ${parquetResp.status}`);
            }
          }
          return;
        }
        const buffer = new Uint8Array(await parquetResp.arrayBuffer());

        // 4. Register parquet bytes into DuckDB virtual filesystem
        if (!cancelled) setStatus("Loading data into DuckDB...");
        const db = await connector.getDuckDB();
        const conn = await connector.getConnection();
        const tempFile = "import.parquet";
        await db.registerFileBuffer(tempFile, buffer);

        // 5. Create table with synthetic id column
        await conn.query(
          `CREATE TABLE dataset AS SELECT row_number() OVER () AS id, * FROM '${tempFile}'`
        );
        await db.dropFile(tempFile);

        if (!cancelled) setReady(true);
      } catch (e) {
        console.error("Embeddings init error:", e);
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    }

    initialize();

    return () => {
      cancelled = true;
      coordinator.clear();
    };
  }, [datasetId, coordinator]);

  if (error) {
    const isNoData = error === "no-data";
    return (
      <div style={{ padding: 24 }}>
        <Link to="/">
          <Button icon={<ArrowLeftOutlined />} style={{ marginBottom: 16 }}>
            Back
          </Button>
        </Link>
        {isNoData ? (
          <Alert
            type="info"
            message="No Embeddings Data Yet"
            description="Run a classification pipeline on this dataset to generate embeddings for visualization."
            showIcon
          />
        ) : (
          <Alert type="error" message="Error" description={error} showIcon />
        )}
      </div>
    );
  }

  if (!ready) {
    return (
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 16,
        }}
      >
        <Spin size="large" />
        <Paragraph type="secondary">{status}</Paragraph>
      </div>
    );
  }

  return (
    <div
      style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          flex: "none",
          display: "flex",
          alignItems: "center",
          gap: 12,
          padding: "8px 8px",
        }}
      >
        <Link to="/">
          <Button icon={<ArrowLeftOutlined />} size="small">
            Back
          </Button>
        </Link>
        <Title level={5} style={{ margin: 0 }}>
          {dataset?.name || datasetId}
        </Title>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        <EmbeddingAtlas
          coordinator={coordinator}
          data={{
            table: "dataset",
            id: "id",
            text: "text",
            projection: { x: "x", y: "y" },
          }}
          defaultChartsConfig={{
            embedding: { data: { x: "x", y: "y", text: "text", category: "conflict" } },
          }}
          embeddingViewConfig={{ autoLabelEnabled: true }}
          colorScheme="light"
        />
      </div>
    </div>
  );
}
