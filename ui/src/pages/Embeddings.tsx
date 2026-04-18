import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingAtlas } from "embedding-atlas/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";
import { Alert, Spin, Typography, Tag } from "antd";
import { SyncOutlined } from "@ant-design/icons";
import type { DatasetInfo } from "../contexts/DatasetContext";

const { Title, Paragraph } = Typography;

const FSM_LABELS: Record<string, string> = {
  LOADING_VOCAB: "Loading vocabulary",
  DISCOVERING: "Discovering tables",
  SAMPLING: "Sampling columns",
  LLM_SWEEP: "Classifying with LLM",
  VALIDATING: "Validating results",
  GENERATING_SYNTH: "Generating synthetic data",
  TRAINING: "Training ML models",
  CLASSIFYING: "Running ML classifiers",
  FUSING: "Fusing evidence",
  EVALUATING: "Evaluating results",
};

export default function Embeddings() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [coordinator] = useState(() => new Coordinator());
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Initializing...");
  const [error, setError] = useState<string | null>(null);
  const [fsmState, setFsmState] = useState<string | null>(null);
  const retryRef = useRef<(() => void) | undefined>(undefined);

  const initialize = useCallback(async (signal: { cancelled: boolean }) => {
    try {
      setError(null);
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
        if (!signal.cancelled) setError(`Dataset "${datasetId}" not found`);
        return;
      }
      if (!signal.cancelled) setDataset(ds);

      // 3. Fetch parquet bytes on main thread (goes through Vite proxy)
      if (!signal.cancelled) setStatus(ds.row_count ? `Downloading ${ds.row_count.toLocaleString()} entities...` : "Downloading data...");
      const parquetResp = await fetch(`/api/datasets/${datasetId}/data`);
      if (!parquetResp.ok) {
        if (!signal.cancelled) {
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
      if (!signal.cancelled) setStatus("Loading data into DuckDB...");
      const db = await connector.getDuckDB();
      const conn = await connector.getConnection();
      const tempFile = "import.parquet";
      await db.registerFileBuffer(tempFile, buffer);

      // 5. Create table with synthetic id column
      await conn.query(
        `CREATE TABLE dataset AS SELECT row_number() OVER () AS id, * FROM '${tempFile}'`
      );
      await db.dropFile(tempFile);

      if (!signal.cancelled) setReady(true);
    } catch (e) {
      console.error("Embeddings init error:", e);
      if (!signal.cancelled) setError(e instanceof Error ? e.message : String(e));
    }
  }, [datasetId, coordinator]);

  // Initial load
  useEffect(() => {
    if (!datasetId) return;
    const signal = { cancelled: false };
    retryRef.current = () => initialize(signal);
    initialize(signal);
    return () => {
      signal.cancelled = true;
      coordinator.clear();
    };
  }, [datasetId, coordinator, initialize]);

  // Poll FSM status when no data — auto-retry when pipeline converges
  useEffect(() => {
    if (error !== "no-data") return;

    let stopped = false;
    const poll = async () => {
      try {
        const resp = await fetch("/api/fsm/status");
        const body = await resp.json();
        const state = body.state as string;
        if (!stopped) setFsmState(state);

        // Pipeline just produced results — retry loading
        if (state === "CONVERGED" && !stopped) {
          stopped = true;
          setFsmState(null);
          retryRef.current?.();
        }
      } catch {
        // gateway not ready yet
      }
    };

    poll();
    const interval = setInterval(poll, 3000);
    return () => { stopped = true; clearInterval(interval); };
  }, [error]);

  if (error) {
    const isNoData = error === "no-data";
    return (
      <div style={{ padding: 24 }}>
        {isNoData ? (
          fsmState && fsmState !== "IDLE" && fsmState !== "ERROR" ? (
            <Alert
              type="info"
              message={
                <span>
                  Pipeline Running{" "}
                  <Tag icon={<SyncOutlined spin />} color="processing">
                    {FSM_LABELS[fsmState] || fsmState}
                  </Tag>
                </span>
              }
              description="Embeddings will load automatically when the pipeline completes."
              showIcon
            />
          ) : (
            <Alert
              type="info"
              message="No Embeddings Data Yet"
              description="Run a classification pipeline on this dataset to generate embeddings for visualization."
              showIcon
            />
          )
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
