import { Coordinator, wasmConnector } from "@uwdata/mosaic-core";
import { EmbeddingAtlas } from "embedding-atlas/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { fetchDatasetParquet, getFsmStatus, listDatasets } from "../api/client";
import type { DatasetInfo } from "../api/types";
import { Spinner, EmptyState, Banner } from "../ui/Feedback";
import { Pill } from "../ui/Pill";
import { StatusDot } from "../ui/StatusDot";
import { fsmLabel, isRunning, stateTone } from "../lib/fsm";
import { Boxes } from "lucide-react";

// Curated predicate panel — ordered for an algo-tuning reviewer.
const INCLUDE_COLUMNS = [
  "belief",
  "confidence",
  "review_decision",
  "predicted_annotation",
  "needs_clarification",
  "llm_confidence",
  "uncertainty",
  "conflict",
  "shap_top1_name",
  "shap_top1_value",
  "shap_top2_name",
  "shap_top2_value",
  "shap_top3_name",
  "shap_top3_value",
  "table_name",
  "column_type",
];

export function EmbeddingAtlasView({ datasetId }: { datasetId: string }) {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [coordinator] = useState(() => new Coordinator());
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState("Initializing…");
  const [error, setError] = useState<string | null>(null);
  const [fsmState, setFsmState] = useState<string | null>(null);
  const retryRef = useRef<(() => void) | undefined>(undefined);

  const initialize = useCallback(
    async (signal: { cancelled: boolean }) => {
      try {
        setError(null);
        setStatus("Initializing DuckDB…");
        const connector = wasmConnector();
        coordinator.databaseConnector(connector);

        setStatus("Fetching dataset metadata…");
        const data = await listDatasets();
        const ds = data.datasets.find((d) => d.id === datasetId);
        if (!ds) {
          if (!signal.cancelled) setError(`Dataset "${datasetId}" not found`);
          return;
        }
        if (!signal.cancelled) setDataset(ds);

        if (!signal.cancelled)
          setStatus(
            ds.row_count
              ? `Downloading ${ds.row_count.toLocaleString()} entities…`
              : "Downloading data…",
          );
        const parquetResp = await fetchDatasetParquet(datasetId);
        if (!parquetResp.ok) {
          if (!signal.cancelled)
            setError(parquetResp.status === 404 ? "no-data" : `HTTP ${parquetResp.status}`);
          return;
        }
        const buffer = new Uint8Array(await parquetResp.arrayBuffer());

        if (!signal.cancelled) setStatus("Loading data into DuckDB…");
        const db = await connector.getDuckDB();
        const conn = await connector.getConnection();
        const tempFile = "import.parquet";
        await db.registerFileBuffer(tempFile, buffer);
        await conn.query(
          `CREATE TABLE dataset AS SELECT row_number() OVER () AS id, * FROM '${tempFile}'`,
        );
        await db.dropFile(tempFile);

        if (!signal.cancelled) setReady(true);
      } catch (e) {
        if (!signal.cancelled) setError(e instanceof Error ? e.message : String(e));
      }
    },
    [datasetId, coordinator],
  );

  useEffect(() => {
    const signal = { cancelled: false };
    retryRef.current = () => {
      setReady(false);
      initialize(signal);
    };
    initialize(signal);
    return () => {
      signal.cancelled = true;
      coordinator.clear();
    };
  }, [datasetId, coordinator, initialize]);

  // While no data, poll FSM and auto-retry on CONVERGED.
  useEffect(() => {
    if (error !== "no-data") return;
    let stopped = false;
    const poll = async () => {
      try {
        const body = await getFsmStatus();
        const state = body.state;
        if (!stopped) setFsmState(state);
        if (state === "CONVERGED" && !stopped) {
          stopped = true;
          setFsmState(null);
          retryRef.current?.();
        }
      } catch {
        /* gateway not ready */
      }
    };
    poll();
    const interval = setInterval(poll, 3000);
    return () => {
      stopped = true;
      clearInterval(interval);
    };
  }, [error]);

  if (error) {
    if (error === "no-data") {
      return (
        <div className="p-6">
          {isRunning(fsmState) ? (
            <Banner tone="info">
              <div className="flex items-center gap-2">
                <span>Pipeline running</span>
                <Pill tone={stateTone(fsmState)}>
                  <StatusDot tone={stateTone(fsmState)} pulse />
                  {fsmLabel(fsmState)}
                </Pill>
              </div>
              <div className="mt-1 text-xs text-gray-500">
                Embeddings will load automatically when the run completes.
              </div>
            </Banner>
          ) : (
            <EmptyState
              icon={<Boxes />}
              title="No embeddings data yet"
              description="Run a classification pipeline on this dataset to generate the embedding projection."
            />
          )}
        </div>
      );
    }
    return (
      <div className="p-6">
        <Banner tone="error">{error}</Banner>
      </div>
    );
  }

  if (!ready) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner label={status} />
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center gap-2 border-b border-surface-3 px-4 py-2">
        <span className="text-sm font-semibold text-white">{dataset?.name || datasetId}</span>
        {dataset?.row_count ? (
          <span className="font-mono text-xs text-gray-500">
            {dataset.row_count.toLocaleString()} entities
          </span>
        ) : null}
      </div>
      <div className="min-h-0 flex-1">
        <EmbeddingAtlas
          coordinator={coordinator}
          data={{ table: "dataset", id: "id", text: "text", projection: { x: "x", y: "y" } }}
          defaultChartsConfig={{
            embedding: { data: { x: "x", y: "y", text: "text", category: "belief" } },
            include: INCLUDE_COLUMNS,
          }}
          embeddingViewConfig={{ autoLabelEnabled: true }}
          colorScheme="dark"
        />
      </div>
    </div>
  );
}
