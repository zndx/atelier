import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export interface DataSourceInfo {
  id: string;
  source_type: string;
  source_uri: string;
  display_name: string;
  vocabulary_mode: string;
  created_at: string;
  metadata: string;
}

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  parquet_path: string;
  row_count: number;
  source_id: string;
  version_number: number;
  is_active: boolean;
  summary: string;
  fsm_run_id: string;
  created_at: string;
}

interface DatasetContextValue {
  sources: DataSourceInfo[];
  activeSourceId: string | null;
  setActiveSourceId: (id: string | null) => void;
  datasets: DatasetInfo[];
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
  refreshSources: () => Promise<void>;
  refreshDatasets: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

const SOURCE_KEY = "atelier:activeSourceId";
const DATASET_KEY = "atelier:activeDatasetId";

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<DataSourceInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [activeSourceId, setActiveSourceRaw] = useState<string | null>(
    () => localStorage.getItem(SOURCE_KEY),
  );
  const [activeDatasetId, setActiveDatasetRaw] = useState<string | null>(
    () => localStorage.getItem(DATASET_KEY),
  );

  const setActiveSourceId = useCallback((id: string | null) => {
    setActiveSourceRaw(id);
    if (id) {
      localStorage.setItem(SOURCE_KEY, id);
    } else {
      localStorage.removeItem(SOURCE_KEY);
    }
    // Clear stale dataset selection — refreshDatasets will auto-select
    setActiveDatasetRaw(null);
    localStorage.removeItem(DATASET_KEY);
  }, []);

  const setActiveDatasetId = useCallback((id: string | null) => {
    setActiveDatasetRaw(id);
    if (id) {
      localStorage.setItem(DATASET_KEY, id);
    } else {
      localStorage.removeItem(DATASET_KEY);
    }
  }, []);

  const refreshSources = useCallback(async () => {
    try {
      const r = await fetch("/api/data-sources");
      const data = await r.json();
      setSources(data.sources || []);
    } catch {
      setSources([]);
    }
  }, []);

  const refreshDatasets = useCallback(async () => {
    try {
      const url = activeSourceId
        ? `/api/datasets?source_id=${encodeURIComponent(activeSourceId)}`
        : "/api/datasets";
      const r = await fetch(url);
      const data = await r.json();
      setDatasets(data.datasets || []);
    } catch {
      setDatasets([]);
    }
  }, [activeSourceId]);

  // Fetch sources on mount
  useEffect(() => {
    refreshSources();
  }, [refreshSources]);

  // Fetch datasets when source changes
  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Auto-select first source if none active
  useEffect(() => {
    if (sources.length === 0) return;
    if (activeSourceId == null || !sources.some((s) => s.id === activeSourceId)) {
      setActiveSourceId(sources[0].id);
    }
  }, [activeSourceId, sources, setActiveSourceId]);

  // Auto-select active dataset version (or most recent)
  useEffect(() => {
    if (datasets.length === 0) return;
    const activeVersion = datasets.find((d) => d.is_active);
    const target = activeVersion ?? datasets[0];
    if (
      activeDatasetId == null ||
      !datasets.some((d) => d.id === activeDatasetId)
    ) {
      setActiveDatasetId(target.id);
    }
  }, [activeDatasetId, datasets, setActiveDatasetId]);

  return (
    <DatasetContext.Provider
      value={{
        sources,
        activeSourceId,
        setActiveSourceId,
        datasets,
        activeDatasetId,
        setActiveDatasetId,
        refreshSources,
        refreshDatasets,
      }}
    >
      {children}
    </DatasetContext.Provider>
  );
}

export function useDataset(): DatasetContextValue {
  const ctx = useContext(DatasetContext);
  if (!ctx) throw new Error("useDataset must be used within DatasetProvider");
  return ctx;
}
