import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export interface DatasetInfo {
  id: string;
  name: string;
  description: string;
  row_count: number;
}

interface DatasetContextValue {
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
  datasets: DatasetInfo[];
  refreshDatasets: () => Promise<void>;
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

const STORAGE_KEY = "atelier:activeDatasetId";

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [activeDatasetId, setActiveRaw] = useState<string | null>(() =>
    localStorage.getItem(STORAGE_KEY),
  );

  const setActiveDatasetId = useCallback((id: string | null) => {
    setActiveRaw(id);
    if (id) {
      localStorage.setItem(STORAGE_KEY, id);
    } else {
      localStorage.removeItem(STORAGE_KEY);
    }
  }, []);

  const refreshDatasets = useCallback(async () => {
    try {
      const r = await fetch("/api/datasets");
      const data = await r.json();
      setDatasets(data.datasets || []);
    } catch {
      setDatasets([]);
    }
  }, []);

  // Fetch on mount
  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Auto-select most recent when none active; clear stale IDs
  useEffect(() => {
    if (datasets.length === 0) return;

    if (activeDatasetId == null) {
      setActiveDatasetId(datasets[datasets.length - 1].id);
    } else if (!datasets.some((d) => d.id === activeDatasetId)) {
      setActiveDatasetId(datasets[datasets.length - 1].id);
    }
  }, [activeDatasetId, datasets, setActiveDatasetId]);

  return (
    <DatasetContext.Provider
      value={{ activeDatasetId, setActiveDatasetId, datasets, refreshDatasets }}
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
