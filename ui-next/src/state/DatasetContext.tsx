import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import {
  activateArtifactSet,
  listArtifactSets,
  listDataSources,
  listDatasets,
} from "../api/client";
import type { DataSourceInfo, DatasetInfo, MLArtifactSet } from "../api/types";

export interface SmokeTestState {
  result: import("../api/types").SmokeTestResult;
  lastRunAt: number;
}

interface DatasetContextValue {
  sources: DataSourceInfo[];
  activeSourceId: string | null;
  setActiveSourceId: (id: string | null, opts?: { userPicked?: boolean }) => void;
  datasets: DatasetInfo[];
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null, opts?: { userPicked?: boolean }) => void;
  refreshSources: () => Promise<void>;
  refreshDatasets: () => Promise<void>;
  artifactSets: MLArtifactSet[];
  activeArtifactSetId: string | null;
  setActiveArtifactSetId: (id: string | null) => Promise<void>;
  refreshArtifactSets: () => Promise<void>;
  statusPlatformId: string | null;
  setStatusPlatformId: (id: string | null) => void;
  smokeTest: SmokeTestState | null;
  setSmokeTest: (s: SmokeTestState | null) => void;
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

const SOURCE_KEY = "atelier:activeSourceId";
const SOURCE_USER_PICKED_KEY = "atelier:activeSourceIdUserPicked";
const DATASET_KEY = "atelier:activeDatasetId";
const DATASET_USER_PICKED_KEY = "atelier:activeDatasetIdUserPicked";
const STATUS_PLATFORM_KEY = "atelier:statusPlatformId";
const SMOKE_KEY = "atelier:smokeTest";

function readSmokeTest(): SmokeTestState | null {
  try {
    const raw = localStorage.getItem(SMOKE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SmokeTestState;
    if (!parsed || typeof parsed.lastRunAt !== "number") return null;
    return parsed;
  } catch {
    return null;
  }
}

function isEnvSeeded(s: DataSourceInfo): boolean {
  if (!s.metadata) return false;
  try {
    const m = JSON.parse(s.metadata);
    return m?.seeded_from_env === true;
  } catch {
    return false;
  }
}

export function DatasetProvider({ children }: { children: ReactNode }) {
  const [sources, setSources] = useState<DataSourceInfo[]>([]);
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [activeSourceId, setActiveSourceRaw] = useState<string | null>(
    () => localStorage.getItem(SOURCE_KEY),
  );
  const [sourceUserPicked, setSourceUserPicked] = useState<boolean>(
    () => localStorage.getItem(SOURCE_USER_PICKED_KEY) === "1",
  );
  const [activeDatasetId, setActiveDatasetRaw] = useState<string | null>(
    () => localStorage.getItem(DATASET_KEY),
  );
  const [datasetUserPicked, setDatasetUserPicked] = useState<boolean>(
    () => localStorage.getItem(DATASET_USER_PICKED_KEY) === "1",
  );
  const [statusPlatformId, setStatusPlatformRaw] = useState<string | null>(
    () => localStorage.getItem(STATUS_PLATFORM_KEY),
  );
  const [smokeTest, setSmokeTestRaw] = useState<SmokeTestState | null>(() => readSmokeTest());
  const [artifactSets, setArtifactSets] = useState<MLArtifactSet[]>([]);
  const [activeArtifactSetId, setActiveArtifactSetIdRaw] = useState<string | null>(null);

  const setActiveSourceId = useCallback((id: string | null, opts?: { userPicked?: boolean }) => {
    setActiveSourceRaw(id);
    if (id) localStorage.setItem(SOURCE_KEY, id);
    else localStorage.removeItem(SOURCE_KEY);
    if (opts?.userPicked) {
      setSourceUserPicked(true);
      localStorage.setItem(SOURCE_USER_PICKED_KEY, "1");
    }
    // New source = fresh dataset context.
    setActiveDatasetRaw(null);
    localStorage.removeItem(DATASET_KEY);
    setDatasetUserPicked(false);
    localStorage.removeItem(DATASET_USER_PICKED_KEY);
  }, []);

  const setActiveDatasetId = useCallback(
    (id: string | null, opts?: { userPicked?: boolean }) => {
      setActiveDatasetRaw(id);
      if (id) localStorage.setItem(DATASET_KEY, id);
      else localStorage.removeItem(DATASET_KEY);
      if (opts?.userPicked) {
        setDatasetUserPicked(true);
        localStorage.setItem(DATASET_USER_PICKED_KEY, "1");
      }
    },
    [],
  );

  const setStatusPlatformId = useCallback((id: string | null) => {
    setStatusPlatformRaw(id);
    if (id) localStorage.setItem(STATUS_PLATFORM_KEY, id);
    else localStorage.removeItem(STATUS_PLATFORM_KEY);
  }, []);

  const setSmokeTest = useCallback((s: SmokeTestState | null) => {
    setSmokeTestRaw(s);
    if (s) localStorage.setItem(SMOKE_KEY, JSON.stringify(s));
    else localStorage.removeItem(SMOKE_KEY);
  }, []);

  const refreshSources = useCallback(async () => {
    try {
      const data = await listDataSources();
      setSources(data.sources || []);
    } catch {
      setSources([]);
    }
  }, []);

  const refreshDatasets = useCallback(async () => {
    try {
      const data = await listDatasets(activeSourceId);
      setDatasets(data.datasets || []);
    } catch {
      setDatasets([]);
    }
  }, [activeSourceId]);

  const refreshArtifactSets = useCallback(async () => {
    try {
      const data = await listArtifactSets();
      const rows: MLArtifactSet[] = data.artifact_sets || [];
      setArtifactSets(rows);
      const active = rows.find((r) => r.is_active);
      setActiveArtifactSetIdRaw(active ? active.id : null);
    } catch {
      setArtifactSets([]);
      setActiveArtifactSetIdRaw(null);
    }
  }, []);

  const setActiveArtifactSetId = useCallback(
    async (id: string | null) => {
      if (id == null) return;
      try {
        await activateArtifactSet(id);
        await refreshArtifactSets();
      } catch {
        /* next refresh reconciles */
      }
    },
    [refreshArtifactSets],
  );

  useEffect(() => {
    refreshSources();
  }, [refreshSources]);

  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  useEffect(() => {
    refreshArtifactSets();
  }, [refreshArtifactSets]);

  // Auto-select source — env-seeded rows win over sources[0]; explicit
  // operator picks are sticky.
  useEffect(() => {
    if (sources.length === 0) return;
    const current = sources.find((s) => s.id === activeSourceId) ?? null;
    const seeded = sources.find(isEnvSeeded) ?? null;
    if (current == null) {
      setActiveSourceId((seeded ?? sources[0]).id);
      return;
    }
    if (!sourceUserPicked && !isEnvSeeded(current) && seeded != null && seeded.id !== current.id) {
      setActiveSourceId(seeded.id);
    }
  }, [activeSourceId, sources, sourceUserPicked, setActiveSourceId]);

  // Auto-select active dataset version (or most recent).
  useEffect(() => {
    if (datasets.length === 0) return;
    const activeVersion = datasets.find((d) => d.is_active);
    const target = activeVersion ?? datasets[0];
    if (activeDatasetId == null || !datasets.some((d) => d.id === activeDatasetId)) {
      setActiveDatasetId(target.id);
    } else if (!datasetUserPicked && activeVersion && activeVersion.id !== activeDatasetId) {
      setActiveDatasetId(activeVersion.id);
    }
  }, [activeDatasetId, datasets, datasetUserPicked, setActiveDatasetId]);

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
        artifactSets,
        activeArtifactSetId,
        setActiveArtifactSetId,
        refreshArtifactSets,
        statusPlatformId,
        setStatusPlatformId,
        smokeTest,
        setSmokeTest,
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
