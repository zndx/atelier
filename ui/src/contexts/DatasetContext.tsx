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

export interface SmokeTestResult {
  success: boolean;
  reply?: string;
  duration_ms?: number;
  session_id?: string;
  total_cost_usd?: number;
  retried?: boolean;
  error?: string;
}

export interface SmokeTestState {
  result: SmokeTestResult;
  lastRunAt: number; // Date.now() milliseconds
}

interface DatasetContextValue {
  sources: DataSourceInfo[];
  activeSourceId: string | null;
  setActiveSourceId: (id: string | null, opts?: { userPicked?: boolean }) => void;
  datasets: DatasetInfo[];
  activeDatasetId: string | null;
  setActiveDatasetId: (id: string | null) => void;
  refreshSources: () => Promise<void>;
  refreshDatasets: () => Promise<void>;
  // Status-page state that should survive navigation.
  statusPlatformId: string | null;
  setStatusPlatformId: (id: string | null) => void;
  smokeTest: SmokeTestState | null;
  setSmokeTest: (s: SmokeTestState | null) => void;
}

const DatasetContext = createContext<DatasetContextValue | null>(null);

const SOURCE_KEY = "atelier:activeSourceId";
const SOURCE_USER_PICKED_KEY = "atelier:activeSourceIdUserPicked";
const DATASET_KEY = "atelier:activeDatasetId";
const STATUS_PLATFORM_KEY = "atelier:statusPlatformId";
const SMOKE_KEY = "atelier:smokeTest";

function readSmokeTest(): SmokeTestState | null {
  try {
    const raw = localStorage.getItem(SMOKE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as SmokeTestState;
    // Sanity check — stored state must have both fields.
    if (!parsed || typeof parsed.lastRunAt !== "number") return null;
    return parsed;
  } catch {
    return null;
  }
}

function isEnvSeeded(s: DataSourceInfo): boolean {
  // metadata is a JSON string; seeder stamps {"seeded_from_env": true}.
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
  // Persist whether the activeSourceId was set by an explicit operator
  // click vs. an auto-pick on first load.  Without this distinction,
  // an early auto-pick of `ootb-sample` (the first source to seed)
  // gets written to localStorage and then blocks the later-arriving
  // env-seeded `hive-poc/default` from taking precedence when the
  // hive discovery seed completes.
  const [sourceUserPicked, setSourceUserPicked] = useState<boolean>(
    () => localStorage.getItem(SOURCE_USER_PICKED_KEY) === "1",
  );
  const [activeDatasetId, setActiveDatasetRaw] = useState<string | null>(
    () => localStorage.getItem(DATASET_KEY),
  );
  const [statusPlatformId, setStatusPlatformRaw] = useState<string | null>(
    () => localStorage.getItem(STATUS_PLATFORM_KEY),
  );
  const [smokeTest, setSmokeTestRaw] = useState<SmokeTestState | null>(
    () => readSmokeTest(),
  );

  const setActiveSourceId = useCallback((id: string | null, opts?: { userPicked?: boolean }) => {
    setActiveSourceRaw(id);
    if (id) {
      localStorage.setItem(SOURCE_KEY, id);
    } else {
      localStorage.removeItem(SOURCE_KEY);
    }
    // Mark operator-initiated selections so the env-seeded
    // preference cannot hijack an explicit choice on subsequent
    // sources polls (see the auto-select effect below for the race
    // this guards against).
    if (opts?.userPicked) {
      setSourceUserPicked(true);
      localStorage.setItem(SOURCE_USER_PICKED_KEY, "1");
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

  const setStatusPlatformId = useCallback((id: string | null) => {
    setStatusPlatformRaw(id);
    if (id) {
      localStorage.setItem(STATUS_PLATFORM_KEY, id);
    } else {
      localStorage.removeItem(STATUS_PLATFORM_KEY);
    }
  }, []);

  const setSmokeTest = useCallback((s: SmokeTestState | null) => {
    setSmokeTestRaw(s);
    if (s) {
      localStorage.setItem(SMOKE_KEY, JSON.stringify(s));
    } else {
      localStorage.removeItem(SMOKE_KEY);
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

  // Auto-select source.  Env-seeded rows win over the default
  // `sources[0]` so fresh CAI deploys land on the operator-intended
  // classification target without a click.  Explicit operator picks
  // (``setActiveSourceId(id, { userPicked: true })``) are sticky and
  // short-circuit this path on subsequent polls.
  //
  // Timing hazard this guards against: on a fresh CAI deploy the
  // OOTB sample source seeds first (local fixtures, succeeds
  // immediately), while the env-seeded ``hive-poc/default`` row
  // requires the Hive discovery round-trip and only appears a few
  // seconds later.  Without the re-evaluation below, the UI auto-
  // picks ootb-sample on first poll, writes it to localStorage, and
  // the later-arriving env-seeded row never takes precedence.
  useEffect(() => {
    if (sources.length === 0) return;
    const current = sources.find((s) => s.id === activeSourceId) ?? null;
    const seeded = sources.find(isEnvSeeded) ?? null;

    // No active source OR the current selection has been removed
    // from the source list → pick the best available.
    if (current == null) {
      setActiveSourceId((seeded ?? sources[0]).id);
      return;
    }
    // Promote to env-seeded when one exists AND the operator hasn't
    // made an explicit sticky choice.  Protects against the seed-
    // race without overriding deliberate selections.
    if (!sourceUserPicked && !isEnvSeeded(current) && seeded != null && seeded.id !== current.id) {
      setActiveSourceId(seeded.id);
    }
  }, [activeSourceId, sources, sourceUserPicked, setActiveSourceId]);

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
