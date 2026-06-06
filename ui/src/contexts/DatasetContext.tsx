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
  // Lineage for Extend Classification (added 20260427).  classify runs
  // set artifact_set_id to the row they produced; extend runs set it
  // to the row they consumed plus parent_dataset_id linking to the
  // source classify dataset.
  artifact_set_id?: string | null;
  parent_dataset_id?: string | null;
  run_kind?: string;
}

export interface MLArtifactSet {
  id: string;
  source_id: string | null;
  fsm_run_id: string | null;
  parent_artifact_set_id: string | null;
  catboost_path: string;
  catboost_classes_path: string;
  svm_path: string | null;
  svm_classes_path: string | null;
  umap_path: string | null;
  classes: string;            // JSON-encoded list
  feature_groups: string | null;
  vocab_signature: string;
  embedding_model: string;
  embedding_dim: number;
  display_name: string | null;
  summary: string | null;
  is_active: boolean;
  is_archived: boolean;
  facets: string | null;
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
  // Set the active dataset.  Pass ``opts.userPicked = true`` when the
  // operator made an explicit selection (e.g. via the version
  // activation button) — the auto-promote-to-server-active effect
  // backs off when this flag is set, preserving manual choices that
  // would otherwise be reverted by the next render.
  setActiveDatasetId: (id: string | null, opts?: { userPicked?: boolean }) => void;
  refreshSources: () => Promise<void>;
  refreshDatasets: () => Promise<void>;
  // ML Artifact Sets (Extend Classification, added 20260427).
  artifactSets: MLArtifactSet[];
  activeArtifactSetId: string | null;
  setActiveArtifactSetId: (id: string | null) => Promise<void>;
  refreshArtifactSets: () => Promise<void>;
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
const DATASET_USER_PICKED_KEY = "atelier:activeDatasetIdUserPicked";
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
  // Mirror of ``sourceUserPicked`` for datasets — distinguishes an
  // explicit operator activation (DataSourceCard's activateVersion)
  // from an auto-pick.  The auto-promote-to-server-active effect only
  // fires when this flag is *not* set, so manual selections aren't
  // reverted by the next render.  Cleared whenever the source changes
  // (new source = fresh dataset selection context).
  const [datasetUserPicked, setDatasetUserPicked] = useState<boolean>(
    () => localStorage.getItem(DATASET_USER_PICKED_KEY) === "1",
  );
  const [statusPlatformId, setStatusPlatformRaw] = useState<string | null>(
    () => localStorage.getItem(STATUS_PLATFORM_KEY),
  );
  const [smokeTest, setSmokeTestRaw] = useState<SmokeTestState | null>(
    () => readSmokeTest(),
  );
  const [artifactSets, setArtifactSets] = useState<MLArtifactSet[]>([]);
  const [activeArtifactSetId, setActiveArtifactSetIdRaw] = useState<string | null>(
    null,  // server-side state — derived from artifactSets[].is_active on refresh
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
    // Clear stale dataset selection — refreshDatasets will auto-select.
    // Also clear the dataset-userPicked sticky bit: a new source means a
    // fresh dataset context, and the auto-promote should be free to run
    // on the new source's is_active version.
    setActiveDatasetRaw(null);
    localStorage.removeItem(DATASET_KEY);
    setDatasetUserPicked(false);
    localStorage.removeItem(DATASET_USER_PICKED_KEY);
  }, []);

  const setActiveDatasetId = useCallback(
    (id: string | null, opts?: { userPicked?: boolean }) => {
      setActiveDatasetRaw(id);
      if (id) {
        localStorage.setItem(DATASET_KEY, id);
      } else {
        localStorage.removeItem(DATASET_KEY);
      }
      if (opts?.userPicked) {
        setDatasetUserPicked(true);
        localStorage.setItem(DATASET_USER_PICKED_KEY, "1");
      }
    },
    [],
  );

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

  const refreshArtifactSets = useCallback(async () => {
    try {
      const r = await fetch("/api/artifact-sets");
      const data = await r.json();
      const rows: MLArtifactSet[] = data.artifact_sets || [];
      setArtifactSets(rows);
      // Mirror server-side is_active into the client-side selector so
      // panels can render `record.is_active` directly without an extra
      // bookkeeping layer.
      const active = rows.find((r) => r.is_active);
      setActiveArtifactSetIdRaw(active ? active.id : null);
    } catch {
      setArtifactSets([]);
      setActiveArtifactSetIdRaw(null);
    }
  }, []);

  const setActiveArtifactSetId = useCallback(async (id: string | null) => {
    if (id == null) return;
    try {
      const r = await fetch(
        `/api/artifact-sets/${encodeURIComponent(id)}/activate`,
        { method: "POST" },
      );
      if (r.ok) {
        await refreshArtifactSets();
      }
    } catch {
      // Swallow — the next refresh will reconcile.
    }
  }, [refreshArtifactSets]);

  // Fetch sources on mount
  useEffect(() => {
    refreshSources();
  }, [refreshSources]);

  // Fetch datasets when source changes
  useEffect(() => {
    refreshDatasets();
  }, [refreshDatasets]);

  // Fetch artifact sets on mount.  Status.tsx's FSM convergence
  // handler also calls refreshArtifactSets so a freshly-completed
  // classify run shows up in the panel without a manual refresh.
  useEffect(() => {
    refreshArtifactSets();
  }, [refreshArtifactSets]);

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

  // Auto-select active dataset version (or most recent).
  //
  // Two separate concerns, each guarded:
  //  1. No stored selection (or stored ID no longer in list) → pick the
  //     best available.  Always safe.
  //  2. A different version is server-marked is_active and the operator
  //     has NOT pinned a manual selection (``datasetUserPicked``) →
  //     promote to it.  Without the userPicked guard this fires after
  //     every ``setActiveDatasetId`` mutation while ``datasets`` still
  //     carries a stale ``is_active`` flag, which causes
  //     ``activateVersion`` clicks to flicker-revert to the previous
  //     active row.  The guard mirrors the ``sourceUserPicked`` pattern
  //     above; ``setActiveSourceId`` clears it on source change so a
  //     fresh source can promote freely on first load.
  useEffect(() => {
    if (datasets.length === 0) return;
    const activeVersion = datasets.find((d) => d.is_active);
    const target = activeVersion ?? datasets[0];
    if (
      activeDatasetId == null ||
      !datasets.some((d) => d.id === activeDatasetId)
    ) {
      setActiveDatasetId(target.id);
    } else if (
      !datasetUserPicked &&
      activeVersion &&
      activeVersion.id !== activeDatasetId
    ) {
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
