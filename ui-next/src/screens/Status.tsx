import { useState } from "react";
import {
  Cpu,
  Database,
  Server,
  Boxes,
  HardDrive,
  Layers,
  Zap,
  RefreshCw,
  CheckCircle2,
} from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import {
  getAcceleration,
  getFsmStatus,
  getStatus,
  getVocabularyStats,
  listDataPlatforms,
  listFsmRuns,
  listTerminalModels,
  setTerminalModel,
  clearTerminalModel,
  activateDataset,
} from "../api/client";
import { useDataset } from "../state/DatasetContext";
import { Card, CardHeader } from "../ui/Card";
import { MetricCard } from "../ui/MetricCard";
import { Pill } from "../ui/Pill";
import { Button } from "../ui/Button";
import { StatusDot, type Tone } from "../ui/StatusDot";
import { Spinner, EmptyState } from "../ui/Feedback";
import { useToast } from "../ui/Toast";
import { ProgressTree } from "../widgets/ProgressTree";
import { RunControls } from "../widgets/RunControls";
import { fsmLabel, stateTone } from "../lib/fsm";

function tone(ok: boolean | undefined): Tone {
  return ok ? "green" : "red";
}

export default function Status() {
  const { push } = useToast();
  const { data: status, refresh: refreshStatus } = usePolling(getStatus, 5000);
  const { data: fsm, refresh: refreshFsm } = usePolling(getFsmStatus, 5000);
  const { data: accel } = usePolling(getAcceleration, 0);
  const { data: runsData, refresh: refreshRuns } = usePolling(listFsmRuns, 10000);
  const { data: platformsData } = usePolling(listDataPlatforms, 0);
  const { data: modelsData, refresh: refreshModels } = usePolling(listTerminalModels, 15000);
  const { data: vocab } = usePolling(() => getVocabularyStats(), 0);

  const {
    datasets,
    activeDatasetId,
    setActiveDatasetId,
    refreshDatasets,
    artifactSets,
    activeArtifactSetId,
    setActiveArtifactSetId,
  } = useDataset();
  const [busyModel, setBusyModel] = useState<string | null>(null);

  const cfg = status?.config ?? {};
  const runs = runsData?.runs ?? [];
  const platforms = platformsData?.platforms ?? [];
  const models = modelsData?.models ?? [];

  const activateVersion = async (id: string) => {
    try {
      await activateDataset(id);
      setActiveDatasetId(id, { userPicked: true });
      await refreshDatasets();
      push("Dataset version activated", "success");
    } catch {
      push("Activation failed", "error");
    }
  };

  const pickModel = async (id: string, isActive: boolean) => {
    setBusyModel(id);
    try {
      if (isActive) await clearTerminalModel();
      else await setTerminalModel(id);
      refreshModels();
    } catch {
      push("Could not change model", "error");
    } finally {
      setBusyModel(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Status</h1>
          <p className="text-sm text-gray-500">Live service health, run control, and registries.</p>
        </div>
        <Button
          size="sm"
          icon={<RefreshCw className="h-3.5 w-3.5" />}
          onClick={() => {
            refreshStatus();
            refreshFsm();
            refreshRuns();
          }}
        >
          Refresh
        </Button>
      </div>

      {/* Health metric row */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <MetricCard
          label="gRPC core"
          value={status?.grpc?.ok ? "up" : "down"}
          tone={tone(status?.grpc?.ok)}
          icon={<Server className="h-3 w-3" />}
        />
        <MetricCard
          label="PostgreSQL"
          value={status?.postgres?.ok ? "up" : "down"}
          tone={tone(status?.postgres?.ok)}
          icon={<Database className="h-3 w-3" />}
        />
        <MetricCard
          label="Qdrant"
          value={status?.qdrant?.ok ? "up" : "down"}
          tone={tone(status?.qdrant?.ok)}
          icon={<Boxes className="h-3 w-3" />}
        />
        <MetricCard
          label="Vocabulary"
          value={vocab?.terms?.toLocaleString() ?? "—"}
          unit="terms"
          icon={<Layers className="h-3 w-3" />}
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* Run control + progress */}
        <Card className="lg:col-span-2">
          <CardHeader title="Classification run" subtitle="Dispatch and monitor the pipeline" />
          <div className="mt-3">
            <RunControls fsm={fsm} onChanged={refreshFsm} />
          </div>
          <div className="mt-4 border-t border-surface-3 pt-3">
            <ProgressTree fsm={fsm} showLineage />
          </div>
        </Card>

        {/* Config + acceleration */}
        <div className="space-y-6">
          <Card>
            <CardHeader icon={<Zap className="h-4 w-4" />} title="Acceleration" />
            <div className="mt-3 space-y-2 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-gray-400">GPU</span>
                <Pill tone={accel?.available ? "green" : "neutral"}>
                  {accel?.available ? accel?.device_name ?? "available" : "CPU only"}
                </Pill>
              </div>
              {accel?.methods &&
                Object.entries(accel.methods).map(([k, v]) => (
                  <div key={k} className="flex items-center justify-between">
                    <span className="font-mono text-xs text-gray-500">{k}</span>
                    <StatusDot tone={v ? "green" : "neutral"} />
                  </div>
                ))}
            </div>
          </Card>

          <Card>
            <CardHeader title="Providers" />
            <div className="mt-3 flex flex-wrap gap-2">
              <Pill tone={cfg.has_anthropic ? "green" : "neutral"}>Anthropic</Pill>
              <Pill tone={cfg.has_bedrock ? "green" : "neutral"}>Bedrock</Pill>
              <Pill tone={cfg.has_classify_llm ? "green" : "amber"}>Classify LLM</Pill>
              <Pill tone={cfg.overwatch_enabled ? "accent" : "neutral"}>Overwatch</Pill>
              <Pill tone={cfg.classify_agent_enabled ? "accent" : "neutral"}>Agent loop</Pill>
            </div>
            {cfg.agent_model && (
              <div className="mt-3 font-mono text-xs text-gray-500">{cfg.agent_model}</div>
            )}
          </Card>
        </div>
      </div>

      {/* Data platforms */}
      <Card>
        <CardHeader icon={<HardDrive className="h-4 w-4" />} title="Data platforms" subtitle={`${platforms.length} configured`} />
        {platforms.length === 0 ? (
          <div className="mt-3">
            <EmptyState title="No data platforms" />
          </div>
        ) : (
          <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
            {platforms.map((p) => (
              <div
                key={p.id}
                className="flex items-center justify-between rounded-md border border-surface-3 bg-surface-1 px-3 py-2"
              >
                <div className="min-w-0">
                  <div className="truncate text-sm text-gray-200">{p.label}</div>
                  <div className="truncate font-mono text-[11px] text-gray-500">
                    {p.table_count ?? 0} tables · {p.column_count ?? 0} cols
                  </div>
                </div>
                <Pill tone="neutral">{p.kind}</Pill>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Dataset versions + Artifact sets */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader title="Dataset versions" subtitle={`${datasets.length} total`} />
          <div className="mt-3 space-y-2">
            {datasets.length === 0 && <EmptyState title="No datasets" />}
            {datasets.map((d) => {
              const active = d.id === activeDatasetId;
              return (
                <div
                  key={d.id}
                  className="flex items-center justify-between rounded-md border border-surface-3 bg-surface-1 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-gray-200">{d.name}</div>
                    <div className="font-mono text-[11px] text-gray-500">
                      v{d.version_number} · {d.row_count?.toLocaleString() ?? "—"} rows
                    </div>
                  </div>
                  {active ? (
                    <Pill tone="green">
                      <CheckCircle2 className="h-3 w-3" /> active
                    </Pill>
                  ) : (
                    <Button size="sm" onClick={() => activateVersion(d.id)}>
                      Activate
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </Card>

        <Card>
          <CardHeader title="ML artifact sets" subtitle={`${artifactSets.length} total`} />
          <div className="mt-3 space-y-2">
            {artifactSets.length === 0 && <EmptyState title="No artifact sets" />}
            {artifactSets.map((a) => {
              const active = a.id === activeArtifactSetId;
              return (
                <div
                  key={a.id}
                  className="flex items-center justify-between rounded-md border border-surface-3 bg-surface-1 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-gray-200">
                      {a.display_name || a.id.slice(0, 12)}
                    </div>
                    <div className="truncate font-mono text-[11px] text-gray-500">
                      {a.embedding_model} · dim {a.embedding_dim}
                    </div>
                  </div>
                  {active ? (
                    <Pill tone="green">
                      <CheckCircle2 className="h-3 w-3" /> active
                    </Pill>
                  ) : (
                    <Button size="sm" onClick={() => setActiveArtifactSetId(a.id)}>
                      Activate
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </Card>
      </div>

      {/* Terminal models */}
      <Card>
        <CardHeader icon={<Cpu className="h-4 w-4" />} title="Agent model catalog" />
        {models.length === 0 ? (
          <div className="mt-3">
            <Spinner />
          </div>
        ) : (
          <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-2">
            {models.map((m) => {
              const isActive = m.id === modelsData?.active;
              return (
                <div
                  key={m.id}
                  className="flex items-center justify-between rounded-md border border-surface-3 bg-surface-1 px-3 py-2"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm text-gray-200">{m.label || m.name || m.id}</div>
                    <div className="truncate font-mono text-[11px] text-gray-500">{m.id}</div>
                  </div>
                  <div className="flex items-center gap-2">
                    {m.available === false && <Pill tone="neutral">unavailable</Pill>}
                    {isActive ? (
                      <Pill tone="accent">active</Pill>
                    ) : (
                      <Button
                        size="sm"
                        disabled={m.available === false || busyModel === m.id}
                        onClick={() => pickModel(m.id, false)}
                      >
                        Use
                      </Button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {modelsData?.override_set && (
          <div className="mt-3">
            <Button size="sm" variant="ghost" onClick={() => pickModel(modelsData.active ?? "", true)}>
              Clear override (use default)
            </Button>
          </div>
        )}
      </Card>

      {/* Run history */}
      <Card>
        <CardHeader title="Run history" subtitle={`${runs.length} runs`} />
        <div className="mt-3 space-y-1.5">
          {runs.length === 0 && <EmptyState title="No runs recorded" />}
          {runs.slice(0, 12).map((r) => (
            <div
              key={r.run_id}
              className="flex items-center justify-between rounded-md border border-surface-3 bg-surface-1 px-3 py-2"
            >
              <div className="min-w-0">
                <div className="truncate font-mono text-xs text-gray-300">{r.run_id}</div>
                <div className="truncate text-[11px] text-gray-500">
                  {r.source_id ?? "—"} · {r.run_kind ?? "classify"}
                </div>
              </div>
              <Pill tone={stateTone(r.state)}>{fsmLabel(r.state)}</Pill>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
