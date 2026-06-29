import { useState } from "react";
import { GitBranch, Activity, ScatterChart, TerminalSquare } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { getFsmStatus, getStatus } from "../api/client";
import { useDataset } from "../state/DatasetContext";
import { Terminal } from "../widgets/Terminal";
import { ProgressTree } from "../widgets/ProgressTree";
import { PipelineCanvas } from "../widgets/PipelineCanvas";
import { EmbeddingAtlasView } from "../widgets/EmbeddingAtlasView";
import { RunControls } from "../widgets/RunControls";
import { PanelTabs, type PanelTab } from "../ui/PanelTabs";
import { EmptyState } from "../ui/Feedback";
import type { RuntimeCapabilities } from "../lib/fsmPipelineLayout";

type Panel = "pipeline" | "canvas" | "embeddings";

const TABS: PanelTab<Panel>[] = [
  { id: "pipeline", label: "Pipeline", icon: <Activity /> },
  { id: "canvas", label: "Topology", icon: <GitBranch /> },
  { id: "embeddings", label: "Embeddings", icon: <ScatterChart /> },
];

export default function Operate() {
  const [panel, setPanel] = useState<Panel>("pipeline");
  const { data: fsm, refresh } = usePolling(getFsmStatus, 3000);
  const { data: status } = usePolling(getStatus, 8000);
  const { activeDatasetId } = useDataset();

  const cfg = status?.config ?? {};
  const capabilities: RuntimeCapabilities = {
    cautious_review_enabled: Boolean(cfg.cautious_review_enabled),
    overwatch_enabled: Boolean(cfg.overwatch_enabled),
    classify_agent_enabled: Boolean(cfg.classify_agent_enabled),
  };

  return (
    <div className="flex h-full">
      {/* Left — agent terminal (58%) */}
      <section className="flex w-[58%] min-w-0 flex-col border-r border-surface-3">
        <div className="flex items-center gap-2 border-b border-surface-3 px-4 py-2">
          <TerminalSquare className="h-4 w-4 text-accent" />
          <span className="text-sm font-semibold text-white">Agent</span>
          <span className="text-xs text-gray-500">Claude Agent SDK · persistent session</span>
        </div>
        <div className="min-h-0 flex-1">
          <Terminal />
        </div>
      </section>

      {/* Right — live visualization panel (42%) */}
      <section className="flex w-[42%] min-w-0 flex-col bg-surface-1">
        <div className="space-y-3 border-b border-surface-3 px-4 py-3">
          <RunControls fsm={fsm} onChanged={refresh} />
          <PanelTabs tabs={TABS} active={panel} onChange={setPanel} />
        </div>

        <div className="min-h-0 flex-1 overflow-hidden">
          {panel === "pipeline" && (
            <div className="h-full overflow-y-auto p-4">
              <ProgressTree fsm={fsm} showLineage />
            </div>
          )}
          {panel === "canvas" && (
            <PipelineCanvas state={fsm?.state ?? null} capabilities={capabilities} />
          )}
          {panel === "embeddings" &&
            (activeDatasetId ? (
              <EmbeddingAtlasView datasetId={activeDatasetId} />
            ) : (
              <div className="p-4">
                <EmptyState
                  icon={<ScatterChart />}
                  title="No dataset selected"
                  description="Select a data source and complete a run to populate the embedding projection."
                />
              </div>
            ))}
        </div>
      </section>
    </div>
  );
}
